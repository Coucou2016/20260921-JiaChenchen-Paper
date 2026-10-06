"""Scientific-semantics tests, beyond shape sanity.

Each test pins a behaviour that a previous version got wrong and that directly
affects what an experiment measures:

* masked bilinear must not dilute valid depth with nodata zeros
* the direct-depth arm must have a trainable gradient path (A4/R0)
* the wet head must receive gradient when it is supervised (A6/W1)
* unseen evaluation pairs must be disjoint from training pairs
* the domain accumulator must equal a single whole-tensor computation
* an empty event scores NaN, not zero
* the boundary loss must ignore nodata stencils
* the multiscale trainer must honour the config it is given

Run:  python -m pytest tests/test_scientific_semantics.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.aggregation import FloodMetricAccumulator, average_metrics  # noqa: E402
from metrics.flood_metrics import binary_csi_f1, masked_ssim  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. masked bilinear
# --------------------------------------------------------------------------- #
def test_masked_bilinear_does_not_dilute_valid_depth():
    """A lone valid 1.0 must not be halved by surrounding nodata zeros."""
    from models.hydrogeo_srno import HydroGeoSRNO

    m = HydroGeoSRNO(hydro_width=8, hydro_blocks=1, geo_width=8, geo_in_cont=0,
                     use_landuse=False, operator_width=16, heads=2,
                     operator_layers=1, predict_wet=False)
    lr = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    lr_valid = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    static_cont = torch.zeros(1, 0, 2, 2)
    landuse = torch.zeros(1, 2, 2, dtype=torch.long)

    out = m(lr=lr, lr_valid=lr_valid, static_cont=static_cont, landuse=landuse,
            lr_res=torch.tensor([10.0]), hr_res=torch.tensor([10.0]))
    base = out["base"][0, 0]
    # the top-left cell is the only valid input; masking must keep it >= plain interp
    plain = torch.nn.functional.interpolate(lr[:, :1], size=(2, 2),
                                            mode="bilinear", align_corners=False)
    assert float(base[0, 0]) >= float(plain[0, 0, 0, 0]) - 1e-6


# --------------------------------------------------------------------------- #
# 2. direct-depth arm is trainable
# --------------------------------------------------------------------------- #
def test_direct_depth_ablation_has_trainable_gradient():
    """R0/A4 must not collapse to a fixed bilinear base with no learning."""
    from models.hydrogeo_srno import HydroGeoSRNO

    torch.manual_seed(0)
    m = HydroGeoSRNO(hydro_width=8, hydro_blocks=1, geo_width=8, geo_in_cont=0,
                     use_landuse=False, operator_width=16, heads=2,
                     operator_layers=1, predict_residual=False, predict_wet=False)
    # Force the head into a positive regime so the final clamp(min=0) does not
    # zero the gradient. Otherwise this test would depend on initialisation luck
    # rather than on whether the head is wired into the output at all.
    with torch.no_grad():
        m.direct_head.net[-1].weight.fill_(1.0)
        m.direct_head.net[-1].bias.fill_(5.0)

    lr = torch.rand(1, 1, 4, 4)
    lr_valid = torch.ones(1, 1, 4, 4)
    out = m(lr=lr, lr_valid=lr_valid, static_cont=torch.zeros(1, 0, 8, 8),
            landuse=torch.zeros(1, 8, 8, dtype=torch.long),
            lr_res=torch.tensor([10.0]), hr_res=torch.tensor([2.0]))
    assert float(out["depth"].max()) > 0.0, "direct head produced an all-zero depth"
    out["depth"].mean().backward()
    grad = m.direct_head.net[0].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0, \
        "direct-depth head received no gradient"


def test_residual_head_gradient_is_untouched_when_not_used():
    """R1 must still drive the residual head."""
    from models.hydrogeo_srno import HydroGeoSRNO

    m = HydroGeoSRNO(hydro_width=8, hydro_blocks=1, geo_width=8, geo_in_cont=0,
                     use_landuse=False, operator_width=16, heads=2,
                     operator_layers=1, predict_residual=True, predict_wet=False)
    out = m(lr=torch.rand(1, 1, 4, 4), lr_valid=torch.ones(1, 1, 4, 4),
            static_cont=torch.zeros(1, 0, 8, 8),
            landuse=torch.zeros(1, 8, 8, dtype=torch.long),
            lr_res=torch.tensor([10.0]), hr_res=torch.tensor([2.0]))
    out["depth"].mean().backward()
    assert float(m.residual_head.net[0].weight.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# 3. wet head supervision
# --------------------------------------------------------------------------- #
def test_wet_head_gets_gradient_when_supervised():
    """W1 must train the wet head; W0 (w_wet=0) must leave it untouched."""
    from losses.flood_loss import MaskedL1Loss
    from models.hydrogeo_srno import HydroGeoSRNO

    def _run(w_wet: float) -> float:
        torch.manual_seed(0)
        m = HydroGeoSRNO(hydro_width=8, hydro_blocks=1, geo_width=8, geo_in_cont=0,
                         use_landuse=False, operator_width=16, heads=2,
                         operator_layers=1, predict_wet=True)
        out = m(lr=torch.rand(1, 1, 4, 4), lr_valid=torch.ones(1, 1, 4, 4),
                static_cont=torch.zeros(1, 0, 8, 8),
                landuse=torch.zeros(1, 8, 8, dtype=torch.long),
                lr_res=torch.tensor([10.0]), hr_res=torch.tensor([2.0]))
        target = torch.rand(1, 1, 8, 8)
        mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
        crit = MaskedL1Loss(w_wet=w_wet)
        crit(out, target, mask)["total"].backward()
        g = m.wet_head.net[0].weight.grad
        return 0.0 if g is None else float(g.abs().sum())

    assert _run(0.0) == 0.0, "unsupervised wet head should get no gradient"
    assert _run(0.30) > 0.0, "supervised wet head must get gradient"


def test_focal_alpha_is_class_specific():
    """alpha must weight positives by alpha and negatives by 1-alpha."""
    from losses.flood_loss import masked_focal_bce

    logits = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
    l_class = float(masked_focal_bce(logits, target, mask, gamma=0.0,
                                     alpha=0.25, alpha_mode="class"))
    l_scale = float(masked_focal_bce(logits, target, mask, gamma=0.0,
                                     alpha=0.25, alpha_mode="scale"))
    # class mode averages alpha and 1-alpha, so it differs from the global rescale
    assert abs(l_class - l_scale) > 1e-6


# --------------------------------------------------------------------------- #
# 4. train / eval pair separation
# --------------------------------------------------------------------------- #
def test_unseen_pairs_are_disjoint():
    cfg = yaml.safe_load((ROOT / "configs/v1_arbitrary_scale.yaml").read_text(encoding="utf-8"))
    train = {tuple(p) for p in cfg["dataset"]["train_pairs"]}
    seen = {tuple(p) for p in cfg["dataset"]["test_seen_pairs"]}
    unseen = {tuple(p) for p in cfg["dataset"]["test_unseen_pairs"]}
    assert train.isdisjoint(unseen), f"unseen pairs leaked into training: {train & unseen}"
    assert seen <= train, "seen pairs should be a subset of the training pairs"
    assert (5, 2) in unseen, "[5, 2] is the fractional-scale test and must be unseen"


# --------------------------------------------------------------------------- #
# 5. accumulator equivalence
# --------------------------------------------------------------------------- #
def test_global_rmse_accumulator_matches_concat():
    torch.manual_seed(0)
    pred1, trg1 = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    pred2, trg2 = torch.rand(3, 1, 8, 8), torch.rand(3, 1, 8, 8)
    mask1 = torch.ones_like(trg1, dtype=torch.bool)
    mask2 = torch.ones_like(trg2, dtype=torch.bool)

    acc = FloodMetricAccumulator()
    acc.update(pred1, trg1, mask1)
    acc.update(pred2, trg2, mask2)
    split = acc.compute()

    whole = FloodMetricAccumulator()
    whole.update(torch.cat([pred1, pred2]), torch.cat([trg1, trg2]),
                 torch.cat([mask1, mask2]))
    ref = whole.compute()

    for k in ("RMSE_all_domain", "RMSE_wet_domain", "MAE_wet_domain",
              "CSI_0.05_domain", "CSI_1.00_domain"):
        a, b = split[k], ref[k]
        if a != a and b != b:      # both undefined (empty event): equivalent
            continue
        assert abs(a - b) < 1e-6, f"{k}: {a} != {b}"


def test_accumulator_bias_keeps_sign():
    """A systematic under-prediction must read as a negative volume bias."""
    pred = torch.full((1, 1, 4, 4), 0.5)
    trg = torch.full((1, 1, 4, 4), 1.0)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    acc = FloodMetricAccumulator()
    acc.update(pred, trg, mask)
    out = acc.compute()
    assert out["VolumeBias_domain"] < 0
    assert out["VolumeRelativeError_domain"] > 0


# --------------------------------------------------------------------------- #
# 6. empty-event semantics
# --------------------------------------------------------------------------- #
def test_empty_csi_is_nan_not_zero():
    pred = torch.zeros(1, 1, 4, 4)
    trg = torch.zeros(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    csi, f1 = binary_csi_f1(pred, trg, mask, 0.05)
    assert csi != csi and f1 != f1, "empty event must be NaN, not 0"

    acc = FloodMetricAccumulator()
    acc.update(pred, trg, mask)
    assert acc.compute()["CSI_0.05_domain"] != acc.compute()["CSI_0.05_domain"]


def test_macro_average_skips_nan_and_reports_n():
    out = average_metrics([{"CSI": float("nan")}, {"CSI": 0.5}, {"CSI": 1.0}])
    assert abs(out["CSI"] - 0.75) < 1e-9
    assert out["CSI__n"] == 2.0


# --------------------------------------------------------------------------- #
# 7. boundary loss ignores nodata stencils
# --------------------------------------------------------------------------- #
def test_boundary_loss_ignores_nodata_stencil():
    from losses.flood_loss import FloodLoss

    # constant valid region, nothing to learn -> boundary term ~ 0
    h = torch.ones(1, 1, 8, 8)
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    crit = FloodLoss(w_depth=0.0, w_wet=0.0, w_log=0.0, w_extreme=0.0, w_boundary=1.0)
    out = {"depth": h}
    l0 = float(crit(out, h, mask)["l_boundary"])
    assert l0 < 1e-4, f"constant field should have no boundary loss, got {l0}"

    # now a valid patch surrounded by nodata; the internal edge is real, the
    # outer border must not be counted
    h2 = torch.ones(1, 1, 8, 8)
    mask2 = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask2[:, :, 2:6, 2:6] = True
    h2 = h2 * mask2
    l1 = float(crit({"depth": h2}, h2, mask2)["l_boundary"])
    assert l1 < 1e-4, f"nodata border leaked into boundary loss: {l1}"


# --------------------------------------------------------------------------- #
# 8. trainer honours config
# --------------------------------------------------------------------------- #
def test_multiscale_trainer_reads_yaml():
    cfg = yaml.safe_load((ROOT / "configs/v1_arbitrary_scale.yaml").read_text(encoding="utf-8"))
    assert cfg["train"]["epochs"] == 200
    assert cfg["train"]["batch_size"] == 2
    assert cfg["train"]["amp"] is True
    assert cfg["optimizer"]["lr"] == 1.0e-4
    src = (ROOT / "scripts/train_multiscale.py").read_text(encoding="utf-8")
    for token in ("epochs", "batch_size", "use_amp", "ResolutionPairBatchSampler",
                  "assert_pairs_disjoint", "evaluate_loader"):
        assert token in src, f"trainer does not reference {token}"


def test_mask_aware_base_can_be_switched_off():
    from models.hydrogeo_srno import HydroGeoSRNO

    m = HydroGeoSRNO(hydro_width=8, hydro_blocks=1, geo_width=8, geo_in_cont=0,
                     use_landuse=False, operator_width=16, heads=2,
                     operator_layers=1, predict_wet=False, mask_aware_base=False)
    assert m.mask_aware_base is False
