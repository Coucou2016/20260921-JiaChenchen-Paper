"""Print the collected pre-model numbers in a compact form for review."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "premodel"


def j(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def main():
    d = j("flood_error.json")
    for scen in ("20a", "100a"):
        print("=== flood", scen)
        for res, m in d["resolutions"][scen].items():
            print(f"  {res:4s} ratio={m['ratio']:4.1f} rmse={m['rmse']:.4f} "
                  f"mae={m['mae']:.4f} bias={m['bias']:+.4f} "
                  f"volrel={m['volume_rel']:+.3f} "
                  f"csi05={m['csi']['0.05']:.3f} csi30={m['csi']['0.30']:.3f} "
                  f"csi100={m['csi']['1.00']:.3f} pear={m['pearson_wet']:.3f} "
                  f"spear={m['spearman_wet']:.3f} ssim={m['ssim']:.3f}")
            dd = m["error_decomp_2m"]
            print(f"        at-2m total rmse={dd['rmse_total']:.4f} "
                  f"sim={dd['rmse_simulation']:.4f} ({dd['share_simulation_var']*100:.1f}%) "
                  f"agg={dd['rmse_aggregation']:.4f} ({dd['share_aggregation_var']*100:.1f}%) "
                  f"cross={dd['share_cross_var']*100:.1f}% "
                  f"chk={dd['identity_check']:.2e} "
                  f"csi_tot={dd['csi_total@0.05']:.3f} csi_agg={dd['csi_agg@0.05']:.3f} "
                  f"rel_rmse={m['rmse_rel_mean_truth']:.2f}")
            areas = {k: (round(v["truth"], 3), round(v["native"], 3))
                     for k, v in m["area_km2"].items()}
            print(f"        areas truth/native {areas}")
            for b in m["by_depth"]:
                if b.get("n"):
                    print(f"        depth {b['bin']:>9s} n={b['n']:>9d} "
                          f"mae={b['mae']:.4f} rmse={b['rmse']:.4f} bias={b['bias']:+.4f}")
            for b in m["by_slope"]:
                if b.get("n"):
                    print(f"        slope {b['bin']:>9s} n={b['n']:>9d} "
                          f"mae={b['mae']:.4f} rmse={b['rmse']:.4f} bias={b['bias']:+.4f}")
    if (OUT / "terrain_distortion.json").exists():
        t = j("terrain_distortion.json")
        print("=== terrain fine", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in t["fine"].items()})
        print("=== channel width 2m", {k: round(v, 2) for k, v in
                                       t["channel_width_2m"].items()})
        for res, m in t["resolutions"].items():
            print(f"  {res:4s} within_std={m['within_cell_std_mean']:.3f} "
                  f"relief={m['relief_mean']:.3f} sink={m['sink_depth_mean']:.3f} "
                  f"sink>0.1={m['sink_frac_gt_0p1']:.3f} slope {m['slope_native_mean']:.4f}"
                  f"/{m['slope_agg_mean']:.4f} ratio={m['slope_ratio']:.3f} "
                  f"width_ratio={m['width_ratio_median']:.2f} "
                  f"narrow_frac={m['water_pixels_narrower_than_cell']:.3f}")
    if (OUT / "velocity_correlation.json").exists():
        v = j("velocity_correlation.json")
        print("=== velocity")
        for scen, rec in v["scenarios"].items():
            for res, m in rec["resolutions"].items():
                print(f"  {scen} {res:4s} r={m['pearson']:.3f} rho={m['spearman']:.3f} "
                      f"mean {m['mean_native']:.3f} vs {m['mean_agg']:.3f}")


if __name__ == "__main__":
    main()
