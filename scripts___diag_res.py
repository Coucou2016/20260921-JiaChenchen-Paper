import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
d = json.load(open("outputs/premodel/premodel_results.json", encoding="utf-8"))
for r in ["5m", "10m", "20m", "30m"]:
    R = d[r]
    g = R["global"]
    print("===", r, "best_ep", R["best_epoch"], "n_train", R["n_train"],
          "n_val", R["n_val"], "n_test", R["n_test"], "epochs", len(R["history"]))
    for arm in ["identity", "bilinear", "premodel"]:
        m = g[arm]
        print("   {:9s} mae={:.4f} rmse={:.4f} bias={:+.4f} vol={:+.1f}% "
              "csi05={:.4f} csi30={:.4f} csi100={:.4f}".format(
                  arm, m["mae"], m["rmse"], m["bias"], m["volume_rel"] * 100,
                  m["csi@0.05"], m["csi@0.30"], m["csi@1.00"]))
    print("   skill", {k: round(v, 4) for k, v in R["skill"].items()})
    for arm, v in R["fine_grid_check"].items():
        print("   fine {:9s}".format(arm), {kk: (round(vv, 4) if isinstance(vv, float)
                                                else vv) for kk, vv in v.items()})
    for pk in ("premodel_vs_identity", "premodel_vs_bilinear", "bilinear_vs_identity"):
        for mk in ("mae", "rmse", "csi005"):
            s = R["paired"][pk][mk]
            if not s:
                continue
            print("   paired {:26s} {:7s} d={:+.4f} ci=[{:+.4f},{:+.4f}] t={:+.2f} "
                  "p_w={:.2e} dz={:+.2f} imp={:.1f}%".format(
                      pk, mk, s["mean_delta"], s["boot_ci_lo"], s["boot_ci_hi"],
                      s["t"], s["wilcoxon_p"], s["cohen_dz"], s["frac_improved"] * 100))
