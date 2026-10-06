import io, json, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
R = "outputs/deep_sweep/"
out = {}
for t in ("D0", "D2"):
    j = json.load(open(R + t + "/visualizations/diagnostics_val.json", encoding="utf-8"))
    n = j["depth"]["n"]; mae = j["depth"]["mae"]; bias = j["depth"]["bias"]
    rmse = j["depth"]["rmse"]; uf = j["depth"]["under_frac"]
    # deep = bins index 7..10 (>= 1.0 m)
    idx = [7, 8, 9, 10]
    N = sum(n[i] for i in idx)
    MAE = sum(n[i]*mae[i] for i in idx)/N
    RMSE = sum(n[i]*rmse[i] for i in idx)/N
    BIAS = sum(n[i]*bias[i] for i in idx)/N
    UF = sum(n[i]*uf[i] for i in idx)/N
    NA = sum(n)
    ALL = sum(n[i]*mae[i] for i in range(len(n)))/NA
    out[t] = dict(deep_n=N, deep_mae=MAE, deep_rmse=RMSE, deep_bias=BIAS,
                  deep_under=UF, n_all=NA, mae_all=ALL)
    print(f"{t}: n_all={NA:,} all_MAE={ALL:.4f} | deep n={N:,} MAE={MAE:.4f} "
          f"RMSE={RMSE:.4f} bias={BIAS:+.4f} under={UF*100:.1f}%")
a, b = out["D0"], out["D2"]
print(f"\ndeep MAE change: {(b['deep_mae']-a['deep_mae'])/a['deep_mae']*100:+.2f}%")
print(f"deep bias change: {b['deep_bias']-a['deep_bias']:+.4f} m")
print(f"all MAE change: {(b['mae_all']-a['mae_all'])/a['mae_all']*100:+.2f}%")
print("\nper-bin table (current authoritative file)")
print("bin, n, D0mae, D2mae, d%, D0bias, D2bias, D0under%, D2under%")
j0 = json.load(open(R + "D0/visualizations/diagnostics_val.json", encoding="utf-8"))
j2 = json.load(open(R + "D2/visualizations/diagnostics_val.json", encoding="utf-8"))
for i, lab in enumerate(j0["depth_bins"]):
    d = (j2["depth"]["mae"][i]-j0["depth"]["mae"][i])/j0["depth"]["mae"][i]*100
    print(f"{lab:9s} {j0['depth']['n'][i]:11.0f} {j0['depth']['mae'][i]:.4f} "
          f"{j2['depth']['mae'][i]:.4f} {d:+6.1f}% {j0['depth']['bias'][i]:+.4f} "
          f"{j2['depth']['bias'][i]:+.4f} {j0['depth']['under_frac'][i]*100:5.1f} "
          f"{j2['depth']['under_frac'][i]*100:5.1f}")
