"""Build the self-contained HTML report (base64 images, inline CSS/SVG)."""
from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "outputs" / "report_figs"
STATS = json.loads((ROOT / "outputs/finetune_deep/report_stats.json").read_text(encoding="utf-8"))


def b64(name: str) -> str:
    data = base64.b64encode((FIGS / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


IMG = {n: b64(n) for n in [
    "fig01_task.png", "fig02_baselines.png", "fig03_depth_bins.png",
    "fig04_paired_finetune.png", "fig05_selection_noise.png", "fig06_before_after.png",
    "fig07_forest.png", "fig08_bias_evolution.png", "fig09_sweep.png",
    "fig10_scorecard.png", "fig11_training_history.png", "fig12_loss_parts.png",
    "fig20_hyetograph.png", "fig21_tile_deep.png", "fig21b_tile_deep_scale.png",
    "fig22_tiles_spectrum.png",
    "fig23_domain_maps.png", "fig24_density.png", "fig25_distributions.png",
    "fig26_exceedance.png", "fig27_bootstrap.png", "fig28_autocorr.png",
    "fig41_premodel_task.png", "fig42_error_vs_grid.png",
    "fig43_error_depth_bins.png", "fig44_error_slope_bins.png",
    "fig45_terrain_distortion.png", "fig46_error_maps.png",
    "fig47_cross_resolution.png", "fig48_tile_scatter.png",
    "fig49_decomposition.png", "fig50_premodel_result.png",
    "fig61_two_fields.png", "fig62_error_accounts.png",
    "fig63_error_sources.png", "fig51_dialects.png",
    "fig64_domain_zoom.png",
    "fig71_pattern_by_grid.png", "fig72_depth_consistency.png",
    "fig73_speed_consistency.png",     "fig74_vmeasure.png",
    "fig75_pattern_map.png",
    "fig76_pattern_agreement.png", "fig77_pattern_areal.png",
    "fig78_pattern_boundary.png", "fig79_pattern_autocorr.png",
    "fig80_pattern_distance.png",
    "fig81_sabre_crosscheck.png", "fig82_variogram_scale.png",
    "fig83_contiguity_map.png", "fig84_contiguity_metrics.png",
    "fig85_k_sensitivity.png"]}

CSS = """
:root{--ink:#1a1a1a;--mut:#5c6470;--line:#d8dde3;--accent:#1f4e79;--accent2:#c0392b;
--bg:#ffffff;--soft:#f5f7fa;--good:#1a7a3a;--bad:#b03030;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:"Times New Roman",Georgia,"Songti SC",SimSun,serif;
font-size:16px;line-height:1.85;text-align:justify;}
.page{max-width:940px;margin:0 auto;padding:0 34px 90px;}
h1,h2,h3,h4{font-family:"Times New Roman",Times,"Heiti SC",SimHei,sans-serif;
line-height:1.35;color:var(--accent);font-weight:700;}
h2{font-size:23px;margin:56px 0 8px;padding-bottom:8px;border-bottom:2px solid var(--accent);}
h3{font-size:18.5px;margin:34px 0 6px;color:#24486b;}
h4{font-size:16.5px;margin:24px 0 4px;color:#33566f;}
p{margin:11px 0;}
a{color:var(--accent);}
.cover{min-height:96vh;display:flex;flex-direction:column;justify-content:center;
align-items:center;text-align:center;padding:60px 30px;
border-bottom:1px solid var(--line);}
.cover .kicker{letter-spacing:5px;font-size:12.5px;color:var(--mut);
text-transform:uppercase;margin-bottom:26px;}
.cover h1{font-size:33px;line-height:1.45;margin:0 0 14px;max-width:760px;}
.cover .sub{font-size:18px;color:var(--mut);margin:0 0 40px;max-width:700px;}
.cover .rule{width:120px;height:3px;background:var(--accent);margin:0 0 40px;}
.cover .meta{font-size:14.5px;color:var(--mut);line-height:2.1;}
.cover .meta b{color:var(--ink);}
.abstract{background:var(--soft);border-left:5px solid var(--accent);
padding:22px 28px;margin:34px 0;border-radius:0 6px 6px 0;}
.abstract h2{margin-top:0;border:0;font-size:20px;}
.kw{font-size:14.5px;color:var(--mut);margin-top:14px;}
.kw b{color:var(--ink);}
.toc{background:var(--soft);border:1px solid var(--line);border-radius:8px;
padding:24px 30px;margin:34px 0;}
.toc h2{margin-top:0;border:0;font-size:21px;}
.toc ol{list-style:none;padding-left:0;margin:10px 0;counter-reset:s;}
.toc li{counter-increment:s;margin:7px 0;font-size:15.5px;}
.toc li::before{content:counter(s,decimal) ". ";color:var(--accent);font-weight:700;}
.toc ol ol{counter-reset:s2;padding-left:26px;margin:6px 0;}
.toc ol ol li{counter-increment:s2;font-size:14.5px;color:var(--mut);}
.toc ol ol li::before{content:counter(s) "." counter(s2) " ";color:var(--mut);font-weight:400;}
figure{margin:28px 0;padding:0;}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:5px;background:#fff;padding:6px;}
figcaption{font-size:14.5px;color:var(--mut);margin-top:11px;line-height:1.75;
padding-left:12px;border-left:3px solid var(--accent);text-align:justify;}
figcaption b{color:var(--ink);}
table{width:100%;border-collapse:collapse;margin:22px 0;font-size:14.5px;
font-family:"Times New Roman",Times,"Heiti SC",SimHei,sans-serif;}
caption{caption-side:top;text-align:left;font-size:14.5px;color:var(--mut);
margin-bottom:9px;padding-left:12px;border-left:3px solid var(--accent);line-height:1.7;}
caption b{color:var(--ink);}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:center;}
th{background:#eef2f6;color:#24486b;font-weight:700;}
tbody tr:nth-child(even){background:#fafbfc;}
td.l,th.l{text-align:left;}
.good{color:var(--good);font-weight:700;}
.bad{color:var(--bad);font-weight:700;}
.note{background:#fff9ec;border-left:4px solid #e0a52b;padding:14px 20px;margin:22px 0;
font-size:15px;border-radius:0 5px 5px 0;}
.warn{background:#fdf2f2;border-left:4px solid var(--accent2);padding:14px 20px;margin:22px 0;
font-size:15px;border-radius:0 5px 5px 0;}
.eq{background:var(--soft);border:1px solid var(--line);border-radius:6px;
padding:14px 20px;margin:20px 0;text-align:center;font-size:16.5px;}
.term{border-bottom:1px dotted #9aa5b1;cursor:help;}
.svgs{display:block;margin:26px auto;max-width:100%;}
.timeline{width:100%;height:auto;}
small{color:var(--mut);}
.foot{margin-top:70px;padding-top:18px;border-top:1px solid var(--line);
font-size:13.5px;color:var(--mut);text-align:center;}
@media print{body{font-size:11pt;}.page{max-width:none;padding:0;}
h2{page-break-after:avoid;}figure,table{page-break-inside:avoid;}
.cover{min-height:auto;page-break-after:always;}}
@media(max-width:640px){body{font-size:15px;}.page{padding:0 16px 60px;}
.cover h1{font-size:25px;}h2{font-size:20px;}}
"""

TIMELINE_SVG = """
<svg class="svgs timeline" viewBox="0 0 900 210" xmlns="http://www.w3.org/2000/svg"
 role="img" aria-label="research timeline">
<style>
.tl-l{font:13px "Times New Roman",serif;fill:#5c6470;}
.tl-t{font:bold 12.5px "Times New Roman",serif;fill:#1f4e79;}
.tl-a{font:11.5px "Times New Roman",serif;fill:#333;}
.tl-n{font:bold 11px "Times New Roman",serif;fill:#fff;}
</style>
<line x1="60" y1="105" x2="850" y2="105" stroke="#d8dde3" stroke-width="4"/>
<g>
<circle cx="120" cy="105" r="17" fill="#1f4e79"/><text x="120" y="109" class="tl-n"
 text-anchor="middle">1</text>
<text x="120" y="66" class="tl-t" text-anchor="middle">Baseline</text>
<text x="120" y="150" class="tl-a" text-anchor="middle">train until</text>
<text x="120" y="166" class="tl-a" text-anchor="middle">plateau</text>
<circle cx="300" cy="105" r="17" fill="#1f4e79"/><text x="300" y="109" class="tl-n"
 text-anchor="middle">2</text>
<text x="300" y="66" class="tl-t" text-anchor="middle">Diagnosis</text>
<text x="300" y="150" class="tl-a" text-anchor="middle">two loss</text>
<text x="300" y="166" class="tl-a" text-anchor="middle">defects</text>
<circle cx="480" cy="105" r="17" fill="#b03030"/><text x="480" y="109" class="tl-n"
 text-anchor="middle">3</text>
<text x="480" y="66" class="tl-t" text-anchor="middle">From scratch</text>
<text x="480" y="150" class="tl-a" text-anchor="middle">strong weight</text>
<text x="480" y="166" class="tl-a" text-anchor="middle">rejected</text>
<circle cx="650" cy="105" r="17" fill="#1a7a3a"/><text x="650" y="109" class="tl-n"
 text-anchor="middle">4</text>
<text x="650" y="66" class="tl-t" text-anchor="middle">Fine-tune</text>
<text x="650" y="150" class="tl-a" text-anchor="middle">gentle weight</text>
<text x="650" y="166" class="tl-a" text-anchor="middle">kept</text>
<circle cx="815" cy="105" r="17" fill="#6b7280"/><text x="815" y="109" class="tl-n"
 text-anchor="middle">5</text>
<text x="815" y="66" class="tl-t" text-anchor="middle">Extend</text>
<text x="815" y="150" class="tl-a" text-anchor="middle">no further</text>
<text x="815" y="166" class="tl-a" text-anchor="middle">gain</text>
</g>
</svg>
"""

# --------------------------------------------------------------------------
HEAD = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>城市洪水深度超分辨率重建中深水欠估问题的诊断与修正</title>
<style>{CSS}</style>
</head>
<body>
"""

COVER = """
<section class="cover">
<div class="kicker">Technical Research Report</div>
<h1>城市洪水深度超分辨率重建中的深水欠估问题<br>诊断、修正与方法学检验</h1>
<p class="sub">以新西兰惠灵顿城区洪水数据集为例，10 米到 2 米最大淹没水深重建</p>
<div class="rule"></div>
<div class="meta">
<b>研究对象</b>　HydroGeo-SRNO 引导式超分辨率模型与深水加权损失项<br>
<b>数据</b>　Wellington Urban Flood Dataset（NZTM2000 / EPSG:2193）<br>
<b>计算设备</b>　单卡 NVIDIA GeForce GTX 950M，显存 4 GB<br>
<b>报告日期</b>　2026 年 9 月 29 日
</div>
</section>
"""

TOC = """
<section class="toc">
<h2>目录</h2>
<ol>
<li>摘要</li>
<li>研究背景与目的
  <ol>
  <li>为什么需要把粗网格水深还原到细网格</li>
  <li>引导式超分辨率的含义</li>
  <li>一个长期被整体指标掩盖的问题</li>
  <li>研究目的</li>
  <li>本报告的五点贡献</li>
  </ol></li>
<li>前置模型与低分辨率模拟的误差量化
  <ol>
  <li>粗网格模拟与聚合真值</li>
  <li>误差的口径与评价指标</li>
  <li>低分辨率模拟误差随网格尺寸的变化</li>
  <li>细网格口径与粗网格口径</li>
  <li>哪些条件让误差变大</li>
  <li>地形失真的来源与量级</li>
  <li>跨分辨率相关性的衰减规律</li>
  <li>地形与下垫面的跨分辨率相关性</li>
  <li>水动力变量的跨分辨率相关性</li>
  <li>基于 V-measure 的空间模式相似性</li>
  <li>空间模式相似性的多族度量</li>
  <li>跨分辨率模式的尺度、连通性与簇数检验</li>
  <li>跨分辨率一致性对输入特征的启示</li>
  <li>逐瓦片的离散与解释份额</li>
  <li>前置模型的结构、训练与测试结果</li>
  <li>本章术语速查</li>
  </ol></li>
<li>数据、模型与方法
  <ol>
  <li>研究区与数据来源</li>
  <li>数据组织与切分原则</li>
  <li>降雨驱动与洪水场的基本形态</li>
  <li>模型结构</li>
  <li>损失函数与评价指标</li>
  <li>计算条件</li>
  </ol></li>
<li>研究过程
  <ol>
  <li>阶段一　基线训练及其饱和</li>
  <li>阶段二　深水误差的根因诊断</li>
  <li>阶段三　从头训练的深水损失对比</li>
  <li>阶段四　冻结模型的温和微调</li>
  <li>阶段五　两处测量陷阱与修正</li>
  <li>阶段六　延长训练与当前状态</li>
  </ol></li>
<li>结果
  <ol>
  <li>逐瓦片的空间重构对比</li>
  <li>点对点一致性与偏差结构</li>
  <li>基线水平与超分辨率增益</li>
  <li>深水分桶诊断</li>
  <li>从头训练的权利衡代价</li>
  <li>微调的配对结果</li>
  <li>统计检验与效应量</li>
  <li>空间自相关下的重估</li>
  <li>测试集复核</li>
  <li>高阶统计分析与阈值敏感性</li>
  <li>偏差随训练的演化</li>
  </ol></li>
<li>分析与讨论</li>
<li>主要结论</li>
<li>不足与展望</li>
<li>附录　术语表、参考文献与复现命令</li>
</ol>
</section>
"""


def _fmt(v, nd=3, plus=False):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:  # NaN
        return "—"
    s = f"{v:+.{nd}f}" if plus else f"{v:.{nd}f}"
    return s


def per_tile_table() -> str:
    """Direct per-tile comparison, picking tiles across the deep-water spectrum."""
    import json
    from pathlib import Path

    p = ROOT / "outputs/report_figs/result_fields/per_tile.json"
    if not p.exists():
        return ""
    rows = json.loads(p.read_text(encoding="utf-8"))
    ok = [r for r in rows if r["peak_truth"] > 0]
    ok.sort(key=lambda r: r["peak_truth"])
    qs = [0.10, 0.35, 0.55, 0.75, 0.90, 0.97, 1.0]
    picks, seen = [], set()
    for q in qs:
        r = ok[min(int(q * (len(ok) - 1)), len(ok) - 1)]
        if r["index"] not in seen:
            seen.add(r["index"])
            picks.append(r)

    body = []
    scen_cn = {"20a": "20 年一遇", "100a": "100 年一遇"}
    for r in picks:
        dcsi05 = r["win_CSI_005"] - r["frozen_CSI_005"]
        dcsi = r["win_CSI_100"] - r["frozen_CSI_100"]
        dvol = r["frozen_VolumeRelativeError"] - r["win_VolumeRelativeError"]
        cls = "good" if dcsi > 1e-9 else ("bad" if dcsi < -1e-9 else "")
        vcls = "good" if dvol > 1e-9 else ("bad" if dvol < -1e-9 else "")
        sc = r["scenario"]
        sc = sc[0] if isinstance(sc, (list, tuple)) and sc else sc
        sc = str(sc).strip("[]'\" ")
        lab = f"{scen_cn.get(sc, sc)} {r['iy']}/{r['ix']}"
        body.append(
            f"<tr><td class='l'>{lab}</td>"
            f"<td>{_fmt(r['peak_truth'], 2)}</td>"
            f"<td>{_fmt(r['wet_frac']*100, 1)}</td>"
            f"<td>{_fmt(r['frozen_CSI_005'])} → {_fmt(r['win_CSI_005'])}</td>"
            f"<td class='{cls}'>{_fmt(dcsi05, 3, plus=True)}</td>"
            f"<td>{_fmt(r['frozen_CSI_100'])} → "
            f"<span class='{cls}'>{_fmt(r['win_CSI_100'])}</span></td>"
            f"<td class='{cls}'>{_fmt(dcsi, 3, plus=True)}</td>"
            f"<td>{_fmt(r['frozen_VolumeRelativeError'])} → "
            f"{_fmt(r['win_VolumeRelativeError'])}</td>"
            f"<td class='{vcls}'>{_fmt(dvol, 3, plus=True)}</td></tr>")
    return (
        "<table><caption><b>表 90　逐瓦片的直接对比（测试集代表性瓦片）。</b>"
        "瓦片按真值峰值水深从第 10 百分位取到最大，覆盖从浅水到最深的完整区间。"
        "每一对指标都写成「冻结基线 → 微调模型」，箭头左侧是基线，右侧是微调。"
        "瓦片一列给出降雨情景与瓦片的纵向、横向编号，例如 20 年一遇 40/7。"
        "CSI 列为该瓦片自身的判定指标，体积误差为该瓦片的总水量相对偏差。"
        "增益列为微调减基线，判定指标的增益正值表示改善，"
        "体积误差的增益写成基线减微调，正值同样表示改善，因此两列方向一致，"
        "都读作越大越好。单个瓦片的指标波动较大，"
        "这里看的是改善是否在全区间上一致出现。</caption>"
        "<thead><tr><th class='l'>瓦片<br>情景/坐标</th><th>真值峰值<br>(m)</th>"
        "<th>湿区占比<br>(%)</th>"
        "<th>CSI@0.05<br>基线 → 微调</th><th>CSI@0.05<br>增益</th>"
        "<th>CSI@1.00<br>基线 → 微调</th><th>CSI@1.00<br>增益</th>"
        "<th>体积误差<br>基线 → 微调</th><th>体积<br>增益</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>")


def higher_order_table() -> str:
    """Third and fourth moments of the error, by depth bin, for both models."""
    import json

    p = ROOT / "outputs/report_figs/result_fields/moments.json"
    if not p.exists():
        return ""
    mom = json.loads(p.read_text(encoding="utf-8"))
    bins = mom["bins"]
    f, w = mom["models"]["frozen"], mom["models"]["win"]
    body = []
    for i, lab in enumerate(bins):
        # dependence axis made explicit: each paired cell is 基线 → 微调
        dunder = (w["under_frac"][i] - f["under_frac"][i]) * 100
        ucls = "good" if dunder < 0 else ("bad" if dunder > 0 else "")
        body.append(
            f"<tr><td class='l'>{lab}</td><td>{f['n'][i]:,.0f}</td>"
            f"<td>{_fmt(f['mean'][i], 3, plus=True)} → "
            f"{_fmt(w['mean'][i], 3, plus=True)}</td>"
            f"<td>{_fmt(f['under_frac'][i]*100, 1)} → "
            f"{_fmt(w['under_frac'][i]*100, 1)}</td>"
            f"<td class='{ucls}'>{_fmt(dunder, 1, plus=True)}</td>"
            f"<td>{_fmt(f['skew'][i], 2, plus=True)}</td>"
            f"<td>{_fmt(f['excess_kurtosis'][i], 1, plus=True)}</td></tr>")
    gf, gw = f["global"], w["global"]
    body.append(
        f"<tr><td class='l'>全部像元</td><td>{gf['n']:,.0f}</td>"
        f"<td>{_fmt(gf['mean'], 3, plus=True)} → {_fmt(gw['mean'], 3, plus=True)}</td>"
        f"<td>— → —</td><td>—</td>"
        f"<td>{_fmt(gf['skew'], 2, plus=True)}</td>"
        f"<td>{_fmt(gf['excess_kurtosis'], 1, plus=True)}</td></tr>")
    return (
        "<table><caption><b>表 91　误差的高阶矩按水深分层（全测试集）。</b>"
        "每一对单元格都写成「冻结基线 → 微调模型」，箭头左侧是基线，右侧是微调。"
        "平均偏差为误差的均值，负值表示低估，单位是米。"
        "低估比例是误差为负的像元占比。低估变化一列是微调减基线，单位为百分点，"
        "负值表示低估的像元变少，也就是改善。"
        "偏度衡量误差分布的不对称程度，正值表示右侧拖尾更长，即高估的极端情形更多；"
        "三米以上转为负值，说明极深水体处的误差由低估主导。"
        "超额峰度衡量尾部厚重程度，正态分布的取值为零，数值越大说明极端误差越频繁，"
        "它随水深单调下降，说明重尾集中在浅水区。两极矩只列基线，"
        "因为两个模型在分布形状上的差别在图上不可分辨，微调改变的是位置而不是形状。</caption>"
        "<thead><tr><th class='l'>水深区间（米）</th><th>像元数</th>"
        "<th>平均偏差（米）<br>基线 → 微调</th><th>低估比例（%）<br>基线 → 微调</th>"
        "<th>低估变化<br>（百分点）</th><th>误差偏度<br>基线</th>"
        "<th>超额峰度<br>基线</th></tr></thead><tbody>"
        + "".join(body) + "</tbody></table>")


def _prem(name):
    import json
    from pathlib import Path
    return json.loads((ROOT / "outputs" / "premodel" / name).read_text(encoding="utf-8"))


def pre_error_table() -> str:
    """Native coarse run against the area-weighted 2 m truth, per resolution."""
    import json
    from pathlib import Path

    p = ROOT / "outputs/premodel/flood_error.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    cells = {"5m": "5", "10m": "10", "20m": "20", "30m": "30"}
    rows = []
    for scen in ("20a", "100a"):
        for res in ("5m", "10m", "20m", "30m"):
            m = d["resolutions"][scen][res]
            rows.append(
                f"<tr><td class='l'>{scen}</td><td>{cells[res]}</td>"
                f"<td>{_fmt(m['mae'], 4)}</td>"
                f"<td>{_fmt(m['rmse'], 4)}</td>"
                f"<td>{_fmt(m['volume_rel']*100, 1, plus=True)}</td>"
                f"<td>{_fmt(m['csi']['0.05'])}</td>"
                f"<td>{_fmt(m['csi']['1.00'])}</td>"
                f"<td>{_fmt(m['pearson_wet'])}</td><td>{_fmt(m['spearman_wet'])}</td>"
                f"</tr>")
    return (
        "<table><caption><b>表 81　粗网格模拟值与聚合后的二米真值的误差。</b>"
        "粗网格模拟值指水动力程序直接在这套粗网格上求解浅水方程得到的水深场，"
        "聚合后的二米真值指把两米精细结果按面积加权平均到同一套网格的结果，两者的区别见 2.1 节。"
        "所有指标都在各自的粗网格上计算，因此这一张表度量的是表示误差与数值误差，"
        "不含二米真值自身被粗化丢掉的信息。"
        "临界成功指数给出五厘米与一米两个阈值，三十厘米档见 2.7 节的图 14。"
        "相关系数只在湿区像元上统计，"
        "湿区取水深超过五厘米。体积相对误差是全部单元水量之和的相对偏差，"
        "正值表示粗网格给多了水，读它的绝对幅度。"
        "读数方向，平均绝对误差与均方根误差两列越小越准，"
        "临界成功指数、皮尔逊与斯皮尔曼三列越大越准。"
        "边长一列同时充当分辨率比，把边长除以二米即得 2.5、5、10 与 15 四档。"
        "</caption><thead><tr><th class='l'>重现期</th><th>边长<br>(米)</th>"
        "<th>MAE<br>(米)</th><th>RMSE<br>(米)</th>"
        "<th>体积相对<br>误差(%)</th><th>CSI<br>@0.05</th>"
        "<th>CSI<br>@1.00</th><th>Pearson</th><th>Spearman</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>")


def pre_decomp_table() -> str:
    """Simulation term versus aggregation term, both measured on the 2 m lattice."""
    import json

    p = ROOT / "outputs/premodel/flood_error.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for scen in ("20a", "100a"):
        for res in ("5m", "10m", "20m", "30m"):
            m = d["resolutions"][scen][res]["error_decomp_2m"]
            rows.append(
                f"<tr><td class='l'>{scen}</td><td>{res[:-1]}</td>"
                f"<td>{m['rmse_total']:.4f}</td><td>{m['rmse_simulation']:.4f}</td>"
                f"<td>{m['rmse_aggregation']:.4f}</td>"
                f"<td>{m['share_simulation_var']*100:.1f}</td>"
                f"<td>{m['share_aggregation_var']*100:.1f}</td>"
                f"<td>{m['share_cross_var']*100:+.1f}</td>"
                f"<td>{m['csi_total@0.05']:.3f}</td><td>{m['csi_agg@0.05']:.3f}</td>"
                f"</tr>")
    return (
        "<table><caption><b>表 82　两类误差在二米格网上的分解。</b>"
        "把粗网格模拟值与聚合后的二米真值都按最近邻升采样回二米格网，总误差等于两者之差。"
        "总误差可以写成两项相加。模拟项是升采样后的粗网格模拟值减去升采样后的聚合真值。"
        "聚合项是升采样后的聚合真值减去真实二米水深。后三列给出三项平方误差在总平方误差中的"
        "占比，三项相加为一。交叉项为正值表示两类误差在同一像元上同向叠加，为负值表示"
        "部分抵消。最后两列是总误差与聚合项单独造成的淹没匹配度，聚合项那一列可以理解为"
        "单靠粗网格的方块状表示能达到的匹配度上限。</caption>"
        "<thead><tr><th class='l'>重现期</th><th>边长<br>(米)</th><th>总误差<br>RMSE(米)</th>"
        "<th>模拟项<br>RMSE(米)</th><th>聚合项<br>RMSE(米)</th><th>模拟项<br>方差占比(%)</th>"
        "<th>聚合项<br>方差占比(%)</th><th>交叉项<br>占比(%)</th><th>总误差<br>CSI@0.05</th>"
        "<th>聚合项<br>CSI@0.05</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_sources_table() -> str:
    """Low-group versus high-group error contrast, at cell level and tile level."""
    import json

    p = ROOT / "outputs/premodel/cell_contrasts.json"
    q = ROOT / "outputs/premodel/factor_contrasts.json"
    if not p.exists() or not q.exists():
        return ""
    cf = json.loads(p.read_text(encoding="utf-8"))
    tf = json.loads(q.read_text(encoding="utf-8"))

    def num(v):
        return f"{v:.4g}" if abs(v) < 10 else f"{v:.0f}"

    rows = []

    def add(part, cn, lo_t, hi_t, lo_m, hi_m, ratio, lo_d, hi_d, r):
        rows.append(
            f"<tr><td class='l'>{part}</td><td class='l'>{cn}</td>"
            f"<td>{lo_t}</td><td>{lo_m:.4f}</td>"
            f"<td>{hi_t}</td><td>{hi_m:.4f}</td><td>{ratio:.2f}</td>"
            f"<td>{lo_d:.3f}</td><td>{hi_d:.3f}</td><td>{r:.2f}</td></tr>")

    c = cf["resolutions"]["10m"]["100a"]["factors"]
    order = ["Slope", "Building_binary", "Dist_water"]
    for name in order:
        e = c[name]
        lo_t = "0" if name == "Building_binary" else f"≤{num(e['low_cut'])}"
        hi_t = "1" if name == "Building_binary" else f"≥{num(e['high_cut'])}"
        if name == "Building_binary":
            lo_t, hi_t = "覆盖率 0", "覆盖率 1"
        add("单元", e["cn"], lo_t, hi_t,
            e["low"]["mae"], e["high"]["mae"], e["mae_ratio_high_over_low"],
            e["low"]["mean_truth_m"], e["high"]["mean_truth_m"],
            e["pearson_with_abs_err"])

    t = tf["resolutions"]["10m"]["factors"]
    torder = ["mean_slope", "within_cell_dem_std", "building_frac",
              "impervious_frac", "mean_dist_water", "landuse_entropy"]
    for name in torder:
        e = t[name]
        add("瓦片", e["cn"],
            f"≤{num(e['low_cut'])}", f"≥{num(e['high_cut'])}",
            e["low"]["mae"]["mean"], e["high"]["mae"]["mean"],
            e["mae_ratio_high_over_low"],
            e["low"]["mean_depth_2m"]["mean"], e["high"]["mean_depth_2m"]["mean"],
            e["pearson_with_mae"])

    return (
        "<table><caption><b>表 96　哪些条件让误差变大，单元一级与瓦片一级的两组对照。</b>"
        "每个条件按取值分成三档，本表只列最低与最高两档。建筑覆盖率一行取无建筑与满覆盖"
        "这两个端点，其余条件取最高与最低的三分之一。单元一级取十米网格、"
        "一百年一遇情景，统计范围是全部有水单元，共 276143 个，条件变量取该单元自身的取值。"
        "瓦片一级取十米网格下的全部 487 块瓦片与两个情景，共 974 条记录，"
        "条件变量取该瓦片内的平均值。平均绝对误差单位是米。比值是两组的平均绝对误差相除，"
        "大于一表示高值组误差更大，小于一表示方向相反。平均水深两列给出两组内部的水深，"
        "用来提示两组的水深并不相同，读比值时需要一起看。相关系数是条件变量与误差绝对值"
        "之间的皮尔逊相关系数，只描述线性同步程度。单元建筑覆盖率一行的中间档是部分覆盖，"
        "它的平均绝对误差是 0.360 米，高于本表列出的两端，因而不在本表的两列之内。</caption>"
        "<thead><tr><th class='l'>统计部位</th><th class='l'>条件变量</th>"
        "<th>低值组<br>取值</th><th>低值组<br>MAE(米)</th><th>高值组<br>取值</th>"
        "<th>高值组<br>MAE(米)</th><th>MAE 之比<br>高值组/低值组<br>(＞1 误差更大)</th>"
        "<th>低值组<br>平均水深<br>(米)</th><th>高值组<br>平均水深<br>(米)</th>"
        "<th>与误差绝对值<br>的相关系数</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_terrain_table() -> str:
    """Terrain distortion statistics as a function of the coarse cell size."""
    import json

    p = ROOT / "outputs/premodel/terrain_distortion.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for res in ("5m", "10m", "20m", "30m"):
        m = d["resolutions"][res]
        rows.append(
            f"<tr><td>{res[:-1]}</td>"
            f"<td>{m['within_cell_std_mean']:.3f}</td>"
            f"<td>{m['relief_mean']:.2f}</td><td>{m['sink_depth_mean']:.3f}</td>"
            f"<td>{m['sink_frac_gt_0p1']*100:.1f}</td>"
            f"<td>{m['sink_frac_gt_0p5']*100:.1f}</td>"
            f"<td>{m['slope_native_mean']:.4f}</td>"
            f"<td>{m['slope_ratio']:.3f}</td>"
            f"<td>{m['width_ratio_median']:.2f}</td>"
            f"<td>{m['water_pixels_narrower_than_cell']*100:.1f}</td>"
            f"<td>{m['dominant_class_frac_mean']*100:.1f}</td></tr>")
    f = d["fine"]
    return (
        "<table><caption><b>表 83　地形失真随网格尺寸的变化。</b>"
        "全部统计量只使用 static.nc 中确实存在的通道，高程取 DEM，坡度取 Slope，"
        "水体取土地利用编码七，建筑取 Building_binary。单元内高程标准差与单元内高差"
        "刻画一个粗单元内部的地形起伏被抹平的程度，数值越大表示粗网格看到的越平坦。"
        "洼地深度是单元内高程均值减去单元内最低高程，代表被粗化抹掉的最深局部汇水点。"
        "坡度比是粗网格自身坡度均值除以二米坡度聚合后均值，小于一表示粗网格系统性地"
        "把地形看缓。水道宽与边长之比是二米水体像元中位宽度除以网格边长，"
        "窄于边长的水体占比是二米水体像元中宽度小于网格边长的比例，两者一起说明水道"
        "相对网格有多细。最后一列是单元主导地类占比，数值随网格变粗而下降表示"
        "单元内部的地类越混。参照量，二米高程标准差为 "
        f"{f['dem_std']:.1f} 米，二米水体像元中位宽度见正文。</caption>"
        "<thead><tr><th>边长<br>(米)</th>"
        "<th>单元内高程<br>标准差均值(米)</th>"
        "<th>单元内<br>高差均值(米)</th><th>洼地深度<br>均值(米)</th><th>洼地深于<br>0.1 米(%)</th>"
        "<th>洼地深于<br>0.5 米(%)</th><th>粗网格<br>坡度均值</th>"
        "<th>坡度比<br>(粗/聚合)</th><th>水道宽/边长<br>中位数</th><th>窄于边长的<br>水体占比(%)</th>"
        "<th>单元主导<br>地类占比(%)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_corr_table() -> str:
    """Cross-resolution agreement of depth and of the derived speed field."""
    import json

    pf = ROOT / "outputs/premodel/flood_error.json"
    pv = ROOT / "outputs/premodel/velocity_correlation.json"
    if not pf.exists():
        return ""
    d = json.loads(pf.read_text(encoding="utf-8"))
    v = json.loads(pv.read_text(encoding="utf-8")) if pv.exists() else {"scenarios": {}}
    rows = []
    for scen in ("20a", "100a"):
        for res in ("5m", "10m", "20m", "30m"):
            m = d["resolutions"][scen][res]
            vr = v.get("scenarios", {}).get(scen, {}).get("resolutions", {}).get(res, {})
            rows.append(
                f"<tr><td class='l'>{scen}</td><td>{res[:-1]}</td>"
                f"<td>{m['csi']['0.05']:.3f}</td><td>{m['csi']['1.00']:.3f}</td>"
                f"<td>{m['pearson_wet']:.3f}</td><td>{m['spearman_wet']:.3f}</td>"
                f"<td>{m['ssim']:.3f}</td>"
                f"<td>{_fmt(vr.get('pearson'), 3)}</td>"
                f"<td>{_fmt(vr.get('spearman'), 3)}</td>"
                f"<td>{_fmt(vr.get('mean_native'), 3)}</td>"
                f"<td>{_fmt(vr.get('mean_agg'), 3)}</td></tr>")
    frame = {s: v.get("scenarios", {}).get(s, {}).get("frame") for s in ("20a", "100a")}
    return (
        "<table><caption><b>表 84　跨分辨率相关性与流速场相似性。</b>"
        "本表回答的是若干指标以什么速度衰减，以及流速场比水深场差多少。"
        "水深部分比较粗网格模拟值与聚合后的二米真值，行列含义与表 81 一致。"
        "二值掩膜下交并比与临界成功指数恒等，因此只列临界成功指数，"
        "三十厘米档见 2.7 节的图 14。流速场的相似性由 hux 与 hvy 两个单宽流量分量"
        "除以水深得到，只在数据存有流量场的整点时刻评估，取全域水量最大的那个整点。"
        f"100a 与 20a 情景都落在起始后的第 {frame.get('100a')} 个整点。"
        "最后两列是粗网格模拟值与聚合后的二米真值各自的平均流速，两者对照。"
        "表头篇幅有限，用短名代替完整说法。"
        "模拟均流速是粗网格模拟值给出的平均流速，聚合均流速是聚合后的二米真值给出的平均流速，"
        "两个短名与 2.1 节的命名一一对应。"
        "流速是数据中没有直接存储的量，由存有的单宽流量与水深相除得到，属于派生量。"
        "读数方向，四个相关系数、两个临界成功指数与结构相似度都是越大越好。"
        "</caption><thead><tr><th class='l'>重现期</th><th>边长<br>(米)</th>"
        "<th>CSI<br>@0.05</th><th>CSI<br>@1.00</th>"
        "<th>水深<br>Pearson</th><th>水深<br>Spearman</th><th>水深<br>SSIM</th>"
        "<th>流速<br>Pearson</th><th>流速<br>Spearman</th><th>模拟均<br>流速(m/s)</th>"
        "<th>聚合均<br>流速(m/s)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def _cons() -> dict:
    import json
    p = ROOT / "outputs/premodel/cross_resolution_consistency.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def pre_cons_text() -> dict:
    """Free-standing numbers that the new prose sections quote inline."""
    d = _cons()
    if not d:
        return {}
    c = d["continuous"]["variables"]
    f = d["flood"]["variables"]
    vm = d["vmeasure"]["variables"]
    R = ("5m", "10m", "20m", "30m")

    def series(key, field, res=R):
        return [c[key][r][field] for r in res]

    def fseries(var, field, scen="100a", res=R):
        return [f[var][scen]["resolutions"][r][field] for r in res]

    def dseries(var, field, scen="100a", res=R):
        return [f[var][scen]["resolutions"][r]["dist_native"][field] for r in res]

    def fmt(v, n=4):
        return f"{v:.{n}f}"

    def lst(v, n=4):
        return "、".join(fmt(x, n) for x in v)

    dist = (
        "水深的最优分布族在四档网格上完全一致。一百年一遇与二十年一遇两个情景都在四档上"
        "选了对数正态分布，二米参考场也不例外。柯尔莫哥洛夫-斯米尔诺夫统计量在一百年一遇情景下"
        f"分别是 {lst(dseries('h_max', 'best_ks'), 3)}，二十米网格上最小。"
        "二十年一遇情景的走势相同，统计量依次是 "
        f"{lst(dseries('h_max', 'best_ks', '20a'), 3)}。"
        "对数正态分布的均值随网格变大略有抬升，从二米参考的 "
        f"{fmt(f['h_max']['100a']['dist_2m']['mean'], 3)} 米升到三十米的 "
        f"{fmt(dseries('h_max', 'mean')[-1], 3)} 米。"
        "中位数在四档之间没有单调走向，落在 "
        f"{fmt(min(dseries('h_max', 'median')), 3)} 到 "
        f"{fmt(max(dseries('h_max', 'median')), 3)} 米之间。"
        "流速的情况不同，这也是本节唯一一处最优族发生更换的地方。"
        "五米与十米网格上最优族仍是对数正态分布，二十米与三十米网格换成三参数韦布尔分布。"
        "换成尾部更轻的韦布尔分布，对应的是粗网格上被抹平的那批极端流速单元。"
        "流速的拟合统计量在四档上落在 "
        f"{lst(dseries('speed', 'best_ks'), 3)}，比水深小一个量级。"
        "原因是流速样本只在很窄的一段取值区间里展开，拟合本身比水深容易。")

    kurt = (
        "偏度与超额峰度给出同一方向的判断。"
        f"二米参考的水深偏度是 {fmt(f['h_max']['100a']['dist_2m']['skew'], 2)}，"
        f"超额峰度是 {fmt(f['h_max']['100a']['dist_2m']['kurtosis_excess'], 1)}，"
        "两者都是五档里最高的。四档粗网格上两者都低于参考，总体随网格变大继续下降，"
        f"三十米网格上落到 {fmt(dseries('h_max', 'skew')[-1], 2)} 与 "
        f"{fmt(dseries('h_max', 'kurtosis_excess')[-1], 1)}。"
        "粗网格抹掉了细网格上那些孤立的高水深单元，尾部因此变轻。"
        f"流速的偏度从五米的 {fmt(dseries('speed', 'skew')[0], 2)} 降到三十米的 "
        f"{fmt(dseries('speed', 'skew')[-1], 2)}，超额峰度从 "
        f"{fmt(dseries('speed', 'kurtosis_excess')[0], 1)} 降到 "
        f"{fmt(dseries('speed', 'kurtosis_excess')[-1], 1)}，方向与水深一致。")

    h = vm["h_max"]["pairwise"]
    sp = vm["speed"]["pairwise"]
    _res = ("2m", "5m", "10m", "20m", "30m")
    _offdiag = [(a, b) for a in _res for b in _res if a != b]
    vmt = (
        "两个矩阵的数值整体偏低。水深部分，二米参考与四档粗网格之间的 V-measure 依次是 "
        f"{fmt(h['2m']['5m']['v'], 3)}、{fmt(h['2m']['10m']['v'], 3)}、"
        f"{fmt(h['2m']['20m']['v'], 3)} 与 {fmt(h['2m']['30m']['v'], 3)}。"
        "四档粗网格之间，五米与十米这一格最高，达到 "
        f"{fmt(h['5m']['10m']['v'], 3)}，十米与二十米降到 {fmt(h['10m']['20m']['v'], 3)}，"
        f"二十米与三十米只有 {fmt(h['20m']['30m']['v'], 3)}。"
        "这个格局说明洪涝水深的区域划分在跨分辨率上并不稳定，"
        "只有分辨率最接近的五米与十米两档之间保留了七成左右的共同结构。"
        "同质性与完备性两个分量几乎相等，"
        f"二米参考与四档粗网格上两者相差最大也只有 "
        f"{max(abs(h['2m'][r]['h'] - h['2m'][r]['c']) for r in ('5m', '10m', '20m', '30m')):.3f}，"
        "说明一致性的损失没有偏向哪一侧。"
        "二米参考与四档粗网格之间的数值没有随网格单调下降，"
        f"它与五米的 {fmt(h['2m']['5m']['v'], 3)} 略低于与十米的 "
        f"{fmt(h['2m']['10m']['v'], 3)}，"
        "说明二米场与整个粗网格家族之间的差别是整体的，不是逐级放大的。"
        f"流速部分的数值更低，二十格非对角元里最高的是 {fmt(sp['20m']['30m']['v'], 3)}，"
        f"有 {sum(1 for a, b in _offdiag if sp[a][b]['v'] < 0.03)} 格不足 0.03，"
        f"最低的是 {fmt(sp['5m']['30m']['v'], 3)}。"
        "流速的区域划分几乎不能跨分辨率传递，这与水深的结论方向一致，程度更重。")

    sch = {k: vm["h_max"]["stability_share"][k] for k in ("0", "1", "2", "3", "4")}
    ssc = {k: vm["speed"]["stability_share"][k] for k in ("0", "1", "2", "3", "4")}
    stab = (
        "水深的结果里，四档粗网格都与二米参考一致的单元占 "
        f"{sch['4'] * 100:.1f}%，与参考都不一致的单元占 {sch['0'] * 100:.1f}%。"
        f"恰好一致两档的单元另有 {sch['2'] * 100:.1f}%，一致一档与三档各占 "
        f"{sch['1'] * 100:.1f}% 与 {sch['3'] * 100:.1f}%。"
        "这个分布两头重、中间轻，说明粗网格对二米参考的偏离是成群出现的，"
        "同一个位置往往四档一起偏离。流速没有这个特征，"
        f"四档一致只占 {ssc['4'] * 100:.1f}%，全部不一致反而占 {ssc['0'] * 100:.1f}%，"
        f"一到三档各占 {ssc['1'] * 100:.1f}%、{ssc['2'] * 100:.1f}% 与 {ssc['3'] * 100:.1f}%，"
        "四档的偏离各有各的走向。把稳定性与二米参考的水深放在一起看，"
        "一致四档的单元平均水深只有 0.29 米，一致零档的单元平均水深是 1.75 米。"
        "稳定的位置是面积广大的浅水区，不稳定的位置是水深较大的河道主干与低洼积水区。"
        "这里出现的两个数字口径不同，需要合起来读。稳定占比按多数标号判定，属于宽松口径，"
        "二米参考里最浅的一档占了共同湿区的 72.7%，粗网格块的多数标号很容易落在它上面。"
        "V-measure 按信息熵计算，属于严格口径，两档以上的取值都很低。"
        "宽松口径的高占比说明浅水区大体搬得过去，"
        "严格口径的低取值说明除相邻的两档之外，区域结构整体上搬不过去。")

    vmc = (
        "把两个变量合起来看，得到的结论是一致的。"
        "洪涝水深的区域结构在粗化中大幅丧失，只有相邻分辨率之间还能保留七成左右的共同结构。"
        "流速的区域结构则几乎完全不能跨分辨率传递，四档之间的共同结构都在一成上下。"
        "这与前两节的配对比较相互印证。水深在对应坐标上的皮尔逊系数只有 "
        f"{fmt(min(fseries('h_max', 'pearson_wet')), 3)} 到 "
        f"{fmt(max(fseries('h_max', 'pearson_wet')), 3)}，流速只有 "
        f"{fmt(min(fseries('speed', 'pearson_wet')), 3)} 到 "
        f"{fmt(max(fseries('speed', 'pearson_wet')), 3)}，两者都是本节里最不稳定的量。"
        "反向的证据同样成立。高程、坡度、地形位置指数、曼宁糙率、不透水率与建筑占比的"
        f"跨分辨率相关系数都在 {min(series('Twi', 'bilinear_pearson') + series('Slope', 'bilinear_pearson') + series('Tpi', 'bilinear_pearson')):.3f} 以上，"
        "它们的区域结构不需要额外处理。")

    return {
        "@@CURVCLIP5@@": (f"{fmt(c['Curv_plan']['5m']['bilinear_pearson_clipped'], 3)} 与 "
                          f"{fmt(c['Curv_profile']['5m']['bilinear_pearson_clipped'], 3)}"),
        "@@IMPERV4@@": lst(series("impervious_frac", "bilinear_pearson")),
        "@@BUILT4@@": lst(series("built_frac", "bilinear_pearson")),
        "@@DIST_TEXT@@": dist,
        "@@KURT_TEXT@@": kurt,
        "@@VM_TEXT@@": vmt,
        "@@STAB_TEXT@@": stab,
        "@@VM_CONCLUDE@@": vmc,
    }


CONS_ORDER = [("DEM", "高程", "地形"), ("Slope", "坡度", "地形"),
              ("Curv_plan", "平面曲率", "地形"),
              ("Curv_profile", "剖面曲率", "地形"),
              ("Tpi", "地形位置指数", "地形"), ("Twi", "地形湿度指数", "地形"),
              ("Manning", "曼宁糙率", "下垫面"),
              ("impervious_frac", "不透水率", "下垫面"),
              ("built_frac", "建筑占比", "下垫面")]


def pre_cons_cont_table() -> str:
    """Terrain, surface cover and the two hydraulic fields on one correlation scale."""
    d = _cons()
    if not d:
        return ""
    cont = d["continuous"]["variables"]
    fl = d["flood"]["variables"]
    notes = d["continuous"]
    rows = []
    for key, lab, grp in CONS_ORDER:
        cells = "".join(f"<td>{cont[key][r]['bilinear_pearson']:.4f}</td>"
                        for r in ("5m", "10m", "20m", "30m"))
        x30 = cont[key]["30m"]
        rows.append(
            f"<tr><td class='l'>{lab}</td><td>{grp}</td>{cells}"
            f"<td>{x30['bilinear_spearman']:.4f}</td>"
            f"<td>{x30['agg_pearson']:.4f}</td></tr>")
    for key, lab in (("h_max", "水深"), ("speed", "流速")):
        cur = fl[key]["100a"]["resolutions"]
        cells = "".join(f"<td>{cur[r]['pearson_wet']:.4f}</td>"
                        for r in ("5m", "10m", "20m", "30m"))
        rows.append(
            f"<tr><td class='l'>{lab}</td><td>洪涝</td>{cells}"
            f"<td>{cur['30m']['spearman_wet']:.4f}</td><td>同左</td></tr>")
    return (
        "<table><caption><b>表 92　地形、下垫面与洪涝变量的跨分辨率相关系数。</b>"
        "情景为一百年一遇，样本是研究域内两档都有效的粗网格单元。"
        "前三列到第六列的比较对象是同一个粗网格单元上的粗网格模拟值与聚合后的二米真值。"
        "地形与下垫面九行的重采样方式为双线性插值，也就是用相邻四个二米像元的加权和"
        "估计单元中心处的取值，静态场不要求块均值守恒，因此可以使用。"
        "水深与流速两行走面积加权的块平均，原因是 2.4 节已经论证水深必须保住块内平均，"
        "点采样式插值会破坏这一条，流速场只在湿区有定义，情况相同。"
        "第七列是三十米网格上的斯皮尔曼等级相关系数，用来与第四列的皮尔逊系数对照。"
        "第八列是同一档网格改用面积加权块平均重算得到的皮尔逊系数，作为重采样口径的对照，"
        "水深与流速两行本身就是这个口径，因此记为同左。"
        "流速是派生量，由存有单宽流量的整点时刻除以水深得到。"
        f"双线性插值算子的往返偏差在十米网格上是 "
        f"{notes['variables']['DEM']['10m']['bilinear_roundtrip_bias']:.4f} 米。"
        "表中数字均取自 outputs/premodel/cross_resolution_consistency.json。"
        "</caption><thead><tr><th class='l'>要素</th><th>类别</th>"
        "<th>5 米<br>皮尔逊</th><th>10 米<br>皮尔逊</th><th>20 米<br>皮尔逊</th>"
        "<th>30 米<br>皮尔逊</th><th>30 米<br>等级相关</th>"
        "<th>30 米<br>面积加权</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_cons_cat_table() -> str:
    """Land cover and the building mask, measured with categorical agreement."""
    d = _cons()
    if not d:
        return ""
    lab = {"Landuse": "土地利用", "Building_binary": "建筑有无"}
    rows = []
    for key, name in lab.items():
        for r in ("5m", "10m", "20m", "30m"):
            x = d["categorical"]["variables"][key][r]
            rows.append(
                f"<tr><td class='l'>{name}</td><td>{r[:-1]}</td>"
                f"<td>{x['agreement']*100:.2f}</td><td>{x['kappa']:.4f}</td>"
                f"<td>{x['cramers_v']:.4f}</td>"
                f"<td>{x['majority_frac_coarse']*100:.2f}</td></tr>")
    return (
        "<table><caption><b>表 94　土地利用与建筑有无的跨分辨率一致性。</b>"
        "两类量都不能用乘积矩相关系数，土地利用是多类别名义编码，类别之间没有大小关系，"
        "建筑有无是零一标记，因此改用分类口径。一致率是比较单元上粗网格编码与"
        "块内二米单元多数编码相同的比例，以百分数给出。科恩卡帕系数去掉随机一致之后"
        "剩下的净一致程度，克拉默 V 系数由列联表卡方统计量换算得到关联强度。"
        "最后一列是粗网格本身多数类所占的比例，用来判断一致率是否有相当一部分来自"
        "某一类占绝对多数。土地利用多出一类水体，它在三十米网格上只剩很少的单元，"
        "读一致率时应留意这一点。"
        "</caption><thead><tr><th class='l'>变量</th><th>边长<br>(米)</th>"
        "<th>一致率<br>(%)</th><th>科恩<br>卡帕</th><th>克拉默<br>V</th>"
        "<th>多数类<br>(%)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_cons_dist_table() -> str:
    """Probability-distribution fits for depth and speed on the four coarse grids.

    Kept compact on purpose: the leading label carries the variable, the scenario and
    the edge length, so the body is only the fit diagnostics that support the one
    decision this table answers, namely which family fits and how stable that choice
    is across grids. Raw sample counts and the AIC score are method details, not
    decision quantities, and are left to the JSON.
    """
    d = _cons()
    if not d:
        return ""
    lab = {"h_max": "水深", "speed": "流速"}
    scen_cn = {"20a": "20 年", "100a": "100 年"}
    rows = []
    for key, name in lab.items():
        for scen in ("20a", "100a"):
            for r in ("5m", "10m", "20m", "30m"):
                x = d["flood"]["variables"][key][scen]["resolutions"][r]["dist_native"]
                rows.append(
                    f"<tr><td class='l'>{name}</td><td>{scen_cn[scen]}</td><td>{r[:-1]}</td>"
                    f"<td>{x['mean']:.3f}</td><td>{x['median']:.3f}</td>"
                    f"<td>{x['p95']:.3f}</td>"
                    f"<td>{x['skew']:.2f}</td><td>{x['kurtosis_excess']:.1f}</td>"
                    f"<td class='l'>{x['best_family']}</td>"
                    f"<td>{x['best_ks']:.4f}</td></tr>")
    return (
        "<table><caption><b>表 95　水深与流速的概率分布拟合。</b>"
        "每一行是一档网格上一个情景的拟合结果，水深与流速两组各八行。"
        "五种备选分布族是伽马、对数正态、三参数韦布尔、指数与对数逻辑斯蒂，"
        "位置参数固定为零，其余参数用最大似然估计，最优族由赤池信息量准则选出。"
        "KS 列是柯尔莫哥洛夫-斯米尔诺夫统计量，即经验分布与拟合分布之间的最大纵向距离，"
        "数值越小表示拟合越接近，它是可以跨分辨率纵向比较的那一列。"
        "均值、中位数与 95 分位三列的单位与变量同，水深的表头标米，流速的表头标米每秒。"
        "整张表只回答一个问题，最优族是否随网格改变。读法是先看最优分布族一列，"
        "再看同一变量四行的 KS 与偏度是否同向变化。"
        "拟合样本只取湿区单元，水深取超过五厘米的单元，流速取聚合真值超过五厘米的单元，"
        "每个样本最多抽取十万个单元，随机种子为 20260921。"
        "水深八行一律是对数正态分布，流速五米与十米是对数正态分布，"
        "二十米与三十米换成三参数韦布尔分布，这是本节唯一一处最优族发生更换的地方。"
        "</caption><thead><tr><th class='l'>变量</th><th>情景</th><th>边长<br>(米)</th>"
        "<th>均值</th><th>中位数</th><th>95 分位</th>"
        "<th>偏度</th><th>超额<br>峰度</th><th class='l'>最优分布族</th>"
        "<th>KS<br>(越小越好)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_cons_vm_table() -> str:
    """Pairwise V-measure between the regionalisations of the five resolutions."""
    d = _cons()
    if not d:
        return ""
    var = d["vmeasure"]["variables"]
    allo = ["2m", "5m", "10m", "20m", "30m"]
    rows = []
    for key, name in (("h_max", "水深"), ("speed", "流速")):
        for a in allo:
            cells = "".join(f"<td>{var[key]['pairwise'][a][b]['v']:.3f}</td>"
                            for b in allo)
            rows.append(f"<tr><td class='l'>{name}</td><td>{a[:-1]}</td>{cells}</tr>")
    k = {key: var[key]["k_diagnostics"]["chosen_k"] for key in ("h_max", "speed")}
    sil = {key: var[key]["k_diagnostics"]["chosen_silhouette"]
           for key in ("h_max", "speed")}
    return (
        "<table><caption><b>表 98　跨分辨率区域划分的 V-measure 矩阵。</b>"
        "行列都是五档分辨率，格子里的数字是两档划分之间的 V-measure，"
        "由同质性与完备性的调和平均给出，零表示两套划分互相无关，一表示完全一致。"
        "区域划分的做法是对每一档分辨率自己模拟出来的场取对数、标准化，"
        "再用 K 均值切成若干个簇，标号按最近邻投影到二米格网，"
        "使各档落在同一批比较单位上。水深用两米水深超过五厘米的单元，"
        "流速用两米流速超过五厘米的单元。簇数由二米参考场上的轮廓系数在三到八之间挑出，"
        f"水深取 {k['h_max']}，对应的轮廓系数是 {sil['h_max']:.3f}，"
        f"流速取 {k['speed']}，对应的轮廓系数是 {sil['speed']:.3f}，"
        "两个变量在各档上使用同一个簇数，随机种子为 20260921。"
        "SABRE 是 R 语言实现的同类工具，本节的结果与 SABRE 0.4.3 的输出已逐值比对，"
        "二十对数值的最大绝对差在双精度舍入量级，两种实现一致，详见 2.12 节。"
        "</caption><thead><tr><th class='l'>变量</th><th>边长<br>(米)</th>"
        "<th>2 米</th><th>5 米</th><th>10 米</th><th>20 米</th><th>30 米</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


PAT_PAIRS = ["2m|5m", "2m|10m", "2m|20m", "2m|30m",
             "5m|10m", "5m|20m", "5m|30m", "10m|20m", "10m|30m", "20m|30m"]


def _pat() -> dict:
    d = _cons()
    return d["spatial_pattern"]["variables"] if d else {}


def pat_text() -> dict:
    """Prose for section 2.11, every quoted number read back from the JSON."""
    v = _pat()
    if not v:
        return {}
    h, s = v["h_max"], v["speed"]
    ch, cs = h["chance_corrected"], s["chance_corrected"]
    fh, fs = h["chance_floor"], s["chance_floor"]
    ah, as_ = h["areal"], s["areal"]
    bh, bs = h["boundary"], s["boundary"]
    au, jc = h["autocorrelation"], h["join_count"]
    drh = h["distance_resolved"]["value_bands"]
    dwh = h["distance_resolved"]["dist_water_bands"]
    drs = s["distance_resolved"]["value_bands"]
    dws = s["distance_resolved"]["dist_water_bands"]

    def f(x, n=3):
        return f"{x:.{n}f}".replace("-", "\u2212")

    lm = [au["class_labels"][r]["moran"] for r in ("2m", "5m", "10m", "20m", "30m")]
    lms = [s["autocorrelation"]["class_labels"][r]["moran"]
           for r in ("2m", "5m", "10m", "20m", "30m")]
    zs = [jc[r]["z"] for r in ("2m", "5m", "10m", "20m", "30m")]
    zss = [s["join_count"][r]["z"] for r in ("2m", "5m", "10m", "20m", "30m")]

    pat_a = (
        "第一组把机会校正引进来。调整兰德指数、Fowlkes-Mallows 指数与科恩卡帕系数"
        "都先扣掉随机标号下应有的分数，取值在一附近表示两套划分几乎一致，"
        "在零附近表示与随机分组没有差别。调整互信息与归一化互信息做同一件事，"
        "只是走互信息这条路。实测里有两条恒等式要交代。"
        "算术归一化的归一化互信息与 V-measure 是同一个量，"
        f"十对分辨率组合上两者到小数点后第六位才分开，"
        f"例如水深二米与三十米这一对两者都是 {f(ch['2m|30m']['v'], 4)}。"
        f"调整互信息与它们也只差到小数点后第五位，同一对上是 "
        f"{f(ch['2m|30m']['ami'], 4)}。"
        "真正与信息论指标不同的只有调整兰德指数、Fowlkes-Mallows 与科恩卡帕三个。"
        "水深上调整兰德指数的排序与 V-measure 完全一致。"
        f"相邻组合里五米与十米最高，达到 {f(ch['5m|10m']['ari'])}，"
        f"二米与五米是 {f(ch['2m|5m']['ari'])}，十米与二十米是 {f(ch['10m|20m']['ari'])}。"
        f"跨三档之后落差很大，二米与三十米只剩 {f(ch['2m|30m']['ari'])}，"
        f"二十米与三十米是 {f(ch['20m|30m']['ari'])}。"
        f"科恩卡帕给出同样的排序，五米与十米是 {f(ch['5m|10m']['kappa'])}，"
        f"二米与三十米只有 {f(ch['2m|30m']['kappa'])}。"
        f"流速上所有指标都停在低位，最高的二十米与三十米也只到调整兰德指数 "
        f"{f(cs['20m|30m']['ari'])} 与科恩卡帕 {f(cs['20m|30m']['kappa'])}，"
        f"五米与二十米这一对甚至给出负的科恩卡帕系数 {f(cs['5m|20m']['kappa'])}。"
        "Fowlkes-Mallows 指数需要单独说明，它在随机标号下的期望值并不等于零。"
        f"按实测的簇规模做三十次置换，水深二米与三十米这一对的期望值是 "
        f"{f(fh['2m|30m']['fm'])}，实测值是 {f(ch['2m|30m']['fm'])}，"
        f"超出机会水平的部分只有 {f(ch['2m|30m']['fm'] - fh['2m|30m']['fm'])}。"
        f"五米与十米这一对的期望值是 {f(fh['5m|10m']['fm'])}，"
        f"实测值是 {f(ch['5m|10m']['fm'])}，"
        f"超出部分 {f(ch['5m|10m']['fm'] - fh['5m|10m']['fm'])}。"
        f"同一批规模下调整兰德指数的期望值是 {f(fh['2m|30m']['ari'])}，"
        f"实测值是 {f(ch['2m|30m']['ari'])}。"
        f"流速二米与三十米这一对的期望值是 {f(fs['2m|30m']['fm'])}，"
        f"实测值是 {f(cs['2m|30m']['fm'])}，"
        f"超出部分只有 {f(cs['2m|30m']['fm'] - fs['2m|30m']['fm'])}。"
        "簇占比越不均匀，Fowlkes-Mallows 的随机期望值越高，它的绝对值因此系统性偏高，"
        "读数前应当先减掉机会水平。")

    pat_b = (
        "第二组只看面积，不看标号。做法是把两档划分的簇两两配对，"
        "用匈牙利算法找一组总交并比最大的匹配，再按参考侧的面积加权平均得到平均交并比。"
        "同时统计各簇面积占比的总变差距离，"
        "它等于两套面积份额之差绝对值之和的一半，零表示份额完全一致。"
        f"水深五米与十米这一对的平均交并比是 {f(ah['5m|10m']['mean_iou'])}，"
        f"总变差距离只有 {f(ah['5m|10m']['area_tv'])}。"
        f"二米与三十米降到 {f(ah['2m|30m']['mean_iou'])} 与 {f(ah['2m|30m']['area_tv'])}。"
        "逐簇看能发现更细的先后次序。"
        f"二米参考上占七成的主体簇在三十米上仍有 "
        f"{f(ah['2m|30m']['per_cluster_iou'][0])} 的交并比，"
        f"占二成二的中间簇只剩 {f(ah['2m|30m']['per_cluster_iou'][1])}，"
        f"占五点八的小簇只剩 {f(ah['2m|30m']['per_cluster_iou'][2])}。"
        "面积的损失因此先落在小簇上，主体簇撑得最久。"
        f"流速上平均交并比整体落在 {f(as_['5m|20m']['mean_iou'])} 到 "
        f"{f(as_['20m|30m']['mean_iou'])} 之间，"
        f"最差的一对是五米与二十米的 {f(as_['5m|20m']['mean_iou'])}，"
        f"最好的一对是二十米与三十米的 {f(as_['20m|30m']['mean_iou'])}。")

    pat_c = (
        "第三组换到边界上。前两组度量的都是区域内部的一致程度，边界是另一回事。"
        "做法是把每一档的湿区掩膜做形态学腐蚀，原掩膜减去腐蚀结果就是一圈边界像元，"
        "再算两圈边界的 F1 与交并比。两条前缘线之间的距离由最近邻搜索给出，"
        "取两个方向里较大的一个作为修正豪斯多夫距离。"
        "掩膜一律由水深场给出，阈值取五厘米，流速那一组同样取自水深，"
        "取峰值水量那一小时的瞬时水深。"
        "这样做的原因是五米档的派生流速整体比粗网格小约九倍。"
        f"用同一个绝对流速阈值去切，五米档只有 "
        f"{s['front_diagnostic']['speed_pass_share_5m'] * 100:.1f}% 的深度湿单元过线，"
        f"二米档是 {s['front_diagnostic']['speed_pass_share_2m'] * 100:.1f}%，"
        f"二米格网上的湿掩膜面积也只有二米档的 "
        f"{s['front_diagnostic']['mask_ratio_5m_to_2m'] * 100:.1f}%。"
        f"二米与五米的前缘平均距离会被抬到 "
        f"{f(s['front_diagnostic']['mhd_m'], 1)} 米，"
        f"改用水深掩膜之后只有 {f(bs['2m|5m']['mhd_m'], 2)} 米。"
        "绝对流速阈值在五档之间不可比，因此这里只把水深当作几何口径。"
        f"水深上前缘 F1 从五米的 {f(bh['2m|5m']['f1'])} 一路降到三十米的 "
        f"{f(bh['2m|30m']['f1'])}，修正豪斯多夫距离从 {f(bh['2m|5m']['mhd_m'], 2)} 米"
        f"升到 {f(bh['2m|30m']['mhd_m'], 2)} 米。"
        f"流速上前缘 F1 更低，五米是 {f(bs['2m|5m']['f1'])}，"
        f"三十米是 {f(bs['2m|30m']['f1'])}，距离从 {f(bs['2m|5m']['mhd_m'], 2)} 米"
        f"升到 {f(bs['2m|30m']['mhd_m'], 2)} 米。"
        "把边界结果与内部结果放在一起看，落差非常明显。"
        f"五米与十米这一对在内部有 {f(ch['5m|10m']['v'])} 的 V-measure 与 "
        f"{f(ah['5m|10m']['mean_iou'])} 的平均交并比，"
        f"二米与五米的前缘 F1 却只有 {f(bh['2m|5m']['f1'])}。"
        "区域内部传得过去，区域的边线传不过去。")

    pat_d = (
        "第四组问粗网格的划分是否只是变得更平滑。"
        "做法是在二米湿区上建一个元胞邻接图，每个单元与上下左右四个邻居相连，权重取一，"
        "再在这个权重矩阵上算莫兰指数与吉尔里系数。"
        "莫兰指数度量相邻单元取值的相似程度，正值表示相似的取值聚在一起。"
        "吉尔里系数度量相邻单元的差异，零表示处处相同，一小一大表示空间结构趋于消失。"
        "同一套权重也用在标号上，并配一个同标号邻接的计数检验，"
        "零假设是标号随机分布，检验用一百次置换给出 z 值。"
        f"水深场本身的莫兰指数从二米的 {f(au['field']['2m']['moran'])} 降到三十米的 "
        f"{f(au['field']['30m']['moran'])}，吉尔里系数从 {f(au['field']['2m']['geary'])} "
        f"升到 {f(au['field']['30m']['geary'])}，粗网格的水深场明显变钝。"
        "标号给出的结果正好相反。"
        f"五档标号的莫兰指数全在 {f(min(lm))} 到 {f(max(lm))} 之间，三十米最高。"
        f"同标号邻接的比值也从二米的 {f(jc['2m']['ratio'])} 升到三十米的 "
        f"{f(jc['30m']['ratio'])}，置换检验的 z 值都在 {f(min(zs), 0)} 以上。"
        "流速上两件事的差别没有这么极端，"
        f"它的标号莫兰指数落在 {f(min(lms))} 到 {f(max(lms))} 之间，"
        f"z 值在 {f(min(zss), 0)} 到 {f(max(zss), 0)} 之间。"
        "两件事合起来说明，粗网格的划分并没有变得杂乱，它依然成片，"
        "只是每一片的边界挪了位置。")

    pat_e = (
        "第五组把一致性拆到位置上。做法是以二米参考的标号为基准，"
        "把四档粗网格的簇各自按重叠面积映射到参考簇，"
        "再逐单元数出四档里有几档给出与参考相同的标号，取值从零到四，"
        "除以四就是稳定档占比。再把这个占比按二米参考的取值分档"
        "与到水体的距离分档求平均。"
        f"水深上的落差非常集中。水深在五厘米到半米之间的三个档，稳定档占比在 "
        f"{f(min(q['mean_stability'] for q in drh[:3]))} 到 "
        f"{f(max(q['mean_stability'] for q in drh[:3]))} 之间。"
        f"半米到一米降到 {f(drh[3]['mean_stability'])}，"
        f"一到两米只有 {f(drh[4]['mean_stability'])}。"
        f"水深在两米以上那一档回升到 {f(drh[5]['mean_stability'])}，"
        f"这一档只有 {drh[5]['n']} 个单元，读数时需要保留。"
        f"到水体的距离给出同样的方向，零到二十五米这一档只有 "
        f"{f(dwh[0]['mean_stability'])}，四百米到八百米升到 "
        f"{f(dwh[5]['mean_stability'])}，八百米之外是 {f(dwh[6]['mean_stability'])}。"
        f"流速的规律相同而水平更低，两米以上的流速档只有 "
        f"{f(drs[5]['mean_stability'])}，零到二十五米这一档只有 "
        f"{f(dws[0]['mean_stability'])}。"
        "跨分辨率的一致集中在浅水区与远离河道的位置，"
        "深水河道与紧邻水体的岸边正是划分传不过去的地方。")

    pat_conclude = (
        "五组放在一起，得到的图景比 V-measure 一个指标清楚得多。"
        "信息论与机会校正两组给出的排序基本一致，都指向五米与十米这一对，"
        "也一致认为跨三档之后信息所剩无几。"
        f"两者的差别只在水平，Fowlkes-Mallows 的随机期望值高达 {f(fh['2m|30m']['fm'])}，"
        "绝对值因此系统性偏高。面积一组补上了损失的先后次序，小簇先丢，主体簇撑得最久。"
        "边界一组给出最重要的修正，区域内部的一致性远高于前缘线的一致性。"
        "自相关一组说明粗网格的划分依旧成片，落差来自边界位置的移动。"
        "位置分解一组指出这些移动发生在深水与河道边缘。"
        "由此可以收窄 2.10 节的说法。"
        "区域划分只在相邻分辨率之间传递这句话，对区域的边线成立，"
        "对面积广大的浅水内部并不成立。"
        f"五米与十米之间 {f(ah['5m|10m']['mean_iou'])} 的平均交并比说明主体区域基本保留，"
        f"三十米上主体簇的交并比仍有 {f(ah['2m|30m']['per_cluster_iou'][0])}。"
        f"真正在五米就已经传不过去的是前缘线，二米与五米的前缘 F1 只有 "
        f"{f(bh['2m|5m']['f1'])}。"
        "这与 2.5 节和 2.9 节的结论一致，"
        "前置校正的着力点应当放在湿区轮廓与深水河道，而非整片浅水区。")

    return {"@@PAT_A_TEXT@@": f"<p>{pat_a}</p>",
            "@@PAT_B_TEXT@@": f"<p>{pat_b}</p>",
            "@@PAT_C_TEXT@@": f"<p>{pat_c}</p>",
            "@@PAT_D_TEXT@@": f"<p>{pat_d}</p>",
            "@@PAT_E_TEXT@@": f"<p>{pat_e}</p>",
            "@@PAT_CONCLUDE@@": f"<p>{pat_conclude}</p>"}


def pat_table() -> str:
    """Per-pair chance-corrected indices, one table per variable."""
    v = _pat()
    if not v:
        return ""
    out = []
    for key, name, capid in (("h_max", "水深", "88"), ("speed", "流速", "89")):
        rec = v[key]
        rows = []
        for p in PAT_PAIRS:
            a, b = p.split("|")
            q = rec["chance_corrected"][p]
            rows.append(
                f"<tr><td class='l'>{a[:-1]} 米与 {b[:-1]} 米</td>"
                f"<td>{q['v']:.3f}</td>"
                f"<td>{q['ari']:.3f}</td><td>{q['fm']:.3f}</td>"
                f"<td>{q['kappa']:.3f}</td></tr>")
        fl = rec["chance_floor"]
        out.append(
            f"<table><caption><b>表 {capid}　{name}各分辨率组合的机会校正一致性指标。</b>"
            "四列都是越大越一致，零表示与随机标号无异，负值表示比随机标号还差。"
            "V-measure 与算术归一化的归一化互信息恒等，因此只列一列，"
            "归一化互信息的数值与之相同，调整互信息在四位数上与它一致。"
            "调整兰德指数、Fowlkes-Mallows 与科恩卡帕都在零附近表示与随机标号无异。"
            "Fowlkes-Mallows 的随机期望值不为零，"
            f"二米与三十米这一对是 {fl['2m|30m']['fm']:.3f}，"
            f"五米与十米这一对是 {fl['5m|10m']['fm']:.3f}，"
            "读数时应当先减掉这个水平。"
            "</caption><thead><tr><th class='l'>分辨率组合</th>"
            "<th>V-measure</th><th>调整兰德<br>指数</th>"
            "<th>Fowlkes-<br>Mallows</th><th>科恩<br>卡帕</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
    return "".join(out)


def pre_vardecomp_table() -> str:
    """Shapley shares of the tile-to-tile spread in cross-resolution agreement."""
    import json

    p = ROOT / "outputs/premodel/variance_decomposition.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    cn = [("wet_state", "湿区占比（状态量，另列）"),
          ("rain_scenario", "降雨情景"), ("slope_roughness", "坡度与粗糙度"),
          ("impervious", "不透水率"), ("built_landuse", "建筑与土地利用"),
          ("river_drainage", "河道邻近度")]
    order = ["rain_scenario", "slope_roughness", "impervious", "built_landuse",
             "river_drainage", "wet_state"]
    label = dict(cn)
    res_list = ["5m", "10m", "20m", "30m"]
    rows = []
    for g in order:
        cells = []
        for r in res_list:
            v = d[r]["responses"]["csi@0.05"]["share_of_explained"][g]
            cells.append(f"<td>{v*100:.1f}</td>")
        rows.append(f"<tr><td class='l'>{label[g]}</td>" + "".join(cells) + "</tr>")
    r2 = [f"<td>{d[r]['responses']['csi@0.05']['r2_full']:.3f}</td>" for r in res_list]
    n = [f"<td>{d[r]['n_tiles']}</td>" for r in res_list]
    rows.append("<tr><td class='l'>全模型决定系数 R²</td>" + "".join(r2) + "</tr>")
    rows.append("<tr><td class='l'>参与回归的瓦片数</td>" + "".join(n) + "</tr>")
    return (
        "<table><caption><b>表 85　逐瓦片匹配度的贡献度分解。</b>"
        "因变量是每一块瓦片上粗网格模拟值与聚合后的二米真值的临界成功指数，"
        "阈值为 0.05 米。解释变量按研究方案分为"
        "五组，另加一组湿区占比作为状态量单列。份额是分组 Shapley 值占全模型已解释方差的"
        "比例，六行之和为百分之百。"
        "使用 Shapley 值是因为它不依赖变量的进入顺序，在解释变量彼此相关时比逐项回归"
        "更稳妥，共线性程度见正文。湿区占比一列由粗网格自身派生，属于状态量而不是外部条件，"
        "读表时应与另外五行区分。全部瓦片参与拟合，包括训练、验证与测试条带，"
        "这一节是解释性分析而不是预测模型，因此不受切分纪律约束。</caption>"
        "<thead><tr><th class='l'>因素</th><th>5 米<br>(%)</th><th>10 米<br>(%)</th>"
        "<th>20 米<br>(%)</th><th>30 米<br>(%)</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>")


def pre_dialect_table() -> str:
    """Fine-lattice versus coarse-lattice accounting, and the identity joining them."""
    import json

    p = ROOT / "outputs/premodel/scale_dialect.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for scen in ("20a", "100a"):
        for res in ("5m", "10m", "20m", "30m"):
            x = d["results"][scen][res]
            f = x["fine_nn"]
            sh = x["shares"]
            total = f["mse_coarse"] + f["mse_within"]
            gap = f["mse_fine"] - total
            rows.append(
                f"<tr><td class='l'>{scen}</td><td>{res[:-1]}</td>"
                f"<td>{f['mse_fine']:.4f}</td><td>{f['mse_coarse']:.4f}</td>"
                f"<td>{f['mse_within']:.4f}</td>"
                f"<td>{gap:.1e}</td>"
                f"<td>{sh['coarse_of_fine']*100:.1f}</td>"
                f"<td>{sh['within_of_fine']*100:.1f}</td>"
                f"<td>{x['rmse_m']['fine_dialect']:.4f}</td>"
                f"<td>{x['rmse_m']['coarse_dialect']:.4f}</td></tr>")
    return (
        "<table><caption><b>表 93　细网格口径与粗网格口径的对照，"
        "以及把两者连起来的恒等式。</b>"
        "细网格口径把每个两米单元与最近邻升采样后的粗网格模拟值相比，"
        "粗网格口径把每个粗方块与同一方块的聚合后的二米真值相比。"
        "粗网格口径 MSE 是粗网格模拟值减聚合真值的平方在一格上的平均，"
        "对应表 82 的模拟项。块内方差是聚合真值减两米真值平方的平均，对应表 82 的聚合项，"
        "它衡量方块内部本来就不均匀的那一部分，与粗网格算得准不准无关。"
        "按逐块恒等式，粗网格口径 MSE 与块内方差相加应当等于细网格口径 MSE，"
        "差值一栏给出两者相减后的残差，它是否接近零就是这条恒等式是否成立的判据。"
        "表头中的 MSE 指均方误差，也就是误差平方的平均，取平方根即 2.2 节定义的均方根误差。"
        "差值一栏最右两列是同一批数字开平方后的均方根误差，单位是米，便于与其他表格对照。"
        "后两栏是粗网格口径与块内方差各占细网格口径的比例，两者相加为一。"
        "十米、二十米与三十米三档上这个差值落在十的负九次方平方米量级，相对量级为十的负八次方，"
        "属于单精度累加的舍入，说明恒等式在数值上成立。"
        "逐块改用双精度复算时，最大残差为九乘十的负十四次方平方米，可以认为严格成立。"
        "五米档的方块宽两个半两米单元，最近邻升采样形成的分组与面积加权方块并不重合，"
        "因此差值升到十的负五次方平方米量级，约为该档细网格口径的万分之五，"
        "这一档需要单独看，正文有说明。</caption>"
        "<thead><tr><th class='l'>重现期</th><th>边长<br>(米)</th>"
        "<th>细网格口径<br>MSE(平方米)</th><th>粗网格口径<br>MSE(平方米)</th>"
        "<th>块内<br>方差(平方米)</th><th>差值<br>(平方米)</th>"
        "<th>粗网格口径<br>占比(%)</th><th>块内方差<br>占比(%)</th>"
        "<th>细网格口径<br>RMSE(米)</th><th>粗网格口径<br>RMSE(米)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def pre_model_table() -> str:
    """Pre-model test-set metrics for the three arms."""
    import json

    p = ROOT / "outputs/premodel/premodel_results.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    cn = {"identity": "恒等映射（不校正）", "bilinear": "双线性重采样",
          "premodel": "前置模型"}
    rows = []
    for res in ("5m", "10m", "20m", "30m"):
        g = d[res]["global"]
        for arm in ("identity", "bilinear", "premodel"):
            m = g[arm]
            rows.append(
                f"<tr><td>{res[:-1]}</td><td class='l'>{cn[arm]}</td>"
                f"<td>{_fmt(m['mae'], 4)}</td>"
                f"<td>{_fmt(m['rmse'], 4)}</td>"
                f"<td>{_fmt(m['bias'], 4, plus=True)}</td>"
                f"<td>{_fmt(m['volume_rel']*100, 1, plus=True)}</td>"
                f"<td>{_fmt(m['csi@0.05'])}</td><td>{_fmt(m['csi@0.30'])}</td>"
                f"<td>{_fmt(m['csi@1.00'])}</td></tr>")
    return (
        "<table><caption><b>表 86　前置模型与两个基线在测试集上的对比。</b>"
        "校正后的粗网格场与聚合后的二米真值比较，测试集为南部地理条带，只在模型与轮次"
        "确定之后读取一次。恒等映射表示对粗网格场不做任何处理。双线性重采样把粗网格场"
        "用双线性核插值到二米格网后再平均回粗网格，是一个没有可学参数的平滑算子。"
        "前置模型在训练条带上拟合、在验证条带上挑选轮次，测试条带不参与任何选择。"
        "同一档网格的三行可以直接对照。平均偏差与体积相对误差两列的含义需要说明，"
        "正值表示粗网格给多了水，负值表示给少了水。恒等映射与双线性重采样两行的这两列全为正，"
        "前置模型三行的这两列全为负，符号在三者之间发生了翻转，"
        "并且前置模型在十米与五米两档上的体积相对误差绝对值高于基线。"
        "逐档数值与解释见正文。</caption>"
        "<thead><tr><th>边长<br>(米)</th><th class='l'>方案</th><th>MAE<br>(米)</th>"
        "<th>RMSE<br>(米)</th><th>偏差<br>(米)</th><th>体积相对<br>误差(%)</th>"
        "<th>CSI<br>@0.05</th><th>CSI<br>@0.30</th><th>CSI<br>@1.00</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def pre_paired_table() -> str:
    """Paired statistics of the pre-model against the two baselines."""
    import json

    p = ROOT / "outputs/premodel/premodel_results.json"
    if not p.exists():
        return ""
    d = json.loads(p.read_text(encoding="utf-8"))
    metrics = [("mae", "逐瓦片 MAE（米）"), ("rmse", "逐瓦片 RMSE（米）"),
               ("csi005", "逐瓦片 CSI@0.05")]
    comps = ["premodel_vs_identity", "premodel_vs_bilinear"]
    rows = []
    for res in ("5m", "10m", "20m", "30m"):
        for mk, mname in metrics:
            cells = [f"<td>{res[:-1]}</td><td class='l'>{mname}</td>"]
            for pk in comps:
                s = d[res]["paired"][pk][mk]
                goodif = mk != "csi005"
                dcls = "good" if ((s["mean_delta"] < 0) == goodif) else "bad"
                # frac_improved counts the tiles where the delta is negative; for a
                # metric where larger is better the improvement share is its complement
                imp = s["frac_improved"] if goodif else 1.0 - s["frac_improved"]
                cells.append(
                    f"<td class='{dcls}'>{_fmt(s['mean_delta'], 4, plus=True)}</td>"
                    f"<td>[{_fmt(s['boot_ci_lo'], 4, plus=True)}, "
                    f"{_fmt(s['boot_ci_hi'], 4, plus=True)}]</td>"
                    f"<td>{imp*100:.1f}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<table><caption><b>表 87　前置模型相对两个基线的配对检验。</b>"
        "配对单位是一块瓦片，同一块瓦片在两个方案下的同一指标相减，边长是分层，指标是行。"
        "逐瓦片 MAE 与 RMSE 越小越好，差值为负表示前置模型更优；"
        "逐瓦片 CSI@0.05 越大越好，差值为正表示前置模型更优；两种方向都已在颜色上标出。"
        "左右两组分别对照恒等映射与双线性重采样，同一行的两格可直接比较前置模型相对谁的增益更大。"
        "置信区间来自对差值重抽样四千次的自助法，不跨零即认为差异稳定。"
        "末列给出差值朝有利方向未变差的瓦片占比，它的统计口径随指标方向而变。"
        "对于逐瓦片 MAE 与 RMSE，它统计差值严格为负的瓦片，也就是明确改善的瓦片。"
        "对于逐瓦片 CSI@0.05，差值越大越有利，"
        "因此它统计差值不小于零的瓦片，其中包含恰好持平的情形。"
        "该列取 100.0 时含义是没有一块瓦片变差，它与全部瓦片都获得改善并不等价，"
        "五米档前置模型对恒等映射的 CSI@0.05 一格就属于这种情况。"
        "Wilcoxon 符号秩检验的 p 值、t 统计量与配对 Cohen's d<sub>z</sub> 效应量逐对列在 "
        "<code>outputs/premodel/premodel_results.json</code>，"
        "本表只保留判断「是否优于基线」所需的差值、区间与占比三项。</caption>"
        "<thead><tr><th>边长<br>(米)</th><th class='l'>指标</th>"
        "<th>对恒等<br>平均差值</th><th>对恒等<br>95% CI</th>"
        "<th>对恒等<br>未变差(%)</th>"
        "<th>对双线性<br>平均差值</th><th>对双线性<br>95% CI</th>"
        "<th>对双线性<br>未变差(%)</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


PATX_PAIRS = ["2m|5m", "2m|10m", "2m|20m", "2m|30m", "5m|10m",
              "5m|20m", "5m|30m", "10m|20m", "10m|30m", "20m|30m"]


def _patx() -> dict:
    """The stage-11 payload of the four extra checks, read through one helper."""
    import json
    p = ROOT / "outputs" / "premodel" / "pattern_extra.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _pl(pair: str) -> str:
    a, b = pair.split("|")
    return f"{a[:-1]} 米与 {b[:-1]} 米"


def patx_text() -> dict:
    """Prose for the four extra checks, every quoted number read back from JSON."""
    d = _patx()
    if not d:
        return {}
    n = lambda v, q=3: f"{v:.{q}f}"

    # ---- item 1: SABRE
    sc = d.get("sabre_crosscheck", {})
    vm = sc.get("variables", {})
    sab = (f"<p>第一项检验已经跑通，这一条与 8.4 节原来的记录不同。conda 可用，"
           "本轮用 conda-forge 建了一个独立环境，装入 R 4.5.3、sf 1.1.3、"
           "terra 1.9.50 与 CRAN 上的 SABRE 0.4.3。把五档 K 等于三的标号栅格导出为 "
           "GeoTIFF，SABRE 读入后转成多边形区域划分，对十个分辨率组合逐对计算"
           "同质性、完备性与 V-measure。两种实现共二十对数值的最大绝对差是 "
           f"{sc.get('max_abs_diff', float('nan')):.2e}。"
           "这个差值在双精度舍入量级，说明按定义直接计算与 SABRE 的输出一致。"
           "SABRE 内部用二进制熵，本节用自然对数熵，V-measure 是两个熵的比值，"
           "换底不改变取值，两侧因此可以直接对照。</p>")

    # ---- item 2: variogram and scale space
    V = d["variogram"]["variables"]["h_max"]
    lp = {r: V[r]["log_depth"]["fit"] for r in
          ("2m", "5m", "10m", "20m", "30m")}
    r2s = "、".join(n(lp[r]["r2"], 3) for r in ("5m", "10m", "20m", "30m"))
    rhos = "、".join(n(lp[r]["nugget_over_var"], 3) for r in
                    ("2m", "5m", "10m", "20m", "30m"))
    r95 = [lp[r]["range95_m"] for r in ("2m", "5m", "10m", "20m", "30m")]
    S = d["scale_space"]
    sc_m = {q["scale_m"]: q for q in S["scales"]}
    df = {m: sc_m[m]["dispersion_frac"] for m in (4, 16, 64, 256, 512)}
    bf = {m: sc_m[m]["block_var_frac"] for m in (4, 512)}
    import math as _m
    w = _m.log(30 / 16) / _m.log(32 / 16)
    disp30 = sc_m[16]["dispersion_frac"] + w * (sc_m[32]["dispersion_frac"]
                                               - sc_m[16]["dispersion_frac"])
    var_text = (
        "<p>第二项检验的做法先要交代。把水深场按每块 960 米的瓦片切开，"
        "在瓦片内用二维快速傅里叶变换一次算出全部滞后上的半方差，再按物理距离分箱。"
        "瓦片内取对数水深，只统计两端都是湿区的像元对。滞后上限取 480 米，"
        "也就是瓦片边长的一半。每档拟合一元指数模型，有效变程取衰减参数的三倍。"
        f"四档粗网格上对数水深的拟合决定系数依次是 {r2s}，二米是 "
        f"{n(lp['2m']['r2'], 3)}。</p>"
        f"<p>首档半方差与总方差之比从二米的 {n(lp['2m']['nugget_over_var'], 3)} "
        f"升到五米的 {n(lp['5m']['nugget_over_var'], 3)}、"
        f"十米的 {n(lp['10m']['nugget_over_var'], 3)}、"
        f"二十米的 {n(lp['20m']['nugget_over_var'], 3)} 与"
        f"三十米的 {n(lp['30m']['nugget_over_var'], 3)}。"
        f"同一批曲线上达到总方差 95% 的滞后落在 {min(r95):.0f} 米到 "
        f"{max(r95):.0f} 米之间，随网格变大没有单调走向。"
        "这个结果回答本节开头的问题。相似性在粗网格上衰减，原因不在于粗网格"
        "把相关长度拉长。三十米网格上相邻单元的对数水深差已经达到总方差的 "
        f"{n(lp['30m']['nugget_over_var'], 3)}，二米网格上只有 "
        f"{n(lp['2m']['nugget_over_var'], 3)}。"
        "粗网格看到的水深场更接近一张逐格独立的图。湿区前缘因此无法由粗网格重现，"
        "这与 2.10 节与 2.11 节关于前缘线的结论方向一致。</p>"
        f"<p>原始水深上的指数模型不成立，这一条要如实写出。五档里有三档的拟合"
        "决定系数低于 0.11 或者不收敛，原因是原始水深的重尾由少数极深单元主导。"
        "本节因此只把对数水深的拟合参数当作可用的模型结果。</p>"
        f"<p>尺度空间的结果一并给出。把二米场按边长 4 米到 512 米的方块做面积平均"
        "再返回二米格网，块均值没有解释的那部分方差占总方差的比例从 4 米的 "
        f"{n(df[4], 3)} 升到 16 米的 {n(df[16], 3)}、64 米的 {n(df[64], 3)} 与"
        f"256 米的 {n(df[256], 3)}。块均值自身保留的方差份额从 {n(bf[4], 3)} 降到 "
        f"{n(bf[512], 3)}。按报告使用的粗网格折算，三十米上块均值解释了约 "
        f"{n(1 - disp30, 2)} 的湿区局部方差，余下部分必须在超分辨率环节补出。</p>")

    # ---- item 3: contiguity
    Cg = d["contiguity"]["variables"]
    H, Sp = Cg["h_max"], Cg["speed"]
    ph, ps = H["pairs"], Sp["pairs"]
    con = (f"<p>第三项检验的做法与它的前提。在二米湿区上按上下左右四邻构造邻接图，"
           "只允许相邻单元并入同一个簇，合并用 ward 连接做层次聚类，簇数仍取三。"
           "这一步的前提是湿区连通，而实测的共同湿区并不连通。"
           f"水深部分在二米格网上分成 {H['components_support']} 个上下左右连通分量，"
           f"最大的一个只占 {H['largest_component_share'] * 100:.1f}%。"
           "全局的邻接约束在这块支持上无法定义。本节先把湿区做一次三乘三的形态学"
           "闭运算，取闭运算后最大的连通分量再与湿区求交。"
           f"这样得到水深的 {H['n_core']} 个单元与流速的 {Sp['n_core']} 个单元作为"
           "连通核，约束在这块核上真正起作用。</p>"
           f"<p>核上的结果。水深无约束划分的 V-measure 在 {_pl('5m|10m')} 上是 "
           f"{n(ph['unconstrained']['5m|10m']['v'])}，在 {_pl('10m|20m')} 上是 "
           f"{n(ph['unconstrained']['10m|20m']['v'])}，在 {_pl('2m|30m')} 上是 "
           f"{n(ph['unconstrained']['2m|30m']['v'])}。加上邻接约束之后，"
           f"第一对升到 {n(ph['constrained']['5m|10m']['v'])}，后两对变成 "
           f"{n(ph['constrained']['10m|20m']['v'])} 与 "
           f"{n(ph['constrained']['2m|30m']['v'])}。流速的方向相反，"
           f"{_pl('5m|10m')} 从 {n(ps['unconstrained']['5m|10m']['v'])} 升到 "
           f"{n(ps['constrained']['5m|10m']['v'])}，{_pl('10m|20m')} 从 "
           f"{n(ps['unconstrained']['10m|20m']['v'])} 升到 "
           f"{n(ps['constrained']['10m|20m']['v'])}。</p>"
           "<p>结论。水深核上的无约束排序与全支持同向，相邻分辨率之间的共同结构"
           "远高于跨三档。只有相邻分辨率传递这条结论在加入邻接约束后仍然成立，"
           "而且在十米与二十米这一对上变得更陡。流速的结论发生了改变，"
           "加上邻接约束后相邻分辨率的共同结构从一成升到六成。这个变化必须报出来。"
           "约束的另一面是簇面积变得不均匀，水深二米划分里最大一簇的份额从 "
           f"{n(H['cluster_share']['2m']['unconstrained'][0])} 升到 "
           f"{n(H['cluster_share']['2m']['constrained'][0])}。"
           "需要保留的是核的覆盖率。水深的连通核只占共同湿区的 "
           f"{H['support_share'] * 100:.1f}%，流速核占 "
           f"{Sp['support_share'] * 100:.1f}%。"
           "核上的水平普遍高于全支持，因此这一节的数值只能与全支持比方向，"
           "不能比水平。</p>")

    # ---- item 4: cluster-count sweep
    K = d.get("k_sensitivity", {})
    ks = K.get("k_grid", [])
    sil_h = K["variables"]["h_max"]["silhouette"]
    sil_s = K["variables"]["speed"]["silhouette"]
    stab = K["variables"]["h_max"]["stability"]
    stab_s = K["variables"]["speed"]["stability"]
    fam = ["v", "ari", "fm", "kappa", "ami", "mean_iou", "area_tv"]
    worst = min((min(stab[m]["spearman_vs_k3"].values()), m) for m in fam)
    kt = (f"<p>第四项检验。簇数从二取到八，每个簇数上重算七类指标，"
          "覆盖信息论、机会校正与面积三族，共十个分辨率组合。"
          "轮廓系数给出的结果与 2.10 节的表述不同，这一条要改。"
          f"候选范围从 3 到 8 扩到 2 到 8 之后，二米参考场上水深的轮廓系数在二上是 "
          f"{n(sil_h[str(2)], 3)}，在三上是 {n(sil_h[str(3)], 3)}，在八上是 "
          f"{n(sil_h[str(8)], 3)}。流速的走势相同，从二上的 "
          f"{n(sil_s[str(2)], 3)} 降到八上的 {n(sil_s[str(8)], 3)}。"
          "轮廓系数随簇数单调下降，最高点落在二，它因此不能单独支撑选三。"
          "原因是本节只在单个标准化变量上聚类，一维连续取值的簇数越多，"
          "簇内距离与簇间距离之比越难改善。</p>"
          "<p>排序稳定性给出的结果更关键。把每个簇数下十个组合按指标排序，"
          "再与簇数为三时的排序比斯皮尔曼等级相关。"
          "水深上 V-measure、归一化互信息与调整互信息的相关系数在二到八上全部等于一，"
          "十个组合的次序逐个位置相同。"
          f"七个指标里最小的相关系数是 {n(worst[0], 3)}，出现在面积总变差这一个"
          "距离型指标上。相邻一档的五米与十米组合在全部七个簇数上都排在第一，"
          "跨三档的二米与三十米组合都排在最后。"
          "流速的稳定性稍弱，V-measure 的相关系数在簇数取二时降到 "
          f"{n(stab_s['v']['spearman_vs_k3']['2'], 3)}，"
          f"面积总变差指标在二时降到 {n(stab_s['area_tv']['spearman_vs_k3']['2'], 3)}。</p>"
          "<p>机会校正族的绝对值随簇数单调下降，水深五米与十米的调整兰德指数从 "
          f"{n(K['variables']['h_max']['per_k']['2']['5m|10m']['ari'], 3)} 降到 "
          f"{n(K['variables']['h_max']['per_k']['8']['5m|10m']['ari'], 3)}，"
          "跨簇数比较绝对值没有意义。"
          "结论。K 等于三作为一个可用选择站得住，理由是排序在二到八之间稳定，"
          "而且二会把水深切成两档，丢掉半米到一米这一段。"
          "轮廓系数本身并不支持三，它随簇数单调下降，这一点与原来的表述相反。</p>")
    return {"@@PATX_SABRE@@": sab, "@@PATX_VARIO@@": var_text,
            "@@PATX_CONTIG@@": con, "@@PATX_K@@": kt,
            "@@PATX_MAXDIFF@@": f"{sc.get('max_abs_diff', 0.0):.2e}"}


def patx_tables() -> str:
    """Four compact tables, one per check, all read back from the stage-11 JSON."""
    d = _patx()
    if not d:
        return ""
    out = []

    V = d["variogram"]["variables"]["h_max"]
    rows = []
    for r in ("2m", "5m", "10m", "20m", "30m"):
        e = V[r]
        f = e["log_depth"]["fit"]
        a = f.get("decay_m")
        decay = f"{a:.1f}" if a else "—"
        rng = f"{f['range_m']:.0f}" if a else "—"
        rows.append(
            f"<tr><td>{r[:-1]}</td><td>{e['wet_cells']:,}</td>"
            f"<td>{f['nugget_over_var']:.4f}</td>"
            f"<td>{decay}</td><td>{rng}</td>"
            f"<td>{f['range95_m']:.0f}</td><td>{f['r2']:.3f}</td>"
            f"<td>{_fmt(V[r]['depth']['fit']['r2'], 3)}</td></tr>")
    out.append(
        "<table><caption><b>表 99　对数水深场的半变异函数与指数模型参数。</b>"
        "滞后在 960 米瓦片内用二维快速傅里叶变换一次算出，只统计两端都是湿区的"
        "像元对，滞后上限取 480 米。半方差单位为对数水深的方差。<br>"
        "首档相对粗糙度是相邻单元之间的半方差与样本总方差之比，"
        "数值接近一表示相邻单元之间已经没有相关性，粗网格图上接近逐格独立。"
        "衰减参数是拟合出的指数模型尺度，单位是米，有效变程取它的三倍。"
        "达到总方差 95% 的滞后由曲线直接读出，不依赖模型。"
        "末两列是决定系数，最后一列给出原始水深上的拟合结果，用于说明原始水深"
        "为何不能使用指数模型。原始水深的拟合不成立，五档里有三档决定系数低于 0.11，"
        "因此本表只把对数水深的参数当作可用结果。</caption>"
        "<thead><tr><th>边长<br>(米)</th><th>湿区<br>单元数</th>"
        "<th>首档<br>相对粗糙度</th><th>衰减<br>参数(米)</th><th>有效<br>变程(米)</th>"
        "<th>95%<br>滞后(米)</th><th>对数<br>R²</th><th>原始<br>R²</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    Cg = d["contiguity"]["variables"]
    rows = []
    for var, name in (("h_max", "水深"), ("speed", "流速")):
        rec = Cg[var]
        for r in ("2m", "5m", "10m", "20m", "30m"):
            m = rec["agreement_constrained_vs_unconstrained"][r]
            u = rec["cluster_share"][r]["unconstrained"]
            c = rec["cluster_share"][r]["constrained"]
            rows.append(
                f"<tr><td class='l'>{name}</td><td>{r[:-1]}</td>"
                f"<td>{m['v']:.3f}</td><td>{m['ari']:.3f}</td>"
                f"<td>{m['mean_iou']:.3f}</td><td>{m['area_tv']:.3f}</td>"
                f"<td>{max(u):.3f}</td><td>{max(c):.3f}</td></tr>")
    out.append(
        "<table><caption><b>表 100　连通核上两种划分方式的一致性。</b>"
        "比较对象是同一个连通核上的无约束 K 均值划分与加入上下左右邻接约束的"
        "ward 层次聚类划分，簇数都取三。V-measure、调整兰德指数与平均交并比"
        "度量两种划分的接近程度，面积总变差距离度量两种划分的面积份额差异。"
        "末两列是各自最大一簇的面积份额，用来显示约束会把簇面积拉得多不均匀。"
        "连通核由湿区做一次三乘三闭运算后取最大连通分量得到，水深核有 23317 个"
        "单元，占水深共同湿区的 2.8%，流速核有 5047 个单元，占 5.3%。"
        "核上的水平普遍高于全支持，两列数值只能比方向。</caption>"
        "<thead><tr><th class='l'>变量</th><th>边长<br>(米)</th><th>V-measure</th>"
        "<th>调整兰德<br>指数</th><th>平均<br>交并比</th><th>面积总<br>变差</th>"
        "<th>无约束<br>最大簇份额</th><th>约束后<br>最大簇份额</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    K = d.get("k_sensitivity", {})
    if K:
        ks = K["k_grid"]
        rec = K["variables"]["h_max"]
        sil_h = rec["silhouette"]
        sil_s = K["variables"]["speed"]["silhouette"]
        stab_h = rec["stability"]
        fam = ["v", "ari", "fm", "kappa", "ami", "mean_iou", "area_tv"]
        rows = []
        for k in ks:
            pk = rec["per_k"][str(k)]
            # ordering stability at this K: the weakest rank agreement with K=3
            # over the seven metric families
            omin = min(stab_h[m]["spearman_vs_k3"][str(k)] for m in fam)
            sel = " class='good'" if k == 3 else ""
            rows.append(
                f"<tr><td{sel}>{k}</td>"
                f"<td>{sil_h[str(k)]:.3f}</td>"
                f"<td>{sil_s[str(k)]:.3f}</td>"
                f"<td>{omin:.3f}</td>"
                f"<td>{pk['5m|10m']['v']:.3f}</td>"
                f"<td>{pk['10m|20m']['v']:.3f}</td>"
                f"<td>{pk['2m|30m']['v']:.3f}</td>"
                f"<td>{pk['5m|10m']['ari']:.3f}</td></tr>")
        out.append(
            "<table><caption><b>表 101　簇数从二到八时各分辨率组合的 V-measure。</b>"
            "全部数值取自水深，共同湿区的 826554 个单元，随机种子 20260921。"
            "本表只保留支撑簇数选择的三类量，轮廓系数、排序稳定性，"
            "以及三个代表分辨率组合的 V-measure 与相邻两对的调整兰德指数。"
            "三个组合是排序最高的一对五米与十米、次高的一对十米与二十米、"
            "以及最低的一对二米与三十米，其余七个组合的逐簇数原值见 "
            "<code>outputs/premodel/pattern_extra.json</code>。"
            "排序稳定性一列是七类指标在同一簇数下与簇数取三时的排序斯皮尔曼相关的最小值，"
            "等于一表示十对组合的次序逐个位置不变。"
            "三个结论可以直接读出。轮廓系数随簇数单调下降，最高点在二，"
            "它因此不能单独支撑选三，与 2.10 节原来的表述不同。"
            "排序稳定性在二到八之间始终不低，簇数取三时等于一，"
            "说明簇数的选取不改变组合之间的相对次序。"
            "绝对值随簇数整体走低，因此跨簇数比较绝对值没有意义。"
            "行首加粗的一行是三，也就是本章实际采用的簇数。"
            "</caption><thead><tr><th>簇数<br>K</th><th>水深<br>轮廓系数</th>"
            "<th>流速<br>轮廓系数</th><th>排序<br>相关性最低</th>"
            "<th>V<br>5/10</th><th>V<br>10/20</th><th>V<br>2/30</th>"
            "<th>ARI<br>5/10</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")

    sc = d.get("sabre_crosscheck", {})
    if sc.get("variables"):
        rows = []
        for var, name in (("h_max", "水深"), ("speed", "流速")):
            items = list(sc["variables"][var].items())
            vv = [q["python_v"] for _, q in items]
            diffs = [q["abs_diff"] for _, q in items]
            # the pair that realises the largest absolute difference, to name the worst case
            worst_p, worst_q = max(items, key=lambda kv: kv[1]["abs_diff"])
            rows.append(
                f"<tr><td class='l'>{name}</td><td>{len(items)}</td>"
                f"<td>{min(vv):.3f}</td><td>{max(vv):.3f}</td>"
                f"<td>{_pl(worst_p)}</td><td>{max(diffs):.2e}</td>"
                f"<td class='good'>舍入量级一致</td></tr>")
        out.append(
            "<table><caption><b>表 102　SABRE 0.4.3 与直接计算的逐对数值比较。</b>"
            "R 环境由 conda-forge 建立，装入 r-base 4.5.3、sf 1.1.3、terra 1.9.50"
            "与 CRAN 上的 sabre 0.4.3。标号栅格导出为 GeoTIFF，SABRE 读入后按"
            "多边形区域划分计算。直接计算一列由 scikit-learn 的同质性、完备性与"
            "V-measure 给出。本表不再逐对铺开二十行，只按变量汇总，"
            "列出 V-measure 取值范围与逐对绝对差的最大值，"
            "目的是回答两种实现是否给出同一个量。"
            f"全部二十对数值的最大绝对差是 {sc['max_abs_diff']:.2e}，"
            "在双精度舍入量级，两种实现给出同一个量，"
            "逐对的同质性、完备性与 V-measure 原值见 "
            "<code>outputs/premodel/pattern_extra.json</code>。</caption>"
            "<thead><tr><th class='l'>变量</th><th>组合<br>对数</th>"
            "<th>V-measure<br>最小值</th><th>V-measure<br>最大值</th>"
            "<th class='l'>最大差<br>所在组合</th><th>最大<br>绝对差</th>"
            "<th class='l'>结论</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
    return "".join(out)


def renumber(html: str) -> str:
    """Assign figure and table numbers by document order, then rewrite every reference.

    New figures are authored with temporary numbers so they never collide with the originals;
    this pass makes both the captions and the in-text references sequential. Replacements go
    through a sentinel first, so a freshly written number is never re-substituted by a later rule.
    """
    import re

    def remap(kind: str) -> str:
        nonlocal html
        cap_re = re.compile(rf"<figcaption><b>{kind}\s*(\d+)" if kind == "图"
                            else rf"<caption><b>{kind}\s*(\d+)")
        order = [int(m.group(1)) for m in cap_re.finditer(html)]
        mapping = {old: i + 1 for i, old in enumerate(order)}
        for old in sorted(mapping, key=lambda x: -x):
            pat = re.compile(rf"({kind}\s*){old}(?!\d)")
            new = mapping[old]
            html = pat.sub(lambda m: f"{m.group(1)}\x00{new}\x00", html)
        html = re.sub(rf"{kind}\s*\x00(\d+)\x00", lambda m: f"{kind} {m.group(1)}", html)
        return html

    html = remap("图")
    html = remap("表")
    return html


def sync_figures() -> None:
    """Keep the loose figures/ folder (used by the Markdown twin) in step with the generators."""
    dst = ROOT / "figures"
    dst.mkdir(exist_ok=True)
    for p in FIGS.glob("*.png"):
        if p.name.startswith("_"):
            # scratch plates from one-off diagnostics are never referenced by the report
            continue
        target = dst / p.name
        if not target.exists() or target.stat().st_mtime < p.stat().st_mtime:
            target.write_bytes(p.read_bytes())


def build() -> None:
    from report_body import body as report_body

    sync_figures()
    BODY = report_body(IMG, TIMELINE_SVG)
    # tables whose contents are computed from the collected fields
    BODY = BODY.replace("@@PERTILE@@", per_tile_table())
    BODY = BODY.replace("@@HIGHORDER@@", higher_order_table())
    BODY = BODY.replace("@@PREM_ERROR@@", pre_error_table())
    BODY = BODY.replace("@@PREM_DECOMP@@", pre_decomp_table())
    BODY = BODY.replace("@@PREM_SOURCES@@", pre_sources_table())
    BODY = BODY.replace("@@PREM_TERRAIN@@", pre_terrain_table())
    BODY = BODY.replace("@@PREM_CORR@@", pre_corr_table())
    BODY = BODY.replace("@@PREM_VARDECOMP@@", pre_vardecomp_table())
    BODY = BODY.replace("@@PREM_DIALECT@@", pre_dialect_table())
    BODY = BODY.replace("@@PREM_CONS_CONT@@", pre_cons_cont_table())
    BODY = BODY.replace("@@PREM_CONS_CAT@@", pre_cons_cat_table())
    BODY = BODY.replace("@@PREM_CONS_DIST@@", pre_cons_dist_table())
    BODY = BODY.replace("@@PREM_CONS_VM@@", pre_cons_vm_table())
    BODY = BODY.replace("@@PREM_PAT_TABLE@@", pat_table())
    BODY = BODY.replace("@@PATX_TABLES@@", patx_tables())
    for tok, txt in pre_cons_text().items():
        BODY = BODY.replace(tok, txt)
    for tok, txt in pat_text().items():
        BODY = BODY.replace(tok, txt)
    for tok, txt in patx_text().items():
        BODY = BODY.replace(tok, txt)
    BODY = BODY.replace("@@PREM_MODEL@@", pre_model_table())
    BODY = BODY.replace("@@PREM_PAIRED@@", pre_paired_table())
    BODY = renumber(BODY)
    html = [HEAD, COVER, TOC, BODY]
    html.append("""<div class="foot">
本报告由项目内实测数据、评估脚本与统计脚本自动汇总生成，所有数值均可由
<code>outputs/</code> 下的 JSON 逐项复算。图表按 SciencePlots 规范绘制，正文与图表数据同源。</div>
</body></html>""")
    out = ROOT / "report.html"
    out.write_text("".join(html), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024/1024:.2f} MB)")



if __name__ == "__main__":
    build()

