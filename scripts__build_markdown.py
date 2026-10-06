"""Convert the report HTML into a Markdown twin (figures referenced as files)."""
from __future__ import annotations

import html as ihtml
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "report.html").read_text(encoding="utf-8")

# Body only, and drop the cover/TOC decorative markup we will rewrite in Markdown.
body = src.split("<body>", 1)[1].split("</body>", 1)[0]

# Images: replace <img ...> inside figure blocks by their file path.
FIGMAP = {
    "fig01_task.png": "figures/fig01_task.png", "fig02_baselines.png": "figures/fig02_baselines.png",
    "fig03_depth_bins.png": "figures/fig03_depth_bins.png",
    "fig04_paired_finetune.png": "figures/fig04_paired_finetune.png",
    "fig05_selection_noise.png": "figures/fig05_selection_noise.png",
    "fig06_before_after.png": "figures/fig06_before_after.png",
    "fig07_forest.png": "figures/fig07_forest.png",
    "fig08_bias_evolution.png": "figures/fig08_bias_evolution.png",
    "fig09_sweep.png": "figures/fig09_sweep.png",
    "fig10_scorecard.png": "figures/fig10_scorecard.png",
    "fig11_training_history.png": "figures/fig11_training_history.png",
    "fig12_loss_parts.png": "figures/fig12_loss_parts.png",
    "fig20_hyetograph.png": "figures/fig20_hyetograph.png",
    "fig21_tile_deep.png": "figures/fig21_tile_deep.png",
    "fig22_tiles_spectrum.png": "figures/fig22_tiles_spectrum.png",
    "fig23_domain_maps.png": "figures/fig23_domain_maps.png",
    "fig24_density.png": "figures/fig24_density.png",
    "fig25_distributions.png": "figures/fig25_distributions.png",
    "fig26_exceedance.png": "figures/fig26_exceedance.png",
    "fig27_bootstrap.png": "figures/fig27_bootstrap.png",
    "fig28_autocorr.png": "figures/fig28_autocorr.png",
    "fig41_premodel_task.png": "figures/fig41_premodel_task.png",
    "fig42_error_vs_grid.png": "figures/fig42_error_vs_grid.png",
    "fig43_error_depth_bins.png": "figures/fig43_error_depth_bins.png",
    "fig44_error_slope_bins.png": "figures/fig44_error_slope_bins.png",
    "fig45_terrain_distortion.png": "figures/fig45_terrain_distortion.png",
    "fig46_error_maps.png": "figures/fig46_error_maps.png",
    "fig47_cross_resolution.png": "figures/fig47_cross_resolution.png",
    "fig48_tile_scatter.png": "figures/fig48_tile_scatter.png",
    "fig49_decomposition.png": "figures/fig49_decomposition.png",
    "fig50_premodel_result.png": "figures/fig50_premodel_result.png",
    "fig51_dialects.png": "figures/fig51_dialects.png",
    "fig61_two_fields.png": "figures/fig61_two_fields.png",
    "fig62_error_accounts.png": "figures/fig62_error_accounts.png",
    "fig63_error_sources.png": "figures/fig63_error_sources.png",
    "fig64_domain_zoom.png": "figures/fig64_domain_zoom.png",
    "fig71_pattern_by_grid.png": "figures/fig71_pattern_by_grid.png",
    "fig72_depth_consistency.png": "figures/fig72_depth_consistency.png",
    "fig73_speed_consistency.png": "figures/fig73_speed_consistency.png",
    "fig74_vmeasure.png": "figures/fig74_vmeasure.png",
    "fig75_pattern_map.png": "figures/fig75_pattern_map.png",
    "fig76_pattern_agreement.png": "figures/fig76_pattern_agreement.png",
    "fig77_pattern_areal.png": "figures/fig77_pattern_areal.png",
    "fig78_pattern_boundary.png": "figures/fig78_pattern_boundary.png",
    "fig79_pattern_autocorr.png": "figures/fig79_pattern_autocorr.png",
    "fig80_pattern_distance.png": "figures/fig80_pattern_distance.png",
    "fig81_sabre_crosscheck.png": "figures/fig81_sabre_crosscheck.png",
    "fig82_variogram_scale.png": "figures/fig82_variogram_scale.png",
    "fig83_contiguity_map.png": "figures/fig83_contiguity_map.png",
    "fig84_contiguity_metrics.png": "figures/fig84_contiguity_metrics.png",
    "fig85_k_sensitivity.png": "figures/fig85_k_sensitivity.png",
}


def clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", " ", s)
    s = re.sub(r"<b>(.*?)</b>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s, flags=re.S)
    s = re.sub(r"<i>(.*?)</i>", r"*\1*", s, flags=re.S)
    s = re.sub(r"<[^>]+>", "", s)
    s = ihtml.unescape(s)
    return re.sub(r"\s+", " ", s).strip()


out: list[str] = []
out.append("# 城市洪水深度超分辨率重建中的深水欠估问题<br>诊断、修正与方法学检验\n")
out.append("*以新西兰惠灵顿城区洪水数据集为例，10 米到 2 米最大淹没水深重建*\n")
out.append("---\n")
out.append("## 目录\n")
out.append("- 摘要")
out.append("- 一　研究背景与目的")
out.append("  - 1.1　为什么需要把粗网格水深还原到细网格")
out.append("  - 1.2　引导式超分辨率的含义")
out.append("  - 1.3　一个长期被整体指标掩盖的问题")
out.append("  - 1.4　研究目的")
out.append("  - 1.5　本报告的五点贡献")
out.append("- 二　前置模型与低分辨率模拟的误差量化")
out.append("  - 2.1　粗网格模拟与聚合真值")
out.append("  - 2.2　误差的口径与评价指标")
out.append("  - 2.3　低分辨率模拟误差随网格尺寸的变化")
out.append("  - 2.4　细网格口径与粗网格口径")
out.append("  - 2.5　哪些条件让误差变大")
out.append("  - 2.6　地形失真的来源与量级")
out.append("  - 2.7　跨分辨率相关性的衰减规律")
out.append("  - 2.8　地形与下垫面的跨分辨率相关性")
out.append("  - 2.9　水动力变量的跨分辨率相关性")
out.append("  - 2.10　基于 V-measure 的空间模式相似性")
out.append("  - 2.11　空间模式相似性的多族度量")
out.append("  - 2.12　跨分辨率模式的尺度、连通性与簇数检验")
out.append("  - 2.13　跨分辨率一致性对输入特征的启示")
out.append("  - 2.14　逐瓦片的离散与解释份额")
out.append("  - 2.15　前置模型的结构、训练与测试结果")
out.append("  - 2.16　本章术语速查")
out.append("- 三　数据、模型与方法")
out.append("  - 3.1　研究区与数据来源")
out.append("  - 3.2　数据组织与切分原则")
out.append("  - 3.3　降雨驱动与洪水场的基本形态")
out.append("  - 3.4　模型结构")
out.append("  - 3.5　损失函数与评价指标")
out.append("  - 3.6　计算条件")
out.append("- 四　研究过程")
out.append("  - 4.1　阶段一　基线训练及其饱和")
out.append("  - 4.2　阶段二　深水误差的根因诊断")
out.append("  - 4.3　阶段三　从头训练的深水损失对比")
out.append("  - 4.4　阶段四　冻结模型的温和微调")
out.append("  - 4.5　阶段五　两处测量陷阱与修正")
out.append("  - 4.6　阶段六　延长训练与当前状态")
out.append("- 五　结果")
out.append("  - 5.1　逐瓦片的空间重构对比")
out.append("  - 5.2　点对点一致性与偏差结构")
out.append("  - 5.3　基线水平与超分辨率增益")
out.append("  - 5.4　深水分桶诊断")
out.append("  - 5.5　从头训练的权利衡代价")
out.append("  - 5.6　微调的配对结果")
out.append("  - 5.7　统计检验与效应量")
out.append("  - 5.8　测试集复核")
out.append("  - 5.9　高阶统计分析与阈值敏感性")
out.append("  - 5.10　偏差随训练的演化")
out.append("- 六　分析与讨论")
out.append("  - 6.1　权重的绝对数字没有意义")
out.append("  - 6.2　为什么微调比从头训练更适合这个任务")
out.append("  - 6.3　测量方式与模型设计同等重要")
out.append("  - 6.4　与从头训练方案的总体对比")
out.append("- 七　主要结论")
out.append("- 八　不足与展望")
out.append("  - 8.1　结论适用范围上的限制")
out.append("  - 8.2　方法层面可以继续优化的地方")
out.append("  - 8.3　主题层面可以拓展的方向")
out.append("- 附录一　术语表")
out.append("- 附录二　复现命令\n")

# Walk block elements in order.
token = re.compile(
    r"<h2>(?P<h2>.*?)</h2>|<h3>(?P<h3>.*?)</h3>|<h4>(?P<h4>.*?)</h4>|"
    r"<figure>(?P<fig>.*?)</figure>|<table>(?P<tab>.*?)</table>|<p>(?P<p>.*?)</p>|"
    r"<section class=\"abstract\">(?P<abs>.*?)</section>",
    re.S)

for m in token.finditer(body):
    if m.group("h2"):
        out.append(f"\n## {clean(m.group('h2'))}\n")
    elif m.group("h3"):
        out.append(f"\n### {clean(m.group('h3'))}\n")
    elif m.group("h4"):
        out.append(f"\n#### {clean(m.group('h4'))}\n")
    elif m.group("abs"):
        inner = m.group("abs")
        for pm in re.finditer(r"<h2>(.*?)</h2>|<p[^>]*>(.*?)</p>", inner, re.S):
            if pm.group(1):
                out.append(f"\n## {clean(pm.group(1))}\n")
            else:
                out.append(clean(pm.group(2)) + "\n")
    elif m.group("p"):
        text = m.group("p")
        if "<svg" in text:
            continue
        if not clean(text) or "capsp" in text:
            continue
        out.append(clean(text) + "\n")
    elif m.group("fig"):
        f = m.group("fig")
        cap = re.search(r"<figcaption>(.*?)</figcaption>", f, re.S)
        num = "?"
        if cap:
            nm = re.search(r"图\s*(\d+)", cap.group(1))
            num = nm.group(1) if nm else "?"
        if "<svg" in f:
            out.append(f"\n![图 {num}](figures/fig00_timeline.png)\n")
        im = re.search(r"alt=\"([^\"]*)\"", f)
        alt = im.group(1) if im else ""
        alt2f = {"task schematic": "fig01_task.png", "baselines": "fig02_baselines.png",
                 "depth-binned diagnostics": "fig03_depth_bins.png",
                 "paired fine-tune": "fig04_paired_finetune.png",
                 "selection noise": "fig05_selection_noise.png",
                 "before after": "fig06_before_after.png", "forest plot": "fig07_forest.png",
                 "bias evolution": "fig08_bias_evolution.png", "from-scratch sweep": "fig09_sweep.png",
                 "scorecard": "fig10_scorecard.png",
                 "baseline training history": "fig11_training_history.png",
                 "loss decomposition": "fig12_loss_parts.png",
                 "hyetograph": "fig20_hyetograph.png",
                 "tile deep": "fig21_tile_deep.png",
                 "tile spectrum": "fig22_tiles_spectrum.png",
                 "domain maps": "fig23_domain_maps.png",
                 "density agreement": "fig24_density.png",
                 "error distributions": "fig25_distributions.png",
                 "exceedance": "fig26_exceedance.png",
                 "bootstrap tiles": "fig27_bootstrap.png",
                 "error autocorrelation": "fig28_autocorr.png",
                 "premodel task": "fig41_premodel_task.png",
                 "error vs grid size": "fig42_error_vs_grid.png",
                 "error by depth bin": "fig43_error_depth_bins.png",
                 "error by slope bin": "fig44_error_slope_bins.png",
                 "terrain distortion": "fig45_terrain_distortion.png",
                 "premodel error maps": "fig46_error_maps.png",
                 "cross resolution decay": "fig47_cross_resolution.png",
                 "per tile scatter": "fig48_tile_scatter.png",
                 "variance decomposition": "fig49_decomposition.png",
                 "premodel result": "fig50_premodel_result.png",
                 "two dialects of comparison": "fig51_dialects.png",
                 "two fields on one lattice": "fig61_two_fields.png",
                 "three-part error accounting": "fig62_error_accounts.png",
                 "which conditions enlarge the error": "fig63_error_sources.png",
                 "zoomed sub-regions across the four coarse grids":
                     "fig64_domain_zoom.png",
                 "terrain and flood pattern at four grid sizes":
                     "fig71_pattern_by_grid.png",
                 "depth pairing and distribution": "fig72_depth_consistency.png",
                 "speed pairing and distribution": "fig73_speed_consistency.png",
                 "V-measure matrices": "fig74_vmeasure.png",
                 "regionalisation and stability map": "fig75_pattern_map.png",
                 "SABRE cross-check": "fig81_sabre_crosscheck.png",
                 "semivariogram and scale space": "fig82_variogram_scale.png",
                 "contiguity constrained regionalisation": "fig83_contiguity_map.png",
                 "contiguity metrics": "fig84_contiguity_metrics.png",
                 "cluster count sensitivity": "fig85_k_sensitivity.png"}
        fname = alt2f.get(alt)
        if fname:
            out.append(f"\n![图 {num}](figures/{fname})\n")
        # caption: first sentence as bold title, rest as prose
        if cap:
            c = cap.group(1)
            bm = re.search(r"<b>(.*?)</b>", c, re.S)
            title = clean(bm.group(1)) if bm else f"图 {num}"
            rest = clean(c.split("</b>", 1)[1]) if bm else clean(c)
            out.append(f"**{title}**\n")
            if rest:
                out.append(rest + "\n")
        # explanation paragraphs that live inside the figure block
        for pm in re.finditer(r"<p[^>]*>(.*?)</p>", f, re.S):
            if "capsp" in pm.group(0) or "<svg" in pm.group(0):
                continue
            txt = clean(pm.group(1))
            if txt:
                out.append(txt + "\n")
    elif m.group("tab"):
        t = m.group("tab")
        cap = re.search(r"<caption>(.*?)</caption>", t, re.S)
        if cap:
            c = cap.group(1)
            bm = re.search(r"<b>(.*?)</b>", c, re.S)
            if bm:
                title = clean(bm.group(1))
                rest = clean(c.split("</b>", 1)[1])
                out.append(f"\n**{title}**\n")
                if rest:
                    out.append(rest + "\n")
            else:
                out.append(f"\n**{clean(c)}**\n")
        rows = re.findall(r"<tr>(.*?)</tr>", t, re.S)
        parsed = []
        for r in rows:
            cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.S)
            parsed.append([clean(c) for c in cells])
        if parsed:
            out.append("| " + " | ".join(parsed[0]) + " |")
            out.append("|" + "|".join(["---"] * len(parsed[0])) + "|")
            for r in parsed[1:]:
                while len(r) < len(parsed[0]):
                    r.append("")
                out.append("| " + " | ".join(r) + " |")
            out.append("")
        if cap:
            # explanation paragraphs that followed the table
            pass

md = "\n".join(out)
md = re.sub(r"\n{4,}", "\n\n\n", md)
(ROOT / "report.md").write_text(md, encoding="utf-8")
print(f"wrote report.md ({len(md)} chars)")
