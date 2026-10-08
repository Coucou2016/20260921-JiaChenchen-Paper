"""Convert the condensed report HTML into a Markdown twin (figures as files).

Mirrors build_markdown.py: same block walker, same figure mapping, but it reads
report_brief.html and writes report_brief.md with its own title and outline.
"""
from __future__ import annotations

import html as ihtml
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "report_brief.html").read_text(encoding="utf-8")

# Body only, and drop the cover/TOC decorative markup we will rewrite in Markdown.
body = src.split("<body>", 1)[1].split("</body>", 1)[0]

# Images: replace <img ...> inside figure blocks by their file path.
FIGMAP = {
    "fig01_task.png": "figures/fig01_task.png",
    "fig04_paired_finetune.png": "figures/fig04_paired_finetune.png",
    "fig07_forest.png": "figures/fig07_forest.png",
    "fig09_sweep.png": "figures/fig09_sweep.png",
    "fig20_hyetograph.png": "figures/fig20_hyetograph.png",
    "fig21_tile_deep.png": "figures/fig21_tile_deep.png",
    "fig21b_tile_deep_scale.png": "figures/fig21b_tile_deep_scale.png",
    "fig23_domain_maps.png": "figures/fig23_domain_maps.png",
    "fig24_density.png": "figures/fig24_density.png",
    "fig27_bootstrap.png": "figures/fig27_bootstrap.png",
    "fig42_error_vs_grid.png": "figures/fig42_error_vs_grid.png",
    "fig45_terrain_distortion.png": "figures/fig45_terrain_distortion.png",
    "fig47_cross_resolution.png": "figures/fig47_cross_resolution.png",
    "fig51_dialects.png": "figures/fig51_dialects.png",
    "fig62_error_accounts.png": "figures/fig62_error_accounts.png",
    "fig63_error_sources.png": "figures/fig63_error_sources.png",
    "fig72_depth_consistency.png": "figures/fig72_depth_consistency.png",
    "fig73_speed_consistency.png": "figures/fig73_speed_consistency.png",
    "fig74_vmeasure.png": "figures/fig74_vmeasure.png",
    "fig76_pattern_agreement.png": "figures/fig76_pattern_agreement.png",
    "fig77_pattern_areal.png": "figures/fig77_pattern_areal.png",
    "fig78_pattern_boundary.png": "figures/fig78_pattern_boundary.png",
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
out.append("# 城市洪水深度超分辨率重建中的深水欠估问题<br>简要版\n")
out.append("*核心图表与分析结论，完整报告见同目录 report.html*\n")
out.append("---\n")
out.append("## 目录\n")
out.append("- 摘要")
out.append("- 一　任务、数据与方法")
out.append("- 二　粗网格模拟的误差、两种口径与来源")
out.append("- 三　跨分辨率规律")
out.append("- 四　空间模式相似性")
out.append("- 五　深水欠估的证据")
out.append("- 六　深水欠估的修正与配对检验")
out.append("- 七　主要结论与适用边界\n")

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
        im = re.search(r"alt=\"([^\"]*)\"", f)
        alt = im.group(1) if im else ""
        alt2f = {
            "task schematic": "fig01_task.png",
            "three-part error accounting": "fig62_error_accounts.png",
            "error vs grid size": "fig42_error_vs_grid.png",
            "two dialects of comparison": "fig51_dialects.png",
            "which conditions enlarge the error": "fig63_error_sources.png",
            "terrain distortion": "fig45_terrain_distortion.png",
            "cross resolution decay": "fig47_cross_resolution.png",
            "depth pairing and distribution": "fig72_depth_consistency.png",
            "speed pairing and distribution": "fig73_speed_consistency.png",
            "V-measure matrices": "fig74_vmeasure.png",
            "chance corrected pattern agreement": "fig76_pattern_agreement.png",
            "areal overlap of the regionalisations": "fig77_pattern_areal.png",
            "wet front boundary agreement": "fig78_pattern_boundary.png",
            "hyetograph": "fig20_hyetograph.png",
            "from-scratch sweep": "fig09_sweep.png",
            "domain maps": "fig23_domain_maps.png",
            "tile deep": "fig21_tile_deep.png",
            "tile deep shared scale": "fig21b_tile_deep_scale.png",
            "density agreement": "fig24_density.png",
            "paired fine-tune": "fig04_paired_finetune.png",
            "bootstrap tiles": "fig27_bootstrap.png",
            "forest plot": "fig07_forest.png",
            "residual chapter deep tile": "fig89_res_tiles_deep.png",
            "residual chapter tile spectrum": "fig90_res_tiles_spectrum.png",
            "residual chapter error structure": "fig91_res_error_structure.png",
            "residual chapter residual field": "fig92_res_residual_field.png",
        }
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

md = "\n".join(out)
md = re.sub(r"\n{4,}", "\n\n\n", md)
(ROOT / "report_brief.md").write_text(md, encoding="utf-8")
print(f"wrote report_brief.md ({len(md)} chars)")
