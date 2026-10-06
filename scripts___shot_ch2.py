"""Screenshot the rewritten chapter-2 opening, table 1 and table 5.

Run after every rebuild of report.html.  Grabs (a) the 2.1 section opening
where the two parallel fields are introduced, (b) the rebuilt table 5 whose
header now carries the short axis names, and (c) the rebuilt table 1 whose
caption now spells out both fields in full.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs" / "checks"
OUT.mkdir(parents=True, exist_ok=True)

PARAS = [
    ("P01_two_fields_intro", "本章从头到尾只比较两份资料"),
    ("P02_coarse_simulation", "第一份是粗网格模拟值"),
    ("P03_aggregation", "第二份是聚合真值"),
    ("P04_short_names", "后文为了行文简洁"),
    ("P05_dialect_intro", "本章称之为细网格口径"),
    ("P06_dialect_derivation", "这条等式可以从一块方块直接算出来"),
    ("P07_dialect_sharing", "本章把粗网格口径当作主口径"),
    ("P08_dialect_roundtrip", "把粗网格值搬到细网格这一步用最近邻升采样"),
    ("P09_dialect_five_m", "五米档的例外需要单独交代"),
    ("P10_supervision_target", "需要先说明这个目标的口径"),
]

TABLES = [
    ("T06_cross_resolution", "表 6　跨分辨率相关性"),
    ("T01_error_summary", "表 1　粗网格模拟值与聚合后的二米真值"),
    ("T03_dialect", "表 3　细网格口径与粗网格口径"),
]


async def main() -> None:
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1180, "height": 1500},
                              device_scale_factor=1)
        await pg.goto((ROOT / "report.html").as_uri())
        await pg.wait_for_timeout(1800)

        info = await pg.evaluate(
            "() => ({h: document.body.scrollHeight,"
            " broken: [...document.images].filter(i => !i.complete || !i.naturalWidth).length,"
            " figs: document.querySelectorAll('figure').length,"
            " tabs: document.querySelectorAll('table').length,"
            " words: document.body.innerText.includes('原生')})")
        print("doc:", info)

        # section 2.1 heading plus the paragraphs that follow it
        el = await pg.query_selector("xpath=//h3[contains(., '2.1　粗网格模拟与聚合真值')]")
        if el:
            await el.scroll_into_view_if_needed()
            await pg.wait_for_timeout(300)
            await pg.screenshot(path=str(OUT / "P00_section_2_1.png"))
            print("wrote section 2.1 viewport")

        for name, key in PARAS:
            node = await pg.query_selector(f"xpath=//p[contains(., '{key}')]")
            if not node:
                print("missing paragraph", key)
                continue
            await node.scroll_into_view_if_needed()
            await pg.wait_for_timeout(250)
            await node.screenshot(path=str(OUT / f"{name}.png"))
            print("wrote", name)

        for name, cap in TABLES:
            node = await pg.query_selector(f"xpath=//caption[contains(., '{cap}')]/..")
            if not node:
                print("missing table", cap)
                continue
            await node.scroll_into_view_if_needed()
            await pg.wait_for_timeout(300)
            await node.screenshot(path=str(OUT / f"{name}.png"))
            print("wrote", name)

        # the regenerated schematic and map figures that carried the old wording
        for i, alt in enumerate(["premodel task", "premodel error maps",
                                 "zoomed sub-regions across the four coarse grids",
                                 "two dialects of comparison"], start=1):
            handle = await pg.query_selector(f"img[alt='{alt}']")
            if not handle:
                print("missing alt", alt)
                continue
            fig = await handle.evaluate_handle("n => n.closest('figure')")
            el = fig.as_element()
            await el.scroll_into_view_if_needed()
            await pg.wait_for_timeout(250)
            await el.screenshot(path=str(OUT / f"F{i:02d}_{alt.replace(' ', '_')}.png"))
            print("wrote", alt)

        await b.close()


asyncio.run(main())
