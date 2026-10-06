"""Screenshot the pre-model chapter of report.html for a visual check.

Used after every rebuild of report.html.  Grabs the chapter heading, each new
figure block and the new tables, so that font fallback, subplot legibility and
caption layout can be inspected without opening the file by hand.
"""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs" / "checks"
OUT.mkdir(parents=True, exist_ok=True)

ALTS = ["premodel task", "error vs grid size", "error by depth bin",
        "error by slope bin", "premodel error maps", "terrain distortion",
        "cross resolution decay", "per tile scatter", "variance decomposition",
        "premodel result"]


async def main() -> None:
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1180, "height": 1500},
                              device_scale_factor=1)
        await pg.goto((ROOT / "report.html").as_uri())
        await pg.wait_for_timeout(1500)

        info = await pg.evaluate(
            "() => ({h: document.body.scrollHeight,"
            " broken: [...document.images].filter(i => !i.complete || !i.naturalWidth).length,"
            " figs: document.querySelectorAll('figure').length,"
            " tabs: document.querySelectorAll('table').length})")
        print("doc:", info)

        # the chapter heading itself
        el = await pg.query_selector("xpath=//h2[contains(., '前置模型与低分辨率')]")
        if el:
            await el.scroll_into_view_if_needed()
            await pg.wait_for_timeout(300)
            await pg.screenshot(path=str(OUT / "P00_chapter_open.png"))

        for i, alt in enumerate(ALTS, start=1):
            handle = await pg.query_selector(f"img[alt='{alt}']")
            if not handle:
                print("missing alt", alt)
                continue
            fig = await handle.evaluate_handle("n => n.closest('figure')")
            box = await fig.as_element().bounding_box()
            if box and box["height"] > 2100:
                await fig.as_element().scroll_into_view_if_needed()
                await pg.wait_for_timeout(250)
                await pg.screenshot(path=str(OUT / f"P{i:02d}_{alt.replace(' ', '_')}.png"))
            else:
                await fig.as_element().scroll_into_view_if_needed()
                await pg.wait_for_timeout(250)
                await fig.as_element().screenshot(
                    path=str(OUT / f"P{i:02d}_{alt.replace(' ', '_')}.png"))
            print("wrote", alt)

        for j, cap in enumerate(["表 6", "表 7"], start=1):
            el = await pg.query_selector(f"xpath=//caption[contains(., '{cap}　')]/..")
            if el:
                await el.scroll_into_view_if_needed()
                await pg.wait_for_timeout(250)
                await el.screenshot(path=str(OUT / f"T0{j}_table{cap.split()[-1]}.png"))
                print("wrote", cap)

        # the bias sign-flip disclosure block added to section 2.6
        el = await pg.query_selector("xpath=//p[contains(., '表 6 有一处结论必须单独点出')]")
        if el:
            await el.scroll_into_view_if_needed()
            await pg.wait_for_timeout(300)
            await el.screenshot(path=str(OUT / "D01_bias_signflip.png"))
            print("wrote bias sign-flip paragraph")
        for name, key in (("D02_volume_worsened", "其中两档的体积平衡反而被改差"),
                          ("D03_attribution", "这个结果与后文那一米阈值例外互为表里"),
                          ("D04_stacking_risk", "这个方向还需要与后置环节放在一起提醒")):
            el = await pg.query_selector(f"xpath=//p[contains(., '{key}')]")
            if el:
                await el.scroll_into_view_if_needed()
                await pg.wait_for_timeout(300)
                await el.screenshot(path=str(OUT / f"{name}.png"))
                print("wrote", name)

        await b.close()


asyncio.run(main())
