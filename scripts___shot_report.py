"""Screenshot key regions of report.html for visual verification."""
import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_figs" / "checks"
OUT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1180, "height": 1500},
                              device_scale_factor=1)
        await pg.goto((ROOT / "report.html").as_uri())
        await pg.wait_for_timeout(1200)

        # whole-document metrics
        info = await pg.evaluate(
            "() => ({h: document.body.scrollHeight, w: document.body.scrollWidth,"
            " imgs: [...document.images].filter(i => !i.complete || !i.naturalWidth).length,"
            " figs: document.querySelectorAll('figure').length,"
            " tabs: document.querySelectorAll('table').length})")
        print("doc:", info)

        shots = {"01_cover": 0, "02_abstract": 900, "03_method": 4200}
        for name, y in shots.items():
            await pg.evaluate(f"window.scrollTo(0,{y})")
            await pg.wait_for_timeout(350)
            await pg.screenshot(path=str(OUT / f"{name}.png"))
            print("wrote", name)

        # locate a figure block and a table block, screenshot each
        for sel, name in (("figure:nth-of-type(6)", "04_figure_caption"),
                          ("table:nth-of-type(5)", "05_table")):
            el = await pg.query_selector(sel)
            if el:
                await el.scroll_into_view_if_needed()
                await pg.wait_for_timeout(300)
                await el.screenshot(path=str(OUT / f"{name}.png"))
                print("wrote", name)
        await b.close()


asyncio.run(main())
