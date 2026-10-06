"""Render report_brief.html to report_brief.pdf with Playwright.

Mirrors build_pdf.py so the condensed report keeps the same page geometry,
CSS and self-contained base64 images as the full report.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto((ROOT / "report_brief.html").as_uri(), wait_until="load")
        await page.wait_for_timeout(2000)
        await page.pdf(
            path=str(ROOT / "report_brief.pdf"),
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            display_header_footer=True,
            header_template='<div style="font-size:8px;color:#888;width:100%;text-align:center;">'
                            '城市洪水深度超分辨率重建中的深水欠估问题　简要版</div>',
            footer_template='<div style="font-size:8px;color:#888;width:100%;'
                            'text-align:center;"><span class="pageNumber"></span> / '
                            '<span class="totalPages"></span></div>',
        )
        await browser.close()
    pdf = ROOT / "report_brief.pdf"
    print(f"wrote report_brief.pdf ({pdf.stat().st_size/1024/1024:.2f} MB)")


asyncio.run(main())
