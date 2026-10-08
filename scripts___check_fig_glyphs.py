#!/usr/bin/env python
"""CJK-in-figure auditor and matplotlib missing-glyph harness.

For every target figure script this checker

1. imports the module without running ``main`` (so the module-level rcParams are
   the ones the figure will actually be drawn with),
2. patches ``Figure.savefig`` so that immediately before each plate is written it
   records every rendered text string on that figure,
3. wraps the draw in ``warnings.catch_warnings(record=True)`` and counts the
   matplotlib ``missing from font`` warnings,
4. flags every captured string that still contains a CJK codepoint.

Usage
-----
    python -X utf8 scripts/_check_fig_glyphs.py            # audit every figure
    python -X utf8 scripts/_check_fig_glyphs.py report_exp_ab
    python -X utf8 scripts/_check_fig_glyphs.py --warnings # only the warning tally

The exit status is non-zero when any Chinese string is found in a rendered figure
(label text only; long caption-like strings are reported separately) or when a
``missing from font`` warning is emitted.
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import matplotlib.text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def _texts_on_figure(fig) -> list[str]:
    out: list[str] = []
    for t in fig.findobj(match=lambda o: isinstance(o, matplotlib.text.Text)):
        try:
            s = t.get_text()
        except Exception:
            continue
        if s:
            out.append(s)
    return out


class Capture:
    """Collect, per written PNG, the text strings and the glyph warnings."""

    def __init__(self) -> None:
        self.texts: dict[str, list[str]] = {}
        self.warn: dict[str, int] = {}
        self._rec: list = []
        self._orig = matplotlib.figure.Figure.savefig

    def __enter__(self) -> "Capture":
        checker = self

        def patched(fig, fname, *a, **k):  # noqa: ANN001
            name = Path(str(fname)).name
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                checker._orig(fig, fname, *a, **k)
            checker.texts[name] = _texts_on_figure(fig)
            checker.warn[name] = sum(
                1 for w in rec if "missing from font" in str(w.message))
        matplotlib.figure.Figure.savefig = patched
        return self

    def __exit__(self, *exc) -> None:
        matplotlib.figure.Figure.savefig = self._orig


MODULES = {
    "report_exp_ab": "report_exp_ab",
    "premodel_05_figs": "premodel_05_figs",
    "premodel_08_readability_figs": "premodel_08_readability_figs",
}


def run(which: list[str]) -> int:
    total_cjk = 0
    total_warn = 0
    for key in which:
        mod = importlib.import_module(MODULES[key])
        cap = Capture()
        with cap:
            if key == "report_exp_ab":
                mod.main()
            else:
                fns = getattr(mod, "fns", None)
                if fns is None:
                    # mirror the module's own __main__ dispatch
                    import inspect
                    fns = {n: f for n, f in inspect.getmembers(mod, inspect.isfunction)
                           if n.startswith("fig_")}
                for f in fns.values():
                    f()
        # report
        print(f"\n=== {key} ===")
        for png in sorted(cap.texts):
            strings = cap.texts[png]
            cjk = [s for s in strings if CJK.search(s)]
            nw = cap.warn.get(png, 0)
            total_warn += nw
            total_cjk += len(cjk)
            flag = "CJK!" if cjk else "ok  "
            print(f"  [{flag}] {png:44s} warnings={nw:3d} cjk_strings={len(cjk)}")
            for s in cjk:
                print(f"          · {s[:96]!r}")
    print(f"\nTOTAL: cjk strings in figures = {total_cjk}   "
          f"missing-from-font warnings = {total_warn}")
    return 1 if (total_cjk or total_warn) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", default=None)
    ap.add_argument("--warnings", action="store_true",
                    help="only print the warning tally (skip CJK flagging)")
    args = ap.parse_args()
    which = args.modules or list(MODULES)
    sys.exit(run(which))


if __name__ == "__main__":
    main()
