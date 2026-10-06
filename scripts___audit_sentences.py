"""Check the prose of a built report, excluding table cells, code, the cover
block, headings and inline SVG graphics, so only real sentences are measured,
and flag anything longer than 95 characters or carrying punctuation and
vocabulary the author forbids.

Usage
-----
    python scripts/_audit_sentences.py                 # audits report.html
    python scripts/_audit_sentences.py report_brief.html
    python scripts/_audit_sentences.py report_brief.md

The extraction collapses ALL whitespace BEFORE splitting on sentence punctuation.
This matters because the source in report_body.py wraps a single sentence across
two physical lines; splitting first (on newlines) would count it as two short
sentences and hide genuine over-95 offenders.
"""
import html as ihtml
import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

BANNED = ("链路", "赋能", "抓手", "闭环", "维度上", "值得注意的是")


def strip_html(src: str) -> str:
    body = src
    for pat in (r"<table.*?</table>", r"<code.*?</code>", r"<style.*?</style>",
                r"<script.*?</script>", r"<h[1-6][^>]*>.*?</h[1-6]>",
                r'<section class="cover">.*?</section>',
                r'<section class="toc">.*?</section>', r"<svg.*?</svg>"):
        body = re.sub(pat, " ", body, flags=re.S)
    body = re.sub(r"data:image/[^\"')]+", " ", body)
    body = re.sub(r"<[^>]+>", "", body)             # tags vanish WITHOUT whitespace
    return re.sub(r"\s+", "", ihtml.unescape(body))  # CRITICAL: rejoin wrapped lines


def strip_md(src: str) -> str:
    lines = src.splitlines()
    # drop the markdown cover block (title, lede, rule) before the first heading
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            lines = lines[i:]
            break
    keep = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("|") or s.startswith("#") or s.startswith("!["):
            continue                                    # table rows, headings, images
        if re.fullmatch(r"-{3,}", s) or s.startswith("```"):
            continue                                    # rules and code fences
        if re.fullmatch(r"-\s.*", s):
            continue                                    # outline / TOC list items
        keep.append(ln)
    body = "\n".join(keep)
    body = re.sub(r"`[^`]*`", " ", body)
    body = re.sub(r"\*\*(.*?)\*\*", r"\1", body, flags=re.S)
    body = re.sub(r"\*(.*?)\*", r"\1", body, flags=re.S)
    body = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", body)
    return re.sub(r"\s+", "", ihtml.unescape(body))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "report.html"
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / name
    src = path.read_text(encoding="utf-8")
    text = strip_md(src) if path.suffix == ".md" else strip_html(src)

    sents = [x for x in re.split(r"[。！？]", text) if len(x) > 6]
    # safety guard: a run much longer than any sentence that carries no sentence-ending
    # mark is a layout block (e.g. a cover), not prose
    sents = [x for x in sents if not (len(x) > 200 and not re.search(r"[。！？]", x))]
    prose = "。".join(sents)

    cjk = len(re.findall(r"[\u4e00-\u9fff]", prose))
    print(f"=== {path.name} ===")
    print(f"prose chars (whitespace stripped): {len(prose)}  CJK chars: {cjk}")
    long_s = [x for x in sents if len(x) > 95]
    print(f"sentences: {len(sents)}  over 95 chars: {len(long_s)}  "
          f"max: {max((len(x) for x in sents), default=0)}")
    for x in sorted(long_s, key=len, reverse=True):
        print(f"  [{len(x)}] {x}")

    print()
    checks = [("全角冒号  ：", r"："), ("全角分号  ；", r"；"), ("破折号  ——", r"——"),
              ("不是..而是", r"不是[^。；]{0,40}而是"),
              ("并非..而是", r"并非[^。；]{0,40}而是"), ("原生", r"原生")]
    checks += [(f"禁用词  {w}", re.escape(w)) for w in BANNED]
    for label, pat in checks:
        hits = list(re.finditer(pat, prose))
        print(f"{label}: {len(hits)}")
        for h in hits[:5]:
            print("    ...", prose[max(0, h.start() - 30):h.start() + 30], "...")

    print()
    for w in ("不是", "而是", "并非", "而非", "并没有"):
        print(f"raw {w}: {prose.count(w)}")


main()
