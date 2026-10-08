"""Audit: list any remaining CJK inside string/comment tokens of the figure scripts."""
import io
import re
import sys
import tokenize

sys.stdout.reconfigure(encoding="utf-8")

CJK = re.compile(r"[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]")

tot = 0
for p in sys.argv[1:]:
    src = open(p, encoding="utf-8").read()
    n_str = 0
    n_com = 0
    lines = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        nm = tokenize.tok_name[tok.type]
        if not CJK.search(tok.string):
            continue
        if nm in ("STRING", "FSTRING_MIDDLE"):
            n_str += 1
            lines.append(("STR", tok.start[0], tok.string))
        elif nm == "COMMENT":
            n_com += 1
    print(f"{p}: CJK-string-tokens={n_str}  CJK-comments={n_com}")
    for kind, ln, s in lines[:40]:
        print(f"    {kind} line {ln}: {s!r}")
    tot += n_str
print("TOTAL CJK string tokens left:", tot)
