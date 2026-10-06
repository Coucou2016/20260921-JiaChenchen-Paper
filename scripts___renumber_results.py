"""Renumber the 结果 chapter headings and splice in the new spatially-resolved subsections."""
from __future__ import annotations

from pathlib import Path

P = Path(__file__).resolve().parents[1] / "scripts/report_body.py"
s = P.read_text(encoding="utf-8")

# descending order so we never overwrite a heading we still need
pairs = [("4.7　偏差随训练的演化", "4.10　偏差随训练的演化"),
         ("4.6　测试集复核", "4.8　测试集复核"),
         ("4.5　统计检验与效应量", "4.7　统计检验与效应量"),
         ("4.4　微调的配对结果", "4.6　微调的配对结果"),
         ("4.3　从头训练的权利衡代价", "4.5　从头训练的权利衡代价"),
         ("4.2　深水分桶诊断", "4.4　深水分桶诊断"),
         ("4.1　基线水平与超分辨率增益", "4.3　基线水平与超分辨率增益")]
for a, b in pairs:
    assert f"<h3>{a}</h3>" in s, f"missing {a}"
    assert f"<h3>{b}</h3>" not in s, f"target {b} already present"
    s = s.replace(f"<h3>{a}</h3>", f"<h3>{b}</h3>")

P.write_text(s, encoding="utf-8")
print("renumbered 结果 headings")
for line in s.splitlines():
    if line.startswith("<h3>4.") or line.startswith("<h3>2."):
        print("  ", line)
