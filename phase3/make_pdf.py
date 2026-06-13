"""Render phase3/REPORT.md to team01.pdf via markdown -> HTML -> fpdf2.
Run: PYTHONPATH=/tmp/mdlib:/tmp/fpdflib <python> phase3/make_pdf.py
"""
import re
from pathlib import Path
import markdown
from fpdf import FPDF

HERE = Path(__file__).resolve().parent
md_text = (HERE / "REPORT.md").read_text()

# fpdf2 core fonts are latin-1 only; map common unicode to ASCII.
_SUB = {
    "—": "-", "–": "-", "→": "->", "±": "+/-", "≤": "<=", "≥": ">=",
    "≈": "~", "×": "x", "✅": "[OK]", "⚠️": "[!]", "•": "-",
    "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...",
    "¹": "(1)", "²": "(2)", "³": "(3)", "⁴": "(4)",
    "↔": "to", "≠": "!=", "→": "->", "←": "<-",
}
for k, v in _SUB.items():
    md_text = md_text.replace(k, v)

# Title from first H1; drop it from body (we render it as the doc title).
lines = md_text.splitlines()
title = lines[0].lstrip("# ").strip()
body_md = "\n".join(lines[1:])

html = markdown.markdown(body_md, extensions=["tables"])
# fpdf2 cannot nest inline tags inside <td>; unwrap inline formatting to text.
html = re.sub(r"</?(strong|em|code|b|i)>", "", html)

pdf = FPDF(format="A4")
pdf.set_auto_page_break(auto=True, margin=12)
pdf.set_margins(14, 12, 14)
pdf.add_page()
pdf.set_font("helvetica", "B", 14)
pdf.multi_cell(0, 7, title)
pdf.ln(1)
pdf.set_font("helvetica", size=9)
# Compact line height for body + tables
pdf.write_html(html, tag_styles=None)
out = HERE / "team01.pdf"
pdf.output(str(out))
print("wrote", out, "pages:", pdf.pages_count)
