"""Render docs/project_guide.md into a beginner-friendly PDF (fpdf2).

Supports a small Markdown subset: #/##/### headings, "- " bullets, fenced code
blocks (```), horizontal rules (---), and plain paragraphs. Pure-python, no system
dependencies. Run: python scripts/md_to_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

SRC = Path("docs/project_guide.md")
OUT = Path("docs/project_guide.pdf")

PAGE_W, PAGE_H = 210, 297  # A4
MARGIN = 18
BODY_W = PAGE_W - 2 * MARGIN


def sanitize(text: str) -> str:
    # Core fonts are latin-1; replace anything else so the PDF never fails.
    return text.encode("latin-1", "replace").decode("latin-1")


class Guide(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140)
        self.cell(0, 8, "Fraud Intelligence Platform - Project Guide", align="R")
        self.set_text_color(0)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)


def new_block(pdf: Guide):
    if pdf.get_y() > PAGE_H - 30:
        pdf.add_page()
    else:
        pdf.ln(3)


def write_heading(pdf: Guide, text: str, level: int):
    new_block(pdf)
    size = {1: 17, 2: 13, 3: 11}[level]
    pdf.set_font("Helvetica", "B", size)
    pdf.set_text_color(20, 40, 80)
    pdf.multi_cell(BODY_W, 7, sanitize(text))
    pdf.set_text_color(0)
    pdf.ln(1)


def write_paragraph(pdf: Guide, text: str):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_x(MARGIN)
    pdf.multi_cell(BODY_W, 5.4, sanitize(text))


def write_bullet(pdf: Guide, text: str):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_x(MARGIN + 4)
    pdf.cell(4, 5.2, "-")
    pdf.multi_cell(BODY_W - 8, 5.2, sanitize(text))


def write_code(pdf: Guide, text: str):
    pdf.set_font("Courier", "", 9)
    pdf.set_fill_color(240, 240, 245)
    pdf.set_x(MARGIN)
    pdf.multi_cell(BODY_W, 4.8, sanitize(text), fill=True)
    pdf.set_font("Helvetica", "", 10.5)


def render(src: Path, out: Path):
    pdf = Guide(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    in_code = False
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            in_code = not in_code
            if in_code:
                new_block(pdf)
            else:
                pdf.ln(2)
            continue
        if in_code:
            write_code(pdf, line if line else " ")
            continue
        if not line.strip():
            pdf.ln(2)
            continue
        if re.match(r"^-{3,}$", line.strip()):
            new_block(pdf)
            y = pdf.get_y()
            pdf.line(MARGIN, y, MARGIN + BODY_W, y)
            pdf.ln(3)
            continue
        if line.startswith("### "):
            write_heading(pdf, line[4:], 3)
            continue
        if line.startswith("## "):
            write_heading(pdf, line[3:], 2)
            continue
        if line.startswith("# "):
            write_heading(pdf, line[2:], 1)
            continue
        if re.match(r"^\s*[-*]\s+", line):
            write_bullet(pdf, re.sub(r"^\s*[-*]\s+", "", line))
            continue
        write_paragraph(pdf, line)

    pdf.output(str(out))


if __name__ == "__main__":
    render(SRC, OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
