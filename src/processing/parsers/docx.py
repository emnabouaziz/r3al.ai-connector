"""DOCX text extraction (python-docx), in document order.

- Heading / Title styles become "#"-prefixed lines, so the chunker can
  split on them.
- Tables are serialised row by row as "cell | cell | cell": a table must
  never silently vanish (tariff grids are often the most useful content).
- Encoding is repaired with ftfy (mojibake such as "geÃ¯nformeerd").
- Lines broken mid-sentence are rejoined.
"""

from io import BytesIO

import ftfy
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.processing.parsers.text import TABLE_ROW_SEPARATOR, rejoin_broken_lines


def _heading_level(style_name: str) -> int | None:
    if style_name == "Title":
        return 1
    if style_name.startswith("Heading"):
        digits = "".join(ch for ch in style_name if ch.isdigit())
        return int(digits) if digits else 1
    return None


def extract_text(content: bytes) -> str:
    document = Document(BytesIO(content))
    lines: list[str] = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            style = paragraph.style.name if paragraph.style is not None else ""
            level = _heading_level(style or "")
            if text and level is not None:
                lines += ["", f"{'#' * min(level, 6)} {text}"]
            else:
                lines.append(text)
        elif tag == "tbl":
            lines.append("")
            for row in Table(child, document).rows:
                lines.append(TABLE_ROW_SEPARATOR.join(cell.text.strip() for cell in row.cells))
            lines.append("")
    return rejoin_broken_lines(ftfy.fix_text("\n".join(lines)))
