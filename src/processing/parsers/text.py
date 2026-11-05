"""Text helpers shared by the document parsers and the chunker.

Parsers emit lightweight structure inside plain text:
  - "# Title" / "## Heading" lines for DOCX heading styles;
  - "cell | cell | cell" lines for table rows.
Numbered sections ("3. Exclusions", "Article 12", "2.1 Scope") are
recognised as headings too, which covers PDFs.
"""

import re

HEADING_RE = re.compile(
    r"^(?:#{1,6}[ \t]+\S"
    r"|\d{1,2}(?:\.\d{1,2})*[.)]?[ \t]+[A-ZÀ-Ý]"
    r"|(?:Article|Artikel|Section|Sectie|Chapitre|Hoofdstuk|Chapter)[ \t]+\d+)"
)
TABLE_ROW_SEPARATOR = " | "
_BULLET_RE = re.compile(r"^[-•*][ \t]+")
_SENTENCE_END = tuple(".!?:;…»\"”)")


def is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line))


def is_table_row(line: str) -> bool:
    return TABLE_ROW_SEPARATOR in line


def _starts_block(line: str) -> bool:
    return is_heading(line) or is_table_row(line) or bool(_BULLET_RE.match(line))


def _can_continue(line: str) -> bool:
    return not (is_heading(line) or is_table_row(line) or line.endswith(_SENTENCE_END))


def rejoin_broken_lines(text: str) -> str:
    """Rejoin lines broken mid-sentence.

    A line that does not end in sentence punctuation is joined to the next
    one with a space, unless either line is a heading, a table row or a
    bullet. Blank lines (paragraph breaks) are kept, at most one in a row.
    """
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if out and out[-1] and _can_continue(out[-1]) and not _starts_block(line):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out).strip()
