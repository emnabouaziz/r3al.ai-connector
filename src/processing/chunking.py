"""Split long texts into training-sized chunks.

DECISION: a 30-page policy document is not one training example. Texts
longer than ``max_chars`` are split at headings (numbered sections, DOCX
Heading styles rendered as "#" lines); a section still longer than the
budget is split by grouping paragraphs, then lines. Tiny sections (a lone
heading) are merged into the next one so no chunk is a bare title.

Short texts - every CRM row, every e-mail - come back as a single chunk,
unchanged. Chunks of one document keep the document's group_id (set by the
caller), so they always land on the same side of the train/eval split.
"""

from src.processing.parsers.text import is_heading

DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 200


def _strip_markers(text: str) -> str:
    lines = [line.lstrip("#").strip() if line.startswith("#") else line for line in text.splitlines()]
    return "\n".join(lines).strip()


def _sections(text: str) -> list[str]:
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        if is_heading(line) and any(s.strip() for s in sections[-1]):
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(s).strip() for s in sections if any(x.strip() for x in s)]


def _pack(units: list[str], max_chars: int, joiner: str) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{joiner}{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _split_long(section: str, max_chars: int) -> list[str]:
    if len(section) <= max_chars:
        return [section]
    paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
        else:
            pieces += _pack(paragraph.splitlines(), max_chars, "\n")
    return _pack(pieces, max_chars, "\n\n")


def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHARS, min_chars: int = DEFAULT_MIN_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [_strip_markers(text) if "#" in text else text]
    merged: list[str] = []
    carry = ""
    for section in _sections(text):
        section = f"{carry}\n\n{section}" if carry else section
        if len(section) < min_chars:
            carry = section
            continue
        carry = ""
        merged.append(section)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{carry}"
        else:
            merged.append(carry)
    chunks = [piece for section in merged for piece in _split_long(section, max_chars)]
    return [c for c in (_strip_markers(chunk) for chunk in chunks) if c]
