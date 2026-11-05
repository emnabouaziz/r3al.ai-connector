"""Question generation for document / CRM chunks.

RULE: the answer must be CONTAINED IN THE CHUNK. Every generator returns
(question, answer) pairs where the answer is text drawn from the chunk
itself; the document builder re-checks that ('answer_not_grounded').

LLM EXTENSION POINT. ``QuestionGenerator`` is the seam: an
``LLMQuestionGenerator`` producing more varied questions could be added
later and selected in config, without touching the pipeline. None is
implemented here, on purpose: client content must never reach an external
LLM - the same rule the anonymiser enforces. A LOCAL model, once approved,
would plug in exactly here, and its answers would still have to pass the
grounding check.
"""

import hashlib
import re
from abc import ABC, abstractmethod

from src.contracts import Record
from src.processing.parsers.text import is_heading, is_table_row

_NUMBERING = re.compile(
    r"^(?:\d{1,2}(?:\.\d{1,2})*[.)]?|(?:Article|Artikel|Section|Sectie|Chapitre|Hoofdstuk|Chapter)\s+\d+[.:]?)"
    r"\s*[-–:]?\s*"
)
_PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TERMINAL_PUNCTUATION = tuple(".,:;!?")

TEMPLATES: dict[str, tuple[str, ...]] = {
    "nl": ('Wat zijn de voorwaarden voor "{h}"?', 'Wat moet ik weten over "{h}"?', 'Kunt u "{h}" toelichten?'),
    "fr": ("Quelles sont les conditions pour « {h} » ?", "Que dois-je savoir sur « {h} » ?",
           "Pouvez-vous m'expliquer « {h} » ?"),
    "en": ('What does "{h}" cover?', 'What should I know about "{h}"?', 'Can you explain "{h}"?'),
}
# With the document's title: a bare section heading ("Purpose", "Exclusions") is
# ambiguous on its own - the question must say which document it is about.
TEMPLATES_IN_DOCUMENT: dict[str, tuple[str, ...]] = {
    "nl": ('Wat staat er in "{doc}" over "{h}"?', 'Wat bepaalt "{doc}" over "{h}"?'),
    "fr": ("Que prévoit « {doc} » au sujet de « {h} » ?", "Que dit « {doc} » sur « {h} » ?"),
    "en": ('What does "{doc}" say about "{h}"?', 'In "{doc}", what applies to "{h}"?'),
}
DOCUMENT_TITLE_KEY = "document_title"


def document_title(text: str) -> str | None:
    """First line of a document when it reads as a title (set before chunking)."""
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first = first.lstrip("#").strip()
    return first if first and _is_title(first) and not _PLACEHOLDER.search(first) else None


class QuestionGenerator(ABC):
    @abstractmethod
    def generate(self, chunk: Record) -> list[tuple[str, str]]:
        """Returns (question, answer) pairs, answer drawn from the chunk.

        An empty list means the chunk offers no anchor for a question: the
        caller rejects it ('no_question_anchor') rather than inventing one.
        """


def _is_title(line: str) -> bool:
    words = line.split()
    return (
        1 <= len(words) <= 12
        and not line.endswith(_TERMINAL_PUNCTUATION)
        and not is_table_row(line)
        and any(ch.isalpha() for ch in line)
    )


def _section_start(lines: list[str], i: int, first: bool) -> tuple[str, str] | None:
    """(kind, anchor) when line i opens a section, else None."""
    line = lines[i].strip()
    after_blank = first or not lines[i - 1].strip()
    has_next = any(x.strip() for x in lines[i + 1:])
    if is_heading(line):
        return "heading", _NUMBERING.sub("", line).strip()
    if after_blank and has_next and line.endswith("?") and 3 <= len(line.split()) <= 15:
        return "question", line
    if after_blank and has_next and _is_title(line):
        return "title", line
    return None


def find_sections(text: str) -> list[tuple[str, str, str]]:
    """Sections of a chunk as (kind, anchor, body); kind: heading | question | title.

    A section opens on a numbered heading ("3. Exclusions"), on a short
    question standing on its own line (an FAQ entry), or on a short title
    line (no terminal punctuation) after a blank line. A line ending with ':'
    is a field label ("Interne notitie:"), not a topic, and never opens a
    section. The body runs until the next section: an answer never spills
    into the next heading. Anchors carrying a placeholder are refused.
    """
    lines = text.splitlines()
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    starts: list[tuple[int, str, str]] = []
    for position, i in enumerate(nonempty):
        start = _section_start(lines, i, first=position == 0)
        if start is not None:
            starts.append((i, *start))
    sections: list[tuple[str, str, str]] = []
    for n, (i, kind, anchor) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        body = "\n".join(lines[i + 1:end]).strip()
        if body and anchor and not _PLACEHOLDER.search(anchor):
            sections.append((kind, anchor, body))
    return sections


def leading_text(body: str, max_chars: int) -> str:
    """Prefix of ``body`` (whole lines, then whole sentences) within max_chars.

    Always a substring of the body: the answer stays grounded.
    """
    if len(body) <= max_chars:
        return body
    taken: list[str] = []
    length = 0
    for line in body.splitlines():
        if length + len(line) + 1 <= max_chars:
            taken.append(line)
            length += len(line) + 1
            continue
        sentences = _SENTENCE_END.split(line)
        partial = ""
        for sentence in sentences:
            candidate = f"{partial} {sentence}".strip() if partial else sentence
            if length + len(candidate) > max_chars or candidate not in line:
                break
            partial = candidate
        if partial:
            taken.append(partial)
        break
    return "\n".join(taken).strip()


class TemplatedQuestionGenerator(QuestionGenerator):
    """Fully local: question from the chunk's anchor, answer = its leading text."""

    def __init__(self, max_answer_chars: int = 900) -> None:
        self.max_answer_chars = max_answer_chars

    def generate(self, chunk: Record) -> list[tuple[str, str]]:
        """One (question, answer) per section; the answer is that section's own text."""
        templates = TEMPLATES.get(chunk.language)
        if templates is None:
            return []
        title = chunk.metadata.get(DOCUMENT_TITLE_KEY)
        pairs: list[tuple[str, str]] = []
        for n, (kind, anchor, body) in enumerate(find_sections(chunk.text)):
            # Deterministic template choice (sha256, never the builtin hash()).
            pick = int(hashlib.sha256(f"{chunk.id}/{n}".encode("utf-8")).hexdigest(), 16)
            if kind == "question":
                question = anchor
            elif isinstance(title, str) and title and title != anchor:
                choices = TEMPLATES_IN_DOCUMENT[chunk.language]
                question = choices[pick % len(choices)].format(doc=title, h=anchor)
            else:
                question = templates[pick % len(templates)].format(h=anchor)
            pairs.append((question, leading_text(body, self.max_answer_chars)))
        return pairs
