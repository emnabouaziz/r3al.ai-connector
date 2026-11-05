"""Anonymisation engine: structured PII, then addresses, then NER.

Order of operations (mandatory):
  1. structured PII (IBAN, NATIONAL_ID, EMAIL, PHONE): checksum/structure
     confirmed, so they must not be swallowed by a loose NER span;
  2. ADDRESS pattern;
  3. NER last, over the already partially masked text;
  4. never replace inside an existing placeholder.
The whole sequence is repeated until the text stops changing, which makes
``anonymise`` idempotent by construction: anonymise(anonymise(t)) == anonymise(t).
"""

from collections import Counter
from dataclasses import dataclass, field

from src.processing.anonymize.ner import NerModel, NerSpan
from src.processing.anonymize.patterns import (
    ALL_PATTERNS,
    PiiPattern,
    confirmed_length,
    overlaps,
    placeholder_spans,
)

MAX_PASSES = 5


@dataclass
class AnonymisationResult:
    text: str
    replaced: dict[str, int] = field(default_factory=dict)  # placeholder name -> count
    candidates_rejected: dict[str, int] = field(default_factory=dict)  # pattern -> count


def apply_pattern(
    text: str, pattern: PiiPattern, replaced: Counter[str], rejected: Counter[str]
) -> str:
    protected = placeholder_spans(text)
    out: list[str] = []
    last = 0
    for m in pattern.regex.finditer(text):
        if m.start() < last or overlaps(m.span(), protected):
            continue
        length = confirmed_length(m, pattern)
        if length is None:
            rejected[pattern.pii_type] += 1
            continue
        out.append(text[last:m.start()])
        out.append(f"[{pattern.pii_type}]")
        last = m.start() + length
        replaced[pattern.pii_type] += 1
    out.append(text[last:])
    return "".join(out)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and not text[start].isalnum():
        start += 1
    while end > start and not text[end - 1].isalnum():
        end -= 1
    return start, end


def _minus(start: int, end: int, protected: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """[start, end) minus the protected ranges."""
    segments: list[tuple[int, int]] = []
    cursor = start
    for p_start, p_end in sorted(protected):
        if p_end <= cursor or p_start >= end:
            continue
        if p_start > cursor:
            segments.append((cursor, p_start))
        cursor = max(cursor, p_end)
    if cursor < end:
        segments.append((cursor, end))
    return segments


def _apply_ner(text: str, spans: list[NerSpan], replaced: Counter[str]) -> str:
    protected = placeholder_spans(text)
    # Cut each span around existing placeholders, keep pieces that contain letters.
    pieces: list[NerSpan] = []
    for span in spans:
        for seg_start, seg_end in _minus(span.start, span.end, protected):
            seg_start, seg_end = _strip_span(text, seg_start, seg_end)
            if seg_end > seg_start and any(ch.isalpha() for ch in text[seg_start:seg_end]):
                pieces.append(NerSpan(seg_start, seg_end, span.pii_type))
    # Merge overlapping pieces (PERSON wins ties: recall on names first).
    merged: list[NerSpan] = []
    for piece in sorted(pieces, key=lambda s: (s.start, -s.end, s.pii_type != "PERSON")):
        if merged and piece.start < merged[-1].end:
            last = merged[-1]
            pii_type = "PERSON" if "PERSON" in (last.pii_type, piece.pii_type) else last.pii_type
            merged[-1] = NerSpan(last.start, max(last.end, piece.end), pii_type)
        else:
            merged.append(piece)
    for piece in reversed(merged):
        text = text[: piece.start] + f"[{piece.pii_type}]" + text[piece.end :]
        replaced[piece.pii_type] += 1
    return text


def anonymise(text: str, ner_model: NerModel) -> AnonymisationResult:
    replaced: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for pass_index in range(MAX_PASSES):
        before = text
        pass_rejected: Counter[str] = Counter()
        for pattern in ALL_PATTERNS:
            text = apply_pattern(text, pattern, replaced, pass_rejected)
        text = _apply_ner(text, ner_model.find(text), replaced)
        if pass_index == 0:
            rejected = pass_rejected  # rejections are counted on the first pass only
        if text == before:
            break
    return AnonymisationResult(
        text=text,
        replaced=dict(sorted(replaced.items())),
        candidates_rejected=dict(sorted(rejected.items())),
    )
