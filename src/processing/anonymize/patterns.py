"""Layer 1: wide regex candidates, each confirmed by a validator.

Patterns are deliberately WIDE (high recall). A candidate that fails its
validator is NOT replaced; it is counted in ``candidates_rejected`` so the
precision of every pattern is visible in the quality report.

Because the patterns are wide, a match can swallow a trailing group that is
not part of the value ("BE68 5390 0754 7034 2024"). For trimmable types the
engine retries shorter prefixes, cut at separators, before giving up.

Order is fixed: checksum-confirmed types first (IBAN, NATIONAL_ID), then
EMAIL and PHONE, then ADDRESS.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from src.processing.anonymize.validators import (
    is_valid_be_phone,
    is_valid_be_postcode,
    is_valid_email,
    is_valid_iban,
    is_valid_rrn,
)

PLACEHOLDER_TYPES = ("PERSON", "EMAIL", "PHONE", "IBAN", "NATIONAL_ID", "ADDRESS")
PLACEHOLDER_RE = re.compile(r"\[(?:" + "|".join(PLACEHOLDER_TYPES) + r")\]")
SEPARATORS = " \t./()-"

_SEP = r"[ \t./-]?"


@dataclass(frozen=True)
class PiiPattern:
    pii_type: str  # placeholder name: "IBAN" is replaced by "[IBAN]"
    regex: re.Pattern[str]
    validate: Callable[[str], bool]
    trimmable: bool = False  # retry shorter prefixes cut at separators
    validate_group: str | None = None  # validate this named group instead of the match


STRUCTURED_PATTERNS: tuple[PiiPattern, ...] = (
    PiiPattern(
        "IBAN",
        # Country code + 2 check digits + 11..30 alphanumerics, optional single spaces.
        re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){11,30}\b"),
        is_valid_iban,
        trimmable=True,
    ),
    PiiPattern(
        "NATIONAL_ID",
        # YYMMDDXXXCC with any (or no) separators: 85.07.30-033.28, 85073003328.
        re.compile(rf"(?<![\d.])\d{{2}}{_SEP}\d{{2}}{_SEP}\d{{2}}{_SEP}\d{{3}}{_SEP}\d{{2}}(?!\d)"),
        is_valid_rrn,
    ),
    PiiPattern(
        "EMAIL",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
        is_valid_email,
    ),
    PiiPattern(
        "PHONE",
        # +32 / 0032 / 0 prefix, then 7-10 digits with optional separators.
        # Never starts right after "/" or ".": "14/02/2025 (708,40" is a date then an
        # amount, not the landline "02 2025 708" (found by the final scan on generated text).
        re.compile(r"(?<![\w+/.])(?:\+32|0032|0)[ \t./()-]{0,3}\d(?:[ \t./()-]{0,2}\d){6,9}(?!\d)"),
        is_valid_be_phone,
        trimmable=True,
    ),
)

ADDRESS_PATTERN = PiiPattern(
    "ADDRESS",
    # "<street words> <number>[box], <4-digit postcode> <City>"
    re.compile(
        # Starts on a word boundary, never on a preposition ("naar Kerkstraat 12" keeps "naar").
        r"\b(?!(?:naar|op|te|in|bij|van|vers|chez|au|aux|à|to|at|on|from)[ \t])"
        r"(?:[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’.-]*[ \t]+){1,4}"
        r"\d{1,4}[A-Za-z]?(?:[ \t]*(?:bus|bte|box)[ \t]*\d+)?"
        r",[ \t]*(?P<postcode>\d{4})[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+(?:[ \t-][A-ZÀ-Ý][A-Za-zÀ-ÿ'’-]+)*"
    ),
    is_valid_be_postcode,
    validate_group="postcode",
)

ALL_PATTERNS: tuple[PiiPattern, ...] = (*STRUCTURED_PATTERNS, ADDRESS_PATTERN)


def placeholder_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in PLACEHOLDER_RE.finditer(text)]


def overlaps(span: tuple[int, int], protected: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < p_end and p_start < end for p_start, p_end in protected)


def confirmed_length(match: re.Match[str], pattern: PiiPattern) -> int | None:
    """Length of the confirmed value at the start of ``match``, or None if rejected."""
    if pattern.validate_group is not None:
        return len(match.group(0)) if pattern.validate(match.group(pattern.validate_group)) else None
    value = match.group(0)
    if pattern.validate(value):
        return len(value)
    if pattern.trimmable:
        for cut in range(len(value) - 1, 0, -1):
            if value[cut] in SEPARATORS and value[cut - 1].isalnum() and pattern.validate(value[:cut]):
                return cut
    return None
