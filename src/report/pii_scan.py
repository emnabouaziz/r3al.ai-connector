"""Final blocking PII guard over the written output files.

Re-runs every validated pattern (IBAN, NATIONAL_ID, EMAIL, PHONE, ADDRESS)
over the final train.jsonl / eval.jsonl. Any confirmed hit fails the run.
A Finding carries the file, the line number and the PII TYPE only: never
the matched value, never surrounding text.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.processing.anonymize.engine import apply_pattern
from src.processing.anonymize.patterns import ALL_PATTERNS


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    pii_type: str

    def __str__(self) -> str:
        return f"{Path(self.file).name}:{self.line} {self.pii_type}"


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def scan_text(text: str) -> set[str]:
    """PII types with at least one confirmed (validated) hit outside placeholders.

    Patterns run in the engine's order, masking as they go, so a fragment of
    an already-confirmed value (e.g. digits of a national register number
    that also look like a landline) is not reported twice.
    """
    hits: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    for pattern in ALL_PATTERNS:
        text = apply_pattern(text, pattern, hits, rejected)
    return set(hits)


def scan_output_files(paths: list[str]) -> list[Finding]:
    """Re-run every validated pattern over the FINAL train.jsonl and
    eval.jsonl. Any confirmed hit fails the run with exit code 3.
    Findings report file, line number and PII TYPE ONLY - never the
    matched value, never surrounding text."""
    findings: list[Finding] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                try:
                    texts = _strings(json.loads(line))
                except json.JSONDecodeError:
                    texts = [line]
                types: set[str] = set()
                for text in texts:
                    types |= scan_text(text)
                findings += [Finding(str(path), lineno, t) for t in sorted(types)]
    return findings
