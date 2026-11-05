"""Shared contract of the example builders.

A builder turns pipeline records into question/answer CANDIDATES and
counts every candidate it rejects, by reason. Rejecting is the point: the
rejection table is a headline figure of the quality report.

Pure: no I/O, no network, deterministic.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from src.contracts import Domain, Language, Record

_PLACEHOLDER_OR_WORD = re.compile(r"\[[A-Z_]+\]|[^\W_]+")
_PLACEHOLDER = re.compile(r"^\[[A-Z_]+\]$")


@dataclass
class BuildCandidate:
    user: str
    assistant: str
    domain: Domain
    language: Language
    source_ids: list[str]  # real RawDocument ids the example comes from
    group_id: str  # thread / document group: all its examples stay on one side of the split


@dataclass(frozen=True)
class BuildSettings:
    min_question_chars: int = 30
    min_answer_chars: int = 40
    # Share of placeholder tokens in the ANSWER (what the model learns to produce). Measured on the
    # synthetic data: real answers stay <= 0.09 (a greeting); form examples ("[PERSON] reports a leak
    # at [ADDRESS], call [PHONE], pay [IBAN]") reach 0.13-0.17.
    max_placeholder_share: float = 0.12
    min_chunk_chars: int = 80  # minimum answer drawn from a chunk
    max_answer_chars: int = 900


class ExampleBuilder(ABC):
    REASONS: ClassVar[tuple[str, ...]] = ()  # every reason this builder may report (zeros shown)

    def __init__(self, settings: BuildSettings) -> None:
        self.settings = settings

    @abstractmethod
    def build(self, records: list[Record]) -> tuple[list[BuildCandidate], dict[str, int]]:
        """Returns candidates plus rejection reasons -> counts."""


def placeholder_share(text: str) -> float:
    """Share of tokens that are placeholders ([PERSON], [IBAN], ...)."""
    tokens = _PLACEHOLDER_OR_WORD.findall(text)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _PLACEHOLDER.match(t)) / len(tokens)


@dataclass
class RejectionTally:
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int = 1) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + n
