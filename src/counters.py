"""Pipeline counters and the accounting invariant."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class InvariantViolation(Exception):
    """Raised when ingested_total != kept + dropped_total + capped_total."""


@dataclass
class Counters:
    """Run-level counters. Holds numbers only, never content.

    Two separate ways a valid-or-not record leaves the pipeline:
      - dropped: a QUALITY rejection (empty, duplicate, unresolved domain...),
        recorded through ``drop(stage, reason)``;
      - capped: a deliberate SAMPLING decision on valid records (train-side
        domain cap), recorded through ``cap(domain)``.
    They never share a counter. Invariant:
        ingested_total == kept + dropped_total + capped_total
    """

    ingested_per_source: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, dict[str, int]] = field(default_factory=dict)  # stage -> reason -> count
    # Same drops, broken down by source: source -> reason -> count.
    dropped_per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    capped: dict[str, int] = field(default_factory=dict)  # domain -> examples removed by the cap
    duplicates_exact: int = 0
    duplicates_near: int = 0
    superseded: int = 0
    # Cleaning operations (quoted_history, signature, disclaimer, auto_acknowledgement) -> count.
    cleaned: dict[str, int] = field(default_factory=dict)
    pii_replaced: dict[str, int] = field(default_factory=dict)  # pii type -> count
    # Regex candidates that failed their validator (not replaced): pattern precision.
    pii_candidates_rejected: dict[str, int] = field(default_factory=dict)

    def ingest(self, source: str, n: int = 1) -> None:
        self.ingested_per_source[source] = self.ingested_per_source.get(source, 0) + n

    def drop(self, stage: str, reason: str, n: int = 1, source: str | None = None) -> None:
        by_reason = self.dropped.setdefault(stage, {})
        by_reason[reason] = by_reason.get(reason, 0) + n
        if source is not None:
            per_source = self.dropped_per_source.setdefault(source, {})
            per_source[reason] = per_source.get(reason, 0) + n

    def clean_op(self, operation: str, n: int = 1) -> None:
        self.cleaned[operation] = self.cleaned.get(operation, 0) + n

    def cap(self, domain: str, n: int = 1) -> None:
        self.capped[domain] = self.capped.get(domain, 0) + n

    def add_pii(self, pii_type: str, n: int = 1) -> None:
        self.pii_replaced[pii_type] = self.pii_replaced.get(pii_type, 0) + n

    def reject_pii_candidate(self, pii_type: str, n: int = 1) -> None:
        self.pii_candidates_rejected[pii_type] = self.pii_candidates_rejected.get(pii_type, 0) + n

    @property
    def ingested_total(self) -> int:
        return sum(self.ingested_per_source.values())

    @property
    def dropped_total(self) -> int:
        return sum(sum(reasons.values()) for reasons in self.dropped.values())

    @property
    def capped_total(self) -> int:
        return sum(self.capped.values())

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ingested_total"] = self.ingested_total
        data["dropped_total"] = self.dropped_total
        data["capped_total"] = self.capped_total
        return data

    def check_invariant(self, kept: int) -> None:
        """Raise when ingested_total != kept + dropped_total + capped_total."""
        ingested = self.ingested_total
        dropped = self.dropped_total
        capped = self.capped_total
        if ingested != kept + dropped + capped:
            logger.error(
                "invariant_violation ingested=%d kept=%d dropped=%d capped=%d",
                ingested, kept, dropped, capped,
            )
            raise InvariantViolation(
                f"ingested_total={ingested} != kept={kept} + dropped_total={dropped} "
                f"+ capped_total={capped} (difference={ingested - kept - dropped - capped})"
            )
