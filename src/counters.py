"""Pipeline counters and the accounting invariant."""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class InvariantViolation(Exception):
    """Raised when ingested_total != kept + total drops."""


@dataclass
class Counters:
    """Run-level counters. Holds numbers only, never content.

    Every record removed from the pipeline (including dedup and
    supersession) must also be recorded through ``drop`` so that the
    invariant ``ingested_total == kept + total_dropped`` holds.
    """

    ingested_per_source: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, dict[str, int]] = field(default_factory=dict)  # stage -> reason -> count
    duplicates_exact: int = 0
    duplicates_near: int = 0
    superseded: int = 0
    pii_replaced: dict[str, int] = field(default_factory=dict)  # pii type -> count

    def ingest(self, source: str, n: int = 1) -> None:
        self.ingested_per_source[source] = self.ingested_per_source.get(source, 0) + n

    def drop(self, stage: str, reason: str, n: int = 1) -> None:
        by_reason = self.dropped.setdefault(stage, {})
        by_reason[reason] = by_reason.get(reason, 0) + n

    def add_pii(self, pii_type: str, n: int = 1) -> None:
        self.pii_replaced[pii_type] = self.pii_replaced.get(pii_type, 0) + n

    @property
    def ingested_total(self) -> int:
        return sum(self.ingested_per_source.values())

    @property
    def dropped_total(self) -> int:
        return sum(sum(reasons.values()) for reasons in self.dropped.values())

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ingested_total"] = self.ingested_total
        data["dropped_total"] = self.dropped_total
        return data

    def check_invariant(self, kept: int) -> None:
        """Raise when ingested_total != kept + total drops."""
        ingested = self.ingested_total
        dropped = self.dropped_total
        if ingested != kept + dropped:
            logger.error(
                "invariant_violation ingested=%d kept=%d dropped=%d", ingested, kept, dropped
            )
            raise InvariantViolation(
                f"ingested_total={ingested} != kept={kept} + dropped_total={dropped}"
            )
