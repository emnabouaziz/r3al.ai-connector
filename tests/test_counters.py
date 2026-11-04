import pytest

from src.counters import Counters, InvariantViolation


def test_drop_accumulates_by_stage_and_reason() -> None:
    c = Counters()
    c.drop("clean", "empty_text")
    c.drop("clean", "empty_text", 2)
    c.drop("dedup", "duplicate_exact")
    assert c.dropped == {"clean": {"empty_text": 3}, "dedup": {"duplicate_exact": 1}}
    assert c.dropped_total == 4


def test_check_invariant_passes_when_balanced() -> None:
    c = Counters()
    c.ingest("crm_postgres", 10)
    c.ingest("support_mailbox", 5)
    c.drop("parse", "unreadable", 3)
    c.check_invariant(kept=12)


def test_check_invariant_raises_when_unbalanced() -> None:
    c = Counters()
    c.ingest("crm_postgres", 10)
    c.drop("parse", "unreadable", 3)
    with pytest.raises(InvariantViolation):
        c.check_invariant(kept=8)


def test_as_dict_contains_totals() -> None:
    c = Counters()
    c.ingest("knowledge_docs", 2)
    c.add_pii("iban", 4)
    d = c.as_dict()
    assert d["ingested_total"] == 2
    assert d["dropped_total"] == 0
    assert d["pii_replaced"] == {"iban": 4}
