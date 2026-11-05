import pytest

from src.counters import Counters, InvariantViolation


def test_drop_accumulates_by_stage_and_reason() -> None:
    c = Counters()
    c.drop("clean", "empty_text")
    c.drop("clean", "empty_text", 2)
    c.drop("dedup", "duplicate_exact")
    assert c.dropped == {"clean": {"empty_text": 3}, "dedup": {"duplicate_exact": 1}}
    assert c.dropped_total == 4


def test_capped_record_increments_capped_never_dropped() -> None:
    c = Counters()
    c.cap("claims")
    c.cap("claims", 4)
    c.cap("billing", 2)
    assert c.capped == {"claims": 5, "billing": 2}
    assert c.capped_total == 7
    assert c.dropped == {} and c.dropped_total == 0


def _balanced() -> Counters:
    c = Counters()
    c.ingest("crm_postgres", 100)
    c.drop("tag", "domain_unresolved", 3)
    c.cap("claims", 7)
    return c  # 100 = kept 90 + dropped 3 + capped 7


def test_check_invariant_passes_with_all_three_terms() -> None:
    _balanced().check_invariant(kept=90)


@pytest.mark.parametrize(
    ("kept", "missing"),
    [(97, "capped"), (93, "dropped"), (10, "kept")],
)
def test_check_invariant_fails_when_a_term_is_missing(kept: int, missing: str) -> None:
    # kept=97 behaves as if the capped term were ignored, kept=93 as if dropped were.
    with pytest.raises(InvariantViolation):
        _balanced().check_invariant(kept=kept)


def test_check_invariant_fails_when_capped_is_counted_as_dropped() -> None:
    c = Counters()
    c.ingest("s", 10)
    c.drop("build", "train_domain_cap", 2)  # the old, wrong accounting
    c.cap("claims", 2)
    with pytest.raises(InvariantViolation):
        c.check_invariant(kept=8)


def test_invariant_error_reports_each_term() -> None:
    with pytest.raises(InvariantViolation) as exc:
        _balanced().check_invariant(kept=50)
    message = str(exc.value)
    for fragment in ("ingested_total=100", "kept=50", "dropped_total=3", "capped_total=7", "difference=40"):
        assert fragment in message


def test_as_dict_contains_totals() -> None:
    c = _balanced()
    c.add_pii("iban", 4)
    d = c.as_dict()
    assert d["ingested_total"] == 100
    assert d["dropped_total"] == 3
    assert d["capped_total"] == 7
    assert d["capped"] == {"claims": 7}
    assert d["pii_replaced"] == {"iban": 4}
