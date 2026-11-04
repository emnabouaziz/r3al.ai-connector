import pytest

from src.contracts import Record
from src.dataset.leakage import (
    LeakageError,
    assert_no_group_overlap,
    assert_no_near_duplicate_across_split,
)


def rec(record_id: str, group_id: str, text: str = "") -> Record:
    return Record(
        id=record_id,
        text=text or f"placeholder text for {record_id}",
        group_id=group_id,
        source_id=f"src:{group_id}",
        source_type="email",
        language="nl",
        domain="claims",
    )


TICKET = (
    "Goedemiddag, ik heb vorige week een schadeaangifte ingediend voor mijn wagen na een "
    "aanrijding op de parking. Kunt u mij laten weten wanneer de expert langskomt?"
)
TICKET_VARIANT = TICKET.replace("Goedemiddag", "Goedemiddag,").replace("?", " ?") + " "

UNRELATED = (
    "Bonjour, je souhaite recevoir un devis pour une assurance habitation pour mon nouvel "
    "appartement à Liège. Merci de me recontacter par téléphone."
)


def test_group_overlap_raises_with_offending_ids() -> None:
    train = [rec("a1", "g1"), rec("a2", "g2"), rec("a3", "g3")]
    eval_ = [rec("b1", "g2"), rec("b2", "g3"), rec("b3", "g4")]
    with pytest.raises(LeakageError) as exc:
        assert_no_group_overlap(train, eval_)
    assert exc.value.group_ids == ["g2", "g3"]
    assert "g2" in str(exc.value)


def test_record_id_overlap_raises() -> None:
    with pytest.raises(LeakageError):
        assert_no_group_overlap([rec("x", "g1")], [rec("x", "g2")])


def test_disjoint_groups_pass() -> None:
    assert_no_group_overlap([rec("a", "g1")], [rec("b", "g2")])


def test_near_duplicates_in_different_groups_are_flagged() -> None:
    train = [rec("t1", "g1", TICKET), rec("t2", "g2", UNRELATED)]
    eval_ = [rec("e1", "g9", TICKET_VARIANT)]
    with pytest.raises(LeakageError) as exc:
        assert_no_near_duplicate_across_split(train, eval_)
    assert [(t, e) for t, e, _ in exc.value.pairs] == [("t1", "e1")]
    assert exc.value.pairs[0][2] >= 0.9
    # Message must not leak content.
    assert "schadeaangifte" not in str(exc.value)


def test_near_duplicates_can_be_returned_without_raising() -> None:
    pairs = assert_no_near_duplicate_across_split(
        [rec("t1", "g1", TICKET)], [rec("e1", "g2", TICKET_VARIANT)], raise_on_match=False
    )
    assert [(t, e) for t, e, _ in pairs] == [("t1", "e1")]


def test_accent_and_case_differences_are_normalised() -> None:
    a = "Je souhaite exercer mon droit d'accès aux données personnelles conservées."
    b = "JE SOUHAITE EXERCER MON DROIT D’ACCES AUX DONNEES PERSONNELLES CONSERVEES."
    pairs = assert_no_near_duplicate_across_split(
        [rec("t1", "g1", a)], [rec("e1", "g2", b)], raise_on_match=False
    )
    assert pairs and pairs[0][2] == 1.0


def test_distinct_texts_pass() -> None:
    pairs = assert_no_near_duplicate_across_split(
        [rec("t1", "g1", TICKET)], [rec("e1", "g2", UNRELATED)]
    )
    assert pairs == []


def test_empty_texts_are_ignored() -> None:
    train = [Record("t1", "", "g1", "s", "email")]
    eval_ = [Record("e1", "", "g2", "s", "email")]
    assert assert_no_near_duplicate_across_split(train, eval_) == []
