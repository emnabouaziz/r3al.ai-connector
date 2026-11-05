"""Near-duplicate clustering: correctness, determinism, scale, and split safety."""

import random
import time

from src.contracts import Domain, Language, Record
from src.dataset.splitter import SplitConfig, split
from src.processing import dedup_near
from src.processing.dedup_near import find_near_duplicates, merge_groups

TEMPLATE = (
    "Beste [PERSON], wij hebben uw aanvraag tot adreswijziging goed ontvangen. U ontvangt binnen vijf "
    "werkdagen een bijvoegsel bij uw polis met het nieuwe adres [ADDRESS]. Met vragen kunt u ons bellen."
)


def rec(record_id: str, text: str, group: str | None = None, domain: Domain = "onboarding",
        language: Language = "nl") -> Record:
    return Record(record_id, text, group or f"g-{record_id}", record_id.split("#")[0], "email",
                  language=language, domain=domain)


def test_texts_identical_once_masked_form_one_cluster() -> None:
    # Before anonymisation these differed by name and address; after, they are the same.
    records = [rec("mail:a#0", TEMPLATE), rec("mail:b#0", TEMPLATE + " Tot ziens."), rec("mail:c#0", "Iets anders.")]
    clusters = find_near_duplicates(records, 0.85)
    assert len(clusters) == 1
    assert clusters[0].member_ids == ("mail:a#0", "mail:b#0")
    assert clusters[0].representative_id == "mail:b#0"  # the longest text
    assert clusters[0].min_similarity >= 0.85


def test_representative_tie_break_is_the_smallest_id() -> None:
    records = [rec("mail:z#0", TEMPLATE), rec("mail:a#0", TEMPLATE)]
    assert find_near_duplicates(records)[0].representative_id == "mail:a#0"


def test_distinct_texts_are_not_clustered() -> None:
    texts = [
        "Ik wil mijn woningverzekering opzeggen omdat ik verhuis naar het buitenland.",
        "Kan ik een offerte krijgen voor een omniumverzekering voor mijn nieuwe wagen?",
        "Ik ben ontevreden over de behandeling van mijn klacht en wil de ombudsman contacteren.",
    ]
    assert find_near_duplicates([rec(f"m:{i}#0", t) for i, t in enumerate(texts)]) == []


def test_clusters_are_transitive_and_deterministic() -> None:
    base = TEMPLATE
    variants = [base, base + " Groet.", base + " Groet. Bedankt."]
    records = [rec(f"m:{i}#0", t) for i, t in enumerate(variants)]
    first = find_near_duplicates(records)
    assert [c.member_ids for c in first] == [("m:0#0", "m:1#0", "m:2#0")]
    assert find_near_duplicates(list(reversed(records))) == first


def test_cross_source_cluster_reports_both_sources() -> None:
    records = [rec("crm_postgres:claim_notes:1#0", TEMPLATE), rec("support_mailbox:csv:THR-1#0", TEMPLATE)]
    assert find_near_duplicates(records)[0].sources == ("crm_postgres", "support_mailbox")


def test_700_records_without_quadratic_blowup() -> None:
    rng = random.Random(7)
    words = [f"woord{i}" for i in range(400)]
    records = [rec(f"m:{i:04d}#0", " ".join(rng.choices(words, k=60))) for i in range(690)]
    records += [rec(f"m:dup{i}#0", records[i].text + " extra") for i in range(10)]
    start = time.perf_counter()
    clusters = find_near_duplicates(records)
    elapsed = time.perf_counter() - start
    assert len(clusters) == 10
    assert elapsed < 20, elapsed


def test_cluster_members_share_a_group_and_never_split() -> None:
    # Two near-duplicate records from different groups, each group with a sibling record.
    records = [
        rec("mail:csv#0", TEMPLATE, group="g-csv"),
        rec("mail:csv#1", "Tweede deel van hetzelfde document over de nieuwe polis.", group="g-csv"),
        rec("mail:eml#0", TEMPLATE + " Tot ziens.", group="g-eml"),
        rec("mail:eml#1", "Ander fragment uit de mailbox over een offerte.", group="g-eml"),
    ]
    other_domains: tuple[Domain, ...] = ("claims", "billing", "complaints", "policies", "compliance")
    filler = [rec(f"x:{d}{i}#0", f"tekst {d} nummer {i} " * 3, domain=d) for d in other_domains for i in range(30)]
    clusters = find_near_duplicates(records + filler)
    mapping = merge_groups(records, clusters)
    assert mapping["g-csv"] == mapping["g-eml"]
    for r in records:
        r.group_id = mapping.get(r.group_id, r.group_id)
    assert len({r.group_id for r in records}) == 1
    result = split(records + filler, SplitConfig(min_train_per_domain=1, eval_floor_per_domain=1,
                                                eval_floor_per_cell=0, max_share_per_domain=1.0))
    sides = {("train" if r in result.train else "eval") for r in records if r in result.train + result.eval}
    assert len(sides) == 1


def test_docstring_states_ordering_and_counter_argument() -> None:
    doc = dedup_near.__doc__ or ""
    assert "AFTER anonymisation" in doc and "COUNTER-ARGUMENT" in doc
