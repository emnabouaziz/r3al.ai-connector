from collections import Counter
from dataclasses import replace
from typing import get_args

import pytest

from src.contracts import Domain, Language, Record
from src.dataset.leakage import assert_no_group_overlap
from src.dataset.splitter import SplitConfig, SplitResult, split

DOMAINS: tuple[Domain, ...] = get_args(Domain)
LANGS: tuple[Language, ...] = ("nl", "fr", "en")


def make_group(
    group_id: str, size: int, domain: Domain, language: Language = "nl"
) -> list[Record]:
    return [
        Record(
            id=f"{group_id}-r{i}",
            text=f"text {group_id} {i}",
            group_id=group_id,
            source_id=f"src:{group_id}",
            source_type="email",
            language=language,
            domain=domain,
        )
        for i in range(size)
    ]


def make_dataset(spec: dict[Domain, int], group_size: int = 5) -> list[Record]:
    """spec: domain -> number of groups, spread round-robin over nl/fr/en."""
    records: list[Record] = []
    for domain, n_groups in spec.items():
        for g in range(n_groups):
            lang = LANGS[g % len(LANGS)]
            records += make_group(f"{domain}-g{g}", group_size, domain, lang)
    return records


def train_share(result: SplitResult, domain: str) -> float:
    counts = Counter(r.domain for r in result.train)
    return counts[domain] / len(result.train)  # type: ignore[index]


def assert_floors_never_silently_missed(result: SplitResult, cfg: SplitConfig) -> None:
    eval_groups = Counter((r.domain, r.language) for r in {r.group_id: r for r in result.eval}.values())
    for domain in DOMAINS:
        domain_groups = sum(n for (d, _), n in eval_groups.items() if d == domain)
        cells_ok = all(result.per_cell[(domain, lang)]["floor_met"] for lang in LANGS)
        measurable = domain_groups >= cfg.eval_floor_per_domain and cells_ok
        assert measurable or domain in result.unmeasurable_domains, domain
        if domain in result.unmeasurable_domains:
            assert not measurable, domain


# --- Group integrity --------------------------------------------------------


def test_1000_records_200_groups_no_shared_group_id() -> None:
    records = make_dataset({d: 34 for d in DOMAINS[:5]} | {DOMAINS[5]: 30})
    assert len(records) == 1000 and len({r.group_id for r in records}) == 200
    result = split(records, SplitConfig())
    assert not {r.group_id for r in result.train} & {r.group_id for r in result.eval}
    assert_no_group_overlap(result.train, result.eval)
    assert len(result.train) + len(result.eval) + len(result.capped_records) == 1000


def test_group_of_40_records_stays_whole() -> None:
    records = make_dataset({d: 40 for d in DOMAINS}) + make_group("big", 40, "claims", "nl")
    result = split(records, SplitConfig())
    sides = [
        sum(r.group_id == "big" for r in side)
        for side in (result.train, result.eval, result.capped_records)
    ]
    assert sorted(sides) == [0, 0, 40]


def test_group_majority_vote_sets_cell() -> None:
    # 2 records billing/fr, 1 record claims/nl -> the group belongs to billing/fr.
    recs = make_group("mixed", 3, "billing", "fr")
    recs[0] = replace(recs[0], domain="claims", language="nl")
    base = make_dataset({d: 40 for d in DOMAINS})
    baseline = split(base, SplitConfig()).per_cell[("billing", "fr")]
    result = split(base + recs, SplitConfig())
    side = result.eval if any(r.group_id == "mixed" for r in result.eval) else result.train
    assert sum(r.group_id == "mixed" for r in side) == 3
    cell = result.per_cell[("billing", "fr")]
    total = cell["train_groups"] + cell["eval_groups"]
    assert total == baseline["train_groups"] + baseline["eval_groups"] + 1


def test_group_without_domain_is_rejected() -> None:
    recs = [replace(r, domain=None) for r in make_group("g", 2, "claims")]
    with pytest.raises(ValueError, match="domain_unresolved"):
        split(recs, SplitConfig())


# --- Floors ------------------------------------------------------------------


def test_imbalanced_compliance_is_flagged_unmeasurable() -> None:
    # claims 600 records, compliance 12 records (12 single-record groups).
    records = make_dataset({"claims": 120, "policies": 40, "billing": 40,
                            "onboarding": 40, "complaints": 40})
    records += [r for g in range(12)
                for r in make_group(f"compliance-g{g}", 1, "compliance", LANGS[g % 3])]
    assert sum(r.domain == "claims" for r in records) == 600
    assert sum(r.domain == "compliance" for r in records) == 12

    cfg = SplitConfig()
    result = split(records, cfg)
    assert "compliance" in result.unmeasurable_domains
    # Nothing borrowed: every compliance group went to eval, none elsewhere.
    assert sum(r.domain == "compliance" for r in result.eval) == 12
    assert_floors_never_silently_missed(result, cfg)


def test_rare_domain_with_enough_groups_meets_floor() -> None:
    spec: dict[Domain, int] = {"claims": 120, "policies": 40, "billing": 40,
                               "onboarding": 40, "complaints": 40, "compliance": 30}
    records = make_dataset(spec, group_size=2)
    cfg = SplitConfig()
    result = split(records, cfg)
    assert "compliance" not in result.unmeasurable_domains
    compliance_eval_groups = {r.group_id for r in result.eval if r.domain == "compliance"}
    assert len(compliance_eval_groups) >= cfg.eval_floor_per_domain
    assert_floors_never_silently_missed(result, cfg)


def test_empty_cell_is_reported_with_zeros() -> None:
    records = make_dataset({d: 40 for d in DOMAINS if d != "compliance"})
    records += [r for g in range(20) for r in make_group(f"c-nl{g}", 2, "compliance", "nl")]
    records += [r for g in range(20) for r in make_group(f"c-fr{g}", 2, "compliance", "fr")]
    result = split(records, SplitConfig())
    cell = result.per_cell[("compliance", "en")]
    assert cell["train"] == 0 and cell["eval"] == 0
    assert cell["train_groups"] == 0 and cell["eval_groups"] == 0
    assert cell["floor_met"] is False
    assert ("compliance", "en") in result.unmeasurable_cells
    assert "compliance" in result.unmeasurable_domains
    assert len(result.per_cell) == len(DOMAINS) * len(LANGS)


def test_eval_ratio_is_reached_when_floors_are_small() -> None:
    records = make_dataset({d: 300 for d in DOMAINS}, group_size=1)
    result = split(records, SplitConfig(max_share_per_domain=1.0))
    ratio = len(result.eval) / len(records)
    assert 0.19 <= ratio <= 0.22


# --- Train cap ---------------------------------------------------------------


def _claims_60_percent() -> list[Record]:
    # claims 450 groups, 5 other domains x 60 groups -> 450 / 750 = 60%.
    spec: dict[Domain, int] = {"claims": 450}
    return make_dataset(spec | {d: 60 for d in DOMAINS if d != "claims"})


def test_claims_at_60_percent_is_capped_in_train() -> None:
    records = _claims_60_percent()
    assert sum(r.domain == "claims" for r in records) / len(records) == pytest.approx(0.6)
    cfg = SplitConfig()
    result = split(records, cfg)
    assert result.capped_domains["claims"] > 0
    assert train_share(result, "claims") <= cfg.max_share_per_domain
    assert sum(result.capped_domains.values()) == len(result.capped_records)


def test_cap_never_touches_eval() -> None:
    records = _claims_60_percent()
    capped = split(records, SplitConfig())
    uncapped = split(records, SplitConfig(max_share_per_domain=1.0))
    assert capped.capped_domains and not uncapped.capped_domains
    assert [r.id for r in capped.eval] == [r.id for r in uncapped.eval]
    assert all(r.group_id not in {e.group_id for e in capped.eval}
               for r in capped.capped_records)


def test_rare_domain_below_floor_goes_entirely_to_eval() -> None:
    # Documented consequence of the floors: 20 groups < floor 25 -> no train data.
    base: dict[Domain, int] = {"claims": 300}
    records = make_dataset(base | {d: 20 for d in DOMAINS if d != "claims"})
    result = split(records, SplitConfig(max_share_per_domain=1.0))
    assert not any(r.domain == "compliance" for r in result.train)
    assert "compliance" in result.unmeasurable_domains


def test_infeasible_cap_raises() -> None:
    records = make_dataset({"claims": 40, "billing": 40})
    with pytest.raises(ValueError, match="infeasible"):
        split(records, SplitConfig())


# --- Determinism -------------------------------------------------------------


def test_same_seed_same_result_regardless_of_input_order() -> None:
    records = _claims_60_percent()
    a = split(records, SplitConfig())
    b = split(records, SplitConfig())
    c = split(list(reversed(records)), SplitConfig())
    for other in (b, c):
        assert [r.id for r in a.train] == [r.id for r in other.train]
        assert [r.id for r in a.eval] == [r.id for r in other.eval]
        assert a.per_cell == other.per_cell
        assert a.capped_domains == other.capped_domains


def test_different_seed_changes_assignment() -> None:
    records = make_dataset({d: 100 for d in DOMAINS}, group_size=1)
    a = split(records, SplitConfig(seed=1))
    b = split(records, SplitConfig(seed=2))
    assert {r.id for r in a.eval} != {r.id for r in b.eval}
