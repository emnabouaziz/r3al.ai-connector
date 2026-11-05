from collections import Counter
from dataclasses import replace
from typing import get_args

import pytest

from src.contracts import Domain, Language, Record
from src.dataset.leakage import assert_no_group_overlap, assert_no_near_duplicate_across_split
from src.dataset.splitter import (
    STATUS_ABSENT,
    STATUS_INSUFFICIENT,
    STATUS_MEASURABLE,
    STATUS_PARTIAL,
    SplitConfig,
    SplitResult,
    split,
)

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


def make_dataset(spec: dict[Domain, int], group_size: int = 1) -> list[Record]:
    """spec: domain -> number of groups, spread round-robin over nl/fr/en."""
    records: list[Record] = []
    for domain, n_groups in spec.items():
        for g in range(n_groups):
            records += make_group(f"{domain}-g{g}", group_size, domain, LANGS[g % len(LANGS)])
    return records


def per_domain(records: list[Record]) -> Counter[str]:
    return Counter(str(r.domain) for r in records)


def eval_share(result: SplitResult) -> float:
    kept = len(result.train) + len(result.eval)
    return len(result.eval) / kept if kept else 0.0


def assert_core_invariants(result: SplitResult, cfg: SplitConfig, n_input: int) -> None:
    # Every input record lands in exactly one of train / eval / capped.
    assert len(result.train) + len(result.eval) + len(result.capped_records) == n_input
    assert_no_group_overlap(result.train, result.eval)
    # Eval ceiling, on kept examples (after the cap).
    assert eval_share(result) <= cfg.eval_ratio_max + 1e-9
    # No domain with eval but zero train.
    train, eval_ = per_domain(result.train), per_domain(result.eval)
    for domain, n in eval_.items():
        assert n == 0 or train[domain] > 0, domain
    # Every domain carries an explicit status.
    assert set(result.domain_status) >= set(DOMAINS)


BALANCED: dict[Domain, int] = {d: 100 for d in DOMAINS}


# --- Group integrity ----------------------------------------------------------


def test_1000_records_200_groups_no_shared_group_id() -> None:
    records = make_dataset({d: 34 for d in DOMAINS[:5]} | {DOMAINS[5]: 30}, group_size=5)
    assert len(records) == 1000 and len({r.group_id for r in records}) == 200
    cfg = SplitConfig()
    result = split(records, cfg)
    assert not {r.group_id for r in result.train} & {r.group_id for r in result.eval}
    assert_core_invariants(result, cfg, 1000)


def test_group_of_40_records_stays_whole() -> None:
    records = make_dataset(BALANCED) + make_group("big", 40, "claims", "nl")
    result = split(records, SplitConfig())
    sides = [
        sum(r.group_id == "big" for r in side)
        for side in (result.train, result.eval, result.capped_records)
    ]
    assert sorted(sides) == [0, 0, 40]


def test_floor_is_reached_with_whole_groups_then_stops() -> None:
    # Groups of 4 examples, cell floor 5 -> exactly 2 groups (8 examples) per cell.
    records = make_dataset(BALANCED, group_size=4)
    cfg = SplitConfig(eval_floor_per_domain=0, eval_floor_per_cell=5, eval_ratio=0.0)
    result = split(records, cfg)
    for (_, lang), cell in result.per_cell.items():
        if lang in LANGS:
            assert cell["eval"] == 8 and cell["eval_groups"] == 2


def test_group_majority_vote_sets_cell() -> None:
    recs = make_group("mixed", 3, "billing", "fr")
    recs[0] = replace(recs[0], domain="claims", language="nl")
    base = make_dataset(BALANCED)
    baseline = split(base, SplitConfig()).per_cell[("billing", "fr")]
    result = split(base + recs, SplitConfig())
    cell = result.per_cell[("billing", "fr")]
    assert cell["total"] == baseline["total"] + 3


def test_group_without_domain_is_rejected() -> None:
    recs = [replace(r, domain=None) for r in make_group("g", 2, "claims")]
    with pytest.raises(ValueError, match="domain_unresolved"):
        split(recs, SplitConfig())


# --- D2: min_train takes precedence, explicit statuses --------------------------


def test_domain_with_30_examples_keeps_20_in_train_and_is_partial() -> None:
    spec: dict[Domain, int] = {d: 100 for d in DOMAINS if d != "compliance"} | {"compliance": 30}
    records = make_dataset(spec)
    cfg = SplitConfig(max_share_per_domain=1.0)
    result = split(records, cfg)
    assert result.floor_scaling_applied is None  # the status is not an artefact of scaling
    assert per_domain(result.train)["compliance"] >= 20
    assert result.domain_status["compliance"] == STATUS_PARTIAL
    assert result.per_domain["compliance"]["eval"] < cfg.eval_floor_per_domain
    assert_core_invariants(result, cfg, len(records))


def test_domain_with_12_examples_is_insufficient_but_trains() -> None:
    spec: dict[Domain, int] = {d: 100 for d in DOMAINS if d != "compliance"} | {"compliance": 12}
    records = make_dataset(spec)
    cfg = SplitConfig()
    result = split(records, cfg)
    assert result.domain_status["compliance"] == STATUS_INSUFFICIENT
    assert per_domain(result.train)["compliance"] == 12
    assert per_domain(result.eval)["compliance"] == 0
    assert_core_invariants(result, cfg, len(records))


def test_domain_with_0_examples_is_absent_and_reported_with_zeros() -> None:
    spec: dict[Domain, int] = {d: 100 for d in DOMAINS if d != "compliance"}
    result = split(make_dataset(spec), SplitConfig())
    assert result.domain_status["compliance"] == STATUS_ABSENT
    stats = result.per_domain["compliance"]
    assert (stats["total"], stats["train"], stats["eval"], stats["capped"]) == (0, 0, 0, 0)
    assert stats["status"] == STATUS_ABSENT
    for lang in LANGS:
        cell = result.per_cell[("compliance", lang)]
        assert cell["train"] == cell["eval"] == cell["total"] == 0
    assert len(result.per_cell) == len(DOMAINS) * len(LANGS)


def test_large_balanced_domains_are_measurable() -> None:
    cfg = SplitConfig()
    result = split(make_dataset(BALANCED), cfg)
    assert set(result.domain_status.values()) == {STATUS_MEASURABLE}
    assert_core_invariants(result, cfg, 600)


def test_no_domain_has_eval_without_train_across_sizes() -> None:
    for size in (1, 2, 5, 12, 19, 20, 21, 30, 45, 60):
        spec: dict[Domain, int] = {d: 80 for d in DOMAINS if d != "compliance"} | {"compliance": size}
        records = make_dataset(spec)
        cfg = SplitConfig()
        assert_core_invariants(split(records, cfg), cfg, len(records))


# --- D3: floors in examples, scaling, eval ceiling ------------------------------


def test_floors_above_ceiling_are_scaled_and_reported() -> None:
    # 6 domains x 40 examples: floors demand 6 x 25 = 150 > 0.30 x 240 = 72.
    records = make_dataset({d: 40 for d in DOMAINS})
    cfg = SplitConfig(min_train_per_domain=5, max_share_per_domain=1.0)
    result = split(records, cfg)
    assert result.floor_scaling_applied is not None
    assert 0.0 < result.floor_scaling_applied < 1.0
    assert result.effective_eval_floor_per_domain < cfg.eval_floor_per_domain
    assert_core_invariants(result, cfg, len(records))


def test_no_scaling_when_floors_fit() -> None:
    result = split(make_dataset(BALANCED), SplitConfig())
    assert result.floor_scaling_applied is None
    assert result.effective_eval_floor_per_domain == 25


@pytest.mark.parametrize("group_size", [1, 3, 7])
@pytest.mark.parametrize("ratio_max", [0.1, 0.2, 0.3])
def test_eval_never_exceeds_ceiling(group_size: int, ratio_max: float) -> None:
    spec: dict[Domain, int] = {"claims": 300, "policies": 60, "billing": 40,
                               "onboarding": 25, "complaints": 15, "compliance": 8}
    records = make_dataset(spec, group_size=group_size)
    cfg = SplitConfig(eval_ratio_max=ratio_max)
    assert_core_invariants(split(records, cfg), cfg, len(records))


# --- Train cap (sampling decision) ----------------------------------------------


def _claims_60_percent() -> list[Record]:
    spec: dict[Domain, int] = {"claims": 450}
    return make_dataset(spec | {d: 60 for d in DOMAINS if d != "claims"})


def test_claims_at_60_percent_is_capped_in_train() -> None:
    records = _claims_60_percent()
    cfg = SplitConfig()
    result = split(records, cfg)
    assert result.capped_per_domain["claims"] > 0
    assert per_domain(result.train)["claims"] / len(result.train) <= cfg.max_share_per_domain
    assert sum(result.capped_per_domain.values()) == len(result.capped_records)
    assert_core_invariants(result, cfg, len(records))


def test_cap_never_touches_eval() -> None:
    result = split(_claims_60_percent(), SplitConfig())
    assert result.capped_records
    assert not {r.group_id for r in result.capped_records} & {r.group_id for r in result.eval}


def test_cap_never_goes_below_min_train() -> None:
    records = make_dataset({"claims": 500, "billing": 30, "policies": 30})
    result = split(records, SplitConfig(min_train_per_domain=20))
    train = per_domain(result.train)
    assert all(train[d] >= 20 for d in ("claims", "billing", "policies"))


def test_infeasible_cap_raises() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        split(make_dataset({"claims": 100, "billing": 100}), SplitConfig())


@pytest.mark.parametrize(
    "cfg", [SplitConfig(min_train_per_domain=0), SplitConfig(eval_ratio_max=1.0)]
)
def test_invalid_config_is_rejected(cfg: SplitConfig) -> None:
    with pytest.raises(ValueError):
        split(make_dataset(BALANCED), cfg)


# --- Determinism and leakage ----------------------------------------------------


def test_same_seed_same_result_regardless_of_input_order() -> None:
    spec: dict[Domain, int] = {"claims": 300, "policies": 60, "billing": 40,
                               "onboarding": 25, "complaints": 15, "compliance": 8}
    records = make_dataset(spec, group_size=3)
    a = split(records, SplitConfig())
    b = split(records, SplitConfig())
    c = split(list(reversed(records)), SplitConfig())
    for other in (b, c):
        assert [r.id for r in a.train] == [r.id for r in other.train]
        assert [r.id for r in a.eval] == [r.id for r in other.eval]
        assert [r.id for r in a.capped_records] == [r.id for r in other.capped_records]
        assert a.per_cell == other.per_cell and a.per_domain == other.per_domain
        assert a.domain_status == other.domain_status
        assert a.floor_scaling_applied == other.floor_scaling_applied


def test_different_seed_changes_assignment() -> None:
    records = make_dataset(BALANCED)
    a = split(records, SplitConfig(seed=1))
    b = split(records, SplitConfig(seed=2))
    assert {r.id for r in a.eval} != {r.id for r in b.eval}


def test_leakage_assertions_still_hold() -> None:
    result = split(make_dataset(BALANCED, group_size=2), SplitConfig())
    assert_no_group_overlap(result.train, result.eval)
    assert assert_no_near_duplicate_across_split(result.train, result.eval) == []
