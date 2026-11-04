"""Group-level train/eval split, stratified by (domain, language).

Guiding principle: the eval set does NOT follow the train distribution.
Train is shaped by what we want to learn; eval is floored from below so
that EVERY domain is measurable. The reference metric is the worst domain,
not the average.

Units:
  - the atomic unit is the GROUP (group_id): a group is never split;
  - floors (eval_floor_per_cell, eval_floor_per_domain) count GROUPS,
    because groups are the statistically independent samples;
  - eval_ratio and max_share_per_domain count RECORDS.
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, get_args

from src.contracts import Domain, Record

logger = logging.getLogger(__name__)

Cell = tuple[str, str]  # (domain, language)

ALL_DOMAINS: tuple[str, ...] = get_args(Domain)
FLOORED_LANGUAGES: tuple[str, ...] = ("nl", "fr", "en")


@dataclass
class SplitConfig:
    eval_ratio: float = 0.2
    eval_floor_per_domain: int = 25  # absolute floor, in groups
    eval_floor_per_cell: int = 5  # floor per (domain, language), in groups
    max_share_per_domain: float = 0.35  # cap, TRAIN side only, in records
    seed: int = 42
    # Universe of cells reported in per_cell, even when empty.
    domains: tuple[str, ...] = ALL_DOMAINS
    languages: tuple[str, ...] = FLOORED_LANGUAGES


@dataclass
class SplitResult:
    train: list[Record]
    eval: list[Record]
    per_cell: dict[Cell, dict[str, Any]]  # (domain,lang) -> {train,eval,floor_met,...}
    unmeasurable_domains: list[str]  # floor not met
    capped_domains: dict[str, int]  # domain -> records removed by the train cap
    # Records removed by the train cap: the pipeline must count them as drops
    # (stage="build", reason="train_domain_cap") to keep the counter invariant.
    capped_records: list[Record] = field(default_factory=list)
    unmeasurable_cells: list[Cell] = field(default_factory=list)


@dataclass
class _Group:
    group_id: str
    records: list[Record]
    domain: str
    language: str
    order_key: str

    @property
    def size(self) -> int:
        return len(self.records)


def _order_key(group_id: str, seed: int) -> str:
    return hashlib.sha256((group_id + str(seed)).encode("utf-8")).hexdigest()


def _majority(values: Sequence[str]) -> str:
    """Most common value; ties broken alphabetically (deterministic)."""
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def _build_groups(records: list[Record], seed: int) -> list[_Group]:
    by_group: dict[str, list[Record]] = {}
    for rec in records:
        by_group.setdefault(rec.group_id, []).append(rec)

    groups: list[_Group] = []
    for group_id, recs in by_group.items():
        recs = sorted(recs, key=lambda r: r.id)
        domains = [r.domain for r in recs if r.domain is not None]
        if not domains:
            raise ValueError(
                f"group {group_id} has no resolved domain; drop its records upstream "
                "with reason 'domain_unresolved'"
            )
        known_langs = [r.language for r in recs if r.language != "unknown"]
        language = _majority(known_langs) if known_langs else "unknown"
        groups.append(
            _Group(group_id, recs, _majority(domains), language, _order_key(group_id, seed))
        )
    return groups


def _validate(cfg: SplitConfig) -> None:
    if not 0.0 <= cfg.eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    if cfg.eval_floor_per_cell < 0 or cfg.eval_floor_per_domain < 0:
        raise ValueError("eval floors must be >= 0")
    if not 0.0 < cfg.max_share_per_domain <= 1.0:
        raise ValueError("max_share_per_domain must be in (0, 1]")


def split(records: list[Record], cfg: SplitConfig) -> SplitResult:
    _validate(cfg)
    groups = _build_groups(records, cfg.seed)

    # --- 1-2. Cells, each holding its groups in deterministic hash order.
    remaining: dict[Cell, list[_Group]] = {
        (d, lang): [] for d in cfg.domains for lang in cfg.languages
    }
    for g in groups:
        remaining.setdefault((g.domain, g.language), []).append(g)
    for cell_groups in remaining.values():
        cell_groups.sort(key=lambda g: g.order_key)

    cell_records = {c: sum(g.size for g in gs) for c, gs in remaining.items()}
    cell_group_count = {c: len(gs) for c, gs in remaining.items()}
    domain_group_count: Counter[str] = Counter()
    for (d, _), n in cell_group_count.items():
        domain_group_count[d] += n

    # Poorest first: rare domains, then poor cells; name as final tie-break.
    cells = sorted(
        remaining,
        key=lambda c: (domain_group_count[c[0]], cell_group_count[c], c[0], c[1]),
    )
    domains_by_rarity = sorted(
        {c[0] for c in cells}, key=lambda d: (domain_group_count[d], d)
    )

    eval_groups: dict[Cell, list[_Group]] = {c: [] for c in cells}
    unmeasurable: set[str] = set()
    unmeasurable_cells: list[Cell] = []
    cell_floor: dict[Cell, int] = {}

    def take(cell: Cell) -> _Group:
        g = remaining[cell].pop(0)
        eval_groups[cell].append(g)
        return g

    # --- 3a/4. Cell floor. Never borrow from another cell.
    for cell in cells:
        floor = cfg.eval_floor_per_cell if cell[1] in cfg.languages else 0
        cell_floor[cell] = floor
        while len(eval_groups[cell]) < floor and remaining[cell]:
            take(cell)
        if len(eval_groups[cell]) < floor:
            unmeasurable.add(cell[0])
            unmeasurable_cells.append(cell)

    # --- 3b/4. Domain floor: round-robin over the domain's own cells.
    for domain in domains_by_rarity:
        domain_cells = [c for c in cells if c[0] == domain]
        have = sum(len(eval_groups[c]) for c in domain_cells)
        while have < cfg.eval_floor_per_domain:
            progressed = False
            for cell in domain_cells:
                if have >= cfg.eval_floor_per_domain:
                    break
                if remaining[cell]:
                    take(cell)
                    have += 1
                    progressed = True
            if not progressed:
                break
        if have < cfg.eval_floor_per_domain:
            unmeasurable.add(domain)

    # --- 3c. Pro-rata top-up per cell, towards eval_ratio of its records.
    for cell in cells:
        target = cfg.eval_ratio * cell_records[cell]
        current = sum(g.size for g in eval_groups[cell])
        while remaining[cell] and current < target:
            current += take(cell).size

    # --- 5. The rest goes to train.
    train_groups: dict[Cell, list[_Group]] = {c: list(gs) for c, gs in remaining.items()}

    # --- 6. Train-only cap on domain share. Eval is never touched.
    capped_groups = _apply_train_cap(train_groups, cfg.max_share_per_domain)

    # --- Result assembly (deterministic ordering, independent of input order).
    def flatten(gs_by_cell: dict[Cell, list[_Group]]) -> list[Record]:
        gs = sorted((g for gs in gs_by_cell.values() for g in gs), key=lambda g: g.order_key)
        return [r for g in gs for r in g.records]

    capped_domains: dict[str, int] = {}
    capped_per_cell: Counter[Cell] = Counter()
    for g in capped_groups:
        capped_domains[g.domain] = capped_domains.get(g.domain, 0) + g.size
        capped_per_cell[(g.domain, g.language)] += g.size

    per_cell: dict[Cell, dict[str, Any]] = {}
    for cell in sorted(cells):
        per_cell[cell] = {
            "train": sum(g.size for g in train_groups[cell]),
            "eval": sum(g.size for g in eval_groups[cell]),
            "train_groups": len(train_groups[cell]),
            "eval_groups": len(eval_groups[cell]),
            "capped": capped_per_cell[cell],
            "floor": cell_floor[cell],
            "floor_met": len(eval_groups[cell]) >= cell_floor[cell],
        }

    result = SplitResult(
        train=flatten(train_groups),
        eval=flatten(eval_groups),
        per_cell=per_cell,
        unmeasurable_domains=sorted(unmeasurable),
        capped_domains=dict(sorted(capped_domains.items())),
        capped_records=[r for g in sorted(capped_groups, key=lambda g: g.order_key)
                        for r in g.records],
        unmeasurable_cells=sorted(unmeasurable_cells),
    )
    logger.info(
        "split_done groups=%d train=%d eval=%d capped=%d unmeasurable_domains=%s",
        len(groups),
        len(result.train),
        len(result.eval),
        len(result.capped_records),
        ",".join(result.unmeasurable_domains) or "-",
    )
    return result


def _apply_train_cap(train: dict[Cell, list[_Group]], max_share: float) -> list[_Group]:
    """Remove train groups until no domain exceeds max_share of train records.

    Repeatedly drops the highest-hash group of the domain with the largest
    share (deterministic). Mutates ``train`` and returns the removed groups.
    """
    capped: list[_Group] = []
    if max_share >= 1.0:
        return capped

    while True:
        per_domain: Counter[str] = Counter()
        for (domain, _), gs in train.items():
            per_domain[domain] += sum(g.size for g in gs)
        total = sum(per_domain.values())
        if total == 0:
            return capped
        present = [d for d, n in per_domain.items() if n > 0]
        if len(present) * max_share < 1.0:
            raise ValueError(
                f"max_share_per_domain={max_share} is infeasible with "
                f"{len(present)} domain(s) present in train"
            )
        over = [d for d in present if per_domain[d] / total > max_share]
        if not over:
            return capped
        worst = max(over, key=lambda d: (per_domain[d], d))
        # Highest order_key among this domain's train groups.
        candidates = [(g.order_key, cell) for cell, gs in train.items()
                      if cell[0] == worst for g in gs]
        _, cell = max(candidates)
        victim = max(train[cell], key=lambda g: g.order_key)
        train[cell].remove(victim)
        capped.append(victim)
