"""Group-level train/eval split, stratified by (domain, language).

Guiding principle: the eval set does NOT follow the train distribution.
Train is shaped by what we want to learn; eval is floored from below so
that every domain is measurable, and bounded from above by eval_ratio_max.
The reference metric is the worst domain, not the average.

Units:
  - the atomic unit is the GROUP (group_id): a group is never split;
  - every floor, minimum, ratio and cap counts EXAMPLES (records). Whole
    groups are added until a floor is reached or exceeded, then we stop.

Resolution order, per domain, when constraints conflict:
  1. reserve min_train_per_domain examples for train FIRST
  2. fill the eval floors (per cell, then per domain) from what remains
  3. top up eval towards eval_ratio, within the global eval budget
  4. cap each domain's share of TRAIN at max_share_per_domain
     (never below min_train_per_domain, never touching eval)
  5. if eval > eval_ratio_max of kept examples (train after cap + eval),
     shrink the eval budget and redo 2-5. Floors are scaled down
     proportionally whenever they do not fit the budget; the factor is
     reported, never silent.

Domain status:
  ABSENT                no records at all
  INSUFFICIENT_DATA     fewer than min_train_per_domain examples available
  PARTIALLY_MEASURABLE  eval floor (domain or a non-empty cell) not reached
  MEASURABLE            eval floor reached
"""

import hashlib
import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, get_args

from src.contracts import Domain, Record

logger = logging.getLogger(__name__)

Cell = tuple[str, str]  # (domain, language)

ALL_DOMAINS: tuple[str, ...] = get_args(Domain)
FLOORED_LANGUAGES: tuple[str, ...] = ("nl", "fr", "en")

STATUS_ABSENT: Final = "ABSENT"
STATUS_INSUFFICIENT: Final = "INSUFFICIENT_DATA"
STATUS_PARTIAL: Final = "PARTIALLY_MEASURABLE"
STATUS_MEASURABLE: Final = "MEASURABLE"
ALL_STATUSES: Final = (STATUS_ABSENT, STATUS_INSUFFICIENT, STATUS_PARTIAL, STATUS_MEASURABLE)


@dataclass
class SplitConfig:
    eval_ratio: float = 0.2  # pro-rata top-up target per cell
    eval_floor_per_domain: int = 25  # examples
    eval_floor_per_cell: int = 5  # examples, per (domain, language)
    max_share_per_domain: float = 0.35  # cap, TRAIN side only
    seed: int = 42
    min_train_per_domain: int = 20  # examples; takes precedence over the floors
    eval_ratio_max: float = 0.30  # hard ceiling: eval / (train + eval), after the cap
    # Universe of cells reported in per_cell, even when empty.
    domains: tuple[str, ...] = ALL_DOMAINS
    languages: tuple[str, ...] = FLOORED_LANGUAGES


@dataclass
class SplitResult:
    train: list[Record]
    eval: list[Record]
    per_cell: dict[Cell, dict[str, Any]]  # (domain, lang) -> {train, eval, floor, floor_met, ...}
    domain_status: dict[str, str]  # domain -> one of ALL_STATUSES
    floor_scaling_applied: float | None  # None when no scaling was needed
    capped_per_domain: dict[str, int]  # domain -> examples removed by the train cap
    # Records removed by the train cap. The pipeline reports them through
    # Counters.cap(), never Counters.drop(): capping is not a quality rejection.
    capped_records: list[Record] = field(default_factory=list)
    per_domain: dict[str, dict[str, Any]] = field(default_factory=dict)
    effective_eval_floor_per_domain: int = 0
    effective_eval_floor_per_cell: int = 0
    eval_budget: int = 0
    iterations: int = 0


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
    if not 0.0 <= cfg.eval_ratio_max < 1.0:
        raise ValueError("eval_ratio_max must be in [0, 1)")
    if cfg.eval_floor_per_cell < 0 or cfg.eval_floor_per_domain < 0:
        raise ValueError("eval floors must be >= 0")
    if cfg.min_train_per_domain < 1:
        raise ValueError("min_train_per_domain must be >= 1 (no domain may have eval without train)")
    if not 0.0 < cfg.max_share_per_domain <= 1.0:
        raise ValueError("max_share_per_domain must be in (0, 1]")


def _size(groups: list[_Group]) -> int:
    return sum(g.size for g in groups)


def _flat(by_cell: dict[Cell, list[_Group]]) -> list[_Group]:
    return [g for gs in by_cell.values() for g in gs]


def split(records: list[Record], cfg: SplitConfig) -> SplitResult:
    _validate(cfg)
    groups = _build_groups(records, cfg.seed)

    # --- Cells, groups in deterministic hash order (lowest first).
    cell_groups: dict[Cell, list[_Group]] = {
        (d, lang): [] for d in cfg.domains for lang in cfg.languages
    }
    for g in groups:
        cell_groups.setdefault((g.domain, g.language), []).append(g)
    for gs in cell_groups.values():
        gs.sort(key=lambda g: g.order_key)

    cell_total = {c: _size(gs) for c, gs in cell_groups.items()}
    domain_total: Counter[str] = Counter()
    for (d, _), n in cell_total.items():
        domain_total[d] += n
    domains = sorted({c[0] for c in cell_groups}, key=lambda d: (domain_total[d], d))
    # Poorest first: rare domains, then poor cells; names as final tie-break.
    cells = sorted(cell_groups, key=lambda c: (domain_total[c[0]], cell_total[c], c[0], c[1]))
    total = sum(cell_total.values())

    # --- 1. Reserve min_train examples per domain, round-robin over its cells
    #        (highest hash first, so eval selection keeps the lowest hashes).
    available: dict[Cell, list[_Group]] = {c: list(gs) for c, gs in cell_groups.items()}
    reserved: dict[Cell, list[_Group]] = {c: [] for c in cell_groups}
    reserved_total: dict[str, int] = {}
    for domain in domains:
        dcells = [c for c in cells if c[0] == domain]
        got = 0
        while got < cfg.min_train_per_domain:
            progressed = False
            for cell in dcells:
                if got >= cfg.min_train_per_domain:
                    break
                if available[cell]:
                    g = available[cell].pop()
                    reserved[cell].append(g)
                    got += g.size
                    progressed = True
            if not progressed:
                break
        reserved_total[domain] = got

    # --- Floor demand vs. eval budget.
    present = [d for d in domains if domain_total[d] > 0]

    def demand(d: str) -> int:
        nonempty = sum(1 for c in cells if c[0] == d and cell_total[c] > 0 and c[1] in cfg.languages)
        return max(cfg.eval_floor_per_domain, cfg.eval_floor_per_cell * nonempty)

    floor_sum = sum(demand(d) for d in present)
    budget = math.floor(cfg.eval_ratio_max * total)
    ratio_factor = cfg.eval_ratio_max / (1.0 - cfg.eval_ratio_max)

    # --- 2-5. Fill eval, cap train, check the ceiling; shrink budget if needed.
    for iteration in range(1, total + 3):
        scale = 1.0 if floor_sum <= budget else (budget / floor_sum)
        dom_floor = math.floor(cfg.eval_floor_per_domain * scale)
        cell_floor = math.floor(cfg.eval_floor_per_cell * scale)

        eval_by_cell, leftover = _fill_eval(
            available, cells, cell_total, dom_floor, cell_floor, budget, cfg
        )
        train_by_cell = {c: reserved[c] + leftover[c] for c in cell_groups}
        capped_groups = _apply_train_cap(
            train_by_cell, cfg.max_share_per_domain, cfg.min_train_per_domain
        )
        n_eval = _size(_flat(eval_by_cell))
        n_train = _size(_flat(train_by_cell))
        if n_eval <= cfg.eval_ratio_max * (n_eval + n_train) + 1e-9:
            break
        budget = max(0, min(budget - 1, math.floor(ratio_factor * n_train)))
    else:  # pragma: no cover - budget reaches 0, which always satisfies the ceiling
        raise RuntimeError("eval ceiling could not be satisfied")

    # --- Statuses and reporting.
    capped_per_domain: dict[str, int] = {}
    capped_per_cell: Counter[Cell] = Counter()
    for g in capped_groups:
        capped_per_domain[g.domain] = capped_per_domain.get(g.domain, 0) + g.size
        capped_per_cell[(g.domain, g.language)] += g.size

    per_cell: dict[Cell, dict[str, Any]] = {}
    for cell in sorted(cell_groups):
        floor = cell_floor if cell[1] in cfg.languages else 0
        n_eval_cell = _size(eval_by_cell[cell])
        per_cell[cell] = {
            "total": cell_total[cell],
            "train": _size(train_by_cell[cell]),
            "eval": n_eval_cell,
            "train_groups": len(train_by_cell[cell]),
            "eval_groups": len(eval_by_cell[cell]),
            "capped": capped_per_cell[cell],
            "floor": floor,
            "floor_met": n_eval_cell >= floor,
        }

    domain_status: dict[str, str] = {}
    per_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted(domain_total):
        dcells = [c for c in per_cell if c[0] == domain]
        d_eval = sum(per_cell[c]["eval"] for c in dcells)
        d_train = sum(per_cell[c]["train"] for c in dcells)
        if domain_total[domain] == 0:
            status = STATUS_ABSENT
        elif reserved_total[domain] < cfg.min_train_per_domain:
            status = STATUS_INSUFFICIENT
        elif d_eval < dom_floor or any(
            per_cell[c]["total"] > 0 and not per_cell[c]["floor_met"] for c in dcells
        ):
            status = STATUS_PARTIAL
        else:
            status = STATUS_MEASURABLE
        domain_status[domain] = status
        per_domain[domain] = {
            "total": domain_total[domain],
            "train": d_train,
            "eval": d_eval,
            "capped": capped_per_domain.get(domain, 0),
            "eval_floor": dom_floor,
            "min_train": cfg.min_train_per_domain,
            "status": status,
        }

    def records_of(gs: list[_Group]) -> list[Record]:
        return [r for g in sorted(gs, key=lambda g: g.order_key) for r in g.records]

    result = SplitResult(
        train=records_of(_flat(train_by_cell)),
        eval=records_of(_flat(eval_by_cell)),
        per_cell=per_cell,
        domain_status=domain_status,
        floor_scaling_applied=round(scale, 6) if scale < 1.0 else None,
        capped_per_domain=dict(sorted(capped_per_domain.items())),
        capped_records=records_of(capped_groups),
        per_domain=per_domain,
        effective_eval_floor_per_domain=dom_floor,
        effective_eval_floor_per_cell=cell_floor,
        eval_budget=budget,
        iterations=iteration,
    )
    logger.info(
        "split_done groups=%d train=%d eval=%d capped=%d floor_scaling=%s iterations=%d",
        len(groups), len(result.train), len(result.eval), len(result.capped_records),
        result.floor_scaling_applied if result.floor_scaling_applied is not None else "none",
        iteration,
    )
    return result


def _fill_eval(
    available: dict[Cell, list[_Group]],
    cells: list[Cell],
    cell_total: dict[Cell, int],
    dom_floor: int,
    cell_floor: int,
    budget: int,
    cfg: SplitConfig,
) -> tuple[dict[Cell, list[_Group]], dict[Cell, list[_Group]]]:
    """Pick eval groups from the non-reserved groups. Returns (eval, leftover)."""
    remaining = {c: list(gs) for c, gs in available.items()}
    eval_by_cell: dict[Cell, list[_Group]] = {c: [] for c in available}
    n_eval = 0

    def take(cell: Cell) -> int:
        nonlocal n_eval
        g = remaining[cell].pop(0)
        eval_by_cell[cell].append(g)
        n_eval += g.size
        return g.size

    # Cell floors (examples). Never borrow from another cell.
    for cell in cells:
        if cell[1] not in cfg.languages:
            continue
        while _size(eval_by_cell[cell]) < cell_floor and remaining[cell]:
            take(cell)

    # Domain floors: round-robin over the domain's own cells.
    for domain in dict.fromkeys(c[0] for c in cells):
        dcells = [c for c in cells if c[0] == domain]
        have = sum(_size(eval_by_cell[c]) for c in dcells)
        while have < dom_floor:
            progressed = False
            for cell in dcells:
                if have >= dom_floor:
                    break
                if remaining[cell]:
                    have += take(cell)
                    progressed = True
            if not progressed:
                break

    # Pro-rata top-up towards eval_ratio per cell, within the global budget.
    for cell in cells:
        target = cfg.eval_ratio * cell_total[cell]
        current = _size(eval_by_cell[cell])
        while (
            remaining[cell]
            and current < target
            and n_eval + remaining[cell][0].size <= budget
        ):
            current += take(cell)

    return eval_by_cell, remaining


def _apply_train_cap(
    train: dict[Cell, list[_Group]], max_share: float, min_train: int
) -> list[_Group]:
    """Remove train groups until no domain exceeds max_share of train examples.

    Repeatedly drops the highest-hash group of the domain with the largest
    share (deterministic). A domain is never reduced below min_train. Mutates
    ``train`` and returns the removed groups.
    """
    capped: list[_Group] = []
    if max_share >= 1.0:
        return capped

    stuck: set[str] = set()  # over-share domains that cannot shrink without breaking min_train
    while True:
        per_domain: Counter[str] = Counter()
        for (domain, _), gs in train.items():
            per_domain[domain] += _size(gs)
        total = sum(per_domain.values())
        if total == 0:
            return capped
        present = [d for d, n in per_domain.items() if n > 0]
        if len(present) * max_share < 1.0:
            raise ValueError(
                f"max_share_per_domain={max_share} is infeasible with "
                f"{len(present)} domain(s) present in train"
            )
        over = [d for d in present if per_domain[d] / total > max_share and d not in stuck]
        if not over:
            if stuck:
                logger.warning("train_cap_limited_by_min_train domains=%s", ",".join(sorted(stuck)))
            return capped
        worst = max(over, key=lambda d: (per_domain[d], d))
        candidates = sorted(
            ((g.order_key, cell, g) for cell, gs in train.items() if cell[0] == worst for g in gs),
            key=lambda x: x[0],
            reverse=True,
        )
        victim = next(
            ((cell, g) for _, cell, g in candidates if per_domain[worst] - g.size >= min_train),
            None,
        )
        if victim is None:
            stuck.add(worst)
            continue
        cell, g = victim
        train[cell].remove(g)
        capped.append(g)
