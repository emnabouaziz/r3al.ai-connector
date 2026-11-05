"""Blocking anti-leakage checks between train and eval.

Both checks raise LeakageError: they are meant to stop the pipeline, not
to emit warnings. Error messages carry ids and counts only, never content.
"""

import logging
from collections.abc import Sequence

from src.contracts import Record
from src.processing.minhash import bands_for, jaccard, shingles, signature

logger = logging.getLogger(__name__)

_MAX_IDS_IN_MESSAGE = 20

NearDuplicatePair = tuple[str, str, float]  # (train_record_id, eval_record_id, jaccard)


class LeakageError(Exception):
    """Raised when train and eval share groups or near-duplicate texts."""

    def __init__(
        self,
        message: str,
        group_ids: Sequence[str] = (),
        pairs: Sequence[NearDuplicatePair] = (),
    ) -> None:
        self.group_ids = list(group_ids)
        self.pairs = list(pairs)
        super().__init__(message)


def _preview(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:_MAX_IDS_IN_MESSAGE])
    extra = len(ids) - _MAX_IDS_IN_MESSAGE
    return shown + (f", ... (+{extra})" if extra > 0 else "")


def assert_no_group_overlap(train: Sequence[Record], eval: Sequence[Record]) -> None:
    """Lève LeakageError avec les group_id fautifs.

    Raise LeakageError listing the offending group_ids (also checks record ids).
    """
    shared_groups = sorted({r.group_id for r in train} & {r.group_id for r in eval})
    if shared_groups:
        logger.error("leakage_group_overlap groups=%d", len(shared_groups))
        raise LeakageError(
            f"{len(shared_groups)} group_id(s) present in both train and eval: "
            f"{_preview(shared_groups)}",
            group_ids=shared_groups,
        )
    shared_ids = sorted({r.id for r in train} & {r.id for r in eval})
    if shared_ids:
        logger.error("leakage_record_overlap records=%d", len(shared_ids))
        raise LeakageError(
            f"{len(shared_ids)} record id(s) present in both train and eval: "
            f"{_preview(shared_ids)}"
        )


# --- MinHash / LSH (shared implementation) ---------------------------------
# Kept under their historical private names for existing callers.
_shingles = shingles
_signature = signature
_bands_for = bands_for
_jaccard = jaccard


def assert_no_near_duplicate_across_split(
    train: Sequence[Record],
    eval: Sequence[Record],
    threshold: float = 0.9,
    raise_on_match: bool = True,
) -> list[NearDuplicatePair]:
    """MinHash ou rapidfuzz sur les textes normalisés. Retourne les paires
    suspectes au-dessus du seuil. Un groupe mal formé réintroduit la fuite
    par la porte de derrière : ce contrôle est le filet de sécurité.

    MinHash LSH proposes candidates; each candidate is then verified with
    the exact Jaccard similarity of character 5-gram shingles, so the result
    has no false positives. Returns (train_id, eval_id, jaccard) pairs with
    jaccard >= threshold, sorted. Raises LeakageError when any pair is found
    unless ``raise_on_match`` is False.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    bands, rows = _bands_for(threshold)

    train_shingles: dict[str, frozenset[str]] = {}
    buckets: dict[tuple[int, bytes], list[str]] = {}
    for rec in train:
        sh = _shingles(rec.text)
        if not sh:
            continue
        train_shingles[rec.id] = sh
        sig = _signature(sh)
        for b in range(bands):
            key = (b, sig[b * rows : (b + 1) * rows].tobytes())
            buckets.setdefault(key, []).append(rec.id)

    pairs: list[NearDuplicatePair] = []
    for rec in eval:
        sh = _shingles(rec.text)
        if not sh:
            continue
        sig = _signature(sh)
        candidates: set[str] = set()
        for b in range(bands):
            candidates.update(buckets.get((b, sig[b * rows : (b + 1) * rows].tobytes()), ()))
        for train_id in candidates:
            sim = _jaccard(train_shingles[train_id], sh)
            if sim >= threshold:
                pairs.append((train_id, rec.id, round(sim, 4)))

    pairs.sort()
    if pairs:
        logger.error("leakage_near_duplicates pairs=%d threshold=%.2f", len(pairs), threshold)
        if raise_on_match:
            ids = [f"{t}~{e}" for t, e, _ in pairs]
            raise LeakageError(
                f"{len(pairs)} near-duplicate pair(s) across train/eval "
                f"(jaccard >= {threshold}): {_preview(ids)}",
                pairs=pairs,
            )
    return pairs
