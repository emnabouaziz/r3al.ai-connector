"""Blocking anti-leakage checks between train and eval.

Both checks raise LeakageError: they are meant to stop the pipeline, not
to emit warnings. Error messages carry ids and counts only, never content.
"""

import hashlib
import logging
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.contracts import Record
from src.processing.domain import normalize_tokens

logger = logging.getLogger(__name__)

SHINGLE_SIZE = 5  # character n-grams over normalised text
NUM_PERM = 128
_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
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


# --- MinHash / LSH -----------------------------------------------------------


def _stable_u64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


_PERM_A = np.array(
    [_stable_u64(f"a{i}") % (int(_MERSENNE_PRIME) - 1) + 1 for i in range(NUM_PERM)],
    dtype=np.uint64,
)
_PERM_B = np.array(
    [_stable_u64(f"b{i}") % int(_MERSENNE_PRIME) for i in range(NUM_PERM)], dtype=np.uint64
)


def _shingles(text: str) -> frozenset[str]:
    norm = " ".join(normalize_tokens(text))
    if not norm:
        return frozenset()
    if len(norm) <= SHINGLE_SIZE:
        return frozenset({norm})
    return frozenset(norm[i : i + SHINGLE_SIZE] for i in range(len(norm) - SHINGLE_SIZE + 1))


def _signature(shingles: frozenset[str]) -> npt.NDArray[np.uint64]:
    hv = np.array(
        [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
            for s in sorted(shingles)
        ],
        dtype=np.uint64,
    )
    # uint64 wrap-around on the product is intended (same scheme as datasketch).
    with np.errstate(over="ignore"):
        phv = ((hv[:, None] * _PERM_A + _PERM_B) % _MERSENNE_PRIME) & _MAX_HASH
    sig: npt.NDArray[np.uint64] = phv.min(axis=0)
    return sig


def _bands_for(threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) so the LSH S-curve threshold sits below ``threshold``."""
    for bands, rows in ((16, 8), (32, 4), (64, 2)):
        if (1 / bands) ** (1 / rows) <= threshold - 0.1:
            return bands, rows
    return 128, 1


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


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
