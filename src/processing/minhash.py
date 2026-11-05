"""MinHash + LSH over character 5-gram shingles of normalised text.

Shared by near-duplicate detection (dedup_near) and the cross-split leakage
check. Deterministic: permutations derive from sha256, never from a random
generator or the builtin hash(). LSH proposes candidates in roughly linear
time; every candidate is then confirmed with the EXACT Jaccard similarity,
so results have no false positives.
"""

import hashlib
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from src.processing.domain import normalize_tokens

SHINGLE_SIZE = 5  # character n-grams over normalised text
NUM_PERM = 128
_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


def _stable_u64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


_PERM_A = np.array(
    [_stable_u64(f"a{i}") % (int(_MERSENNE_PRIME) - 1) + 1 for i in range(NUM_PERM)],
    dtype=np.uint64,
)
_PERM_B = np.array(
    [_stable_u64(f"b{i}") % int(_MERSENNE_PRIME) for i in range(NUM_PERM)], dtype=np.uint64
)


def shingles(text: str) -> frozenset[str]:
    norm = " ".join(normalize_tokens(text))
    if not norm:
        return frozenset()
    if len(norm) <= SHINGLE_SIZE:
        return frozenset({norm})
    return frozenset(norm[i : i + SHINGLE_SIZE] for i in range(len(norm) - SHINGLE_SIZE + 1))


def signature(shingle_set: frozenset[str]) -> npt.NDArray[np.uint64]:
    hv = np.array(
        [
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little")
            for s in sorted(shingle_set)
        ],
        dtype=np.uint64,
    )
    # uint64 wrap-around on the product is intended (same scheme as datasketch).
    with np.errstate(over="ignore"):
        phv = ((hv[:, None] * _PERM_A + _PERM_B) % _MERSENNE_PRIME) & _MAX_HASH
    sig: npt.NDArray[np.uint64] = phv.min(axis=0)
    return sig


def bands_for(threshold: float) -> tuple[int, int]:
    """Pick (bands, rows) so the LSH S-curve threshold sits below ``threshold``."""
    for bands, rows in ((16, 8), (32, 4), (64, 2)):
        if (1 / bands) ** (1 / rows) <= threshold - 0.1:
            return bands, rows
    return 128, 1


def band_keys(sig: npt.NDArray[np.uint64], bands: int, rows: int) -> Iterable[tuple[int, bytes]]:
    for b in range(bands):
        yield b, sig[b * rows : (b + 1) * rows].tobytes()


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0
