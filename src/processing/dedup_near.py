"""Near-duplicate detection (replaces the dedup_near stub).

MinHash + LSH over character 5-gram shingles of the normalised text
(src/processing/minhash.py). LSH buckets propose candidate pairs in roughly
linear time; each candidate is confirmed with the exact Jaccard similarity,
then clusters are closed transitively (union-find). Deterministic.

ORDERING DECISION: this runs AFTER anonymisation.
Two tickets that are identical except for the customer's name and IBAN are
the same training example: once both read "[PERSON] ... [IBAN]", their
similarity is 1.0 and they cluster. Before anonymisation, the differing PII
lowers the score, so a mailbox re-export or a copy-pasted template would
slip through and could even land on both sides of the train/eval split.
Running after anonymisation also means no PII ever enters the shingle sets.

COUNTER-ARGUMENT, acknowledged: masking can make two GENUINELY DISTINCT
cases look alike - two claims filed with the same template ("my car was hit
on [date], plate [..], please send an expert") are different customers and
different events. After masking they may cluster and one is dropped. We
accept that loss: for a fine-tuning set, two template-identical examples
teach the model the same thing, and keeping both mostly over-weights the
template. The threshold stays high (0.85 Jaccard on 5-gram shingles) so only
texts that are nearly word-for-word identical merge, and every dropped
member is counted ('near_duplicate') with its representative's id.

A cluster does not silently delete data: the GROUPS of all members are
merged (shared group_id), so whatever remains of them stays on one side of
the split; one representative is kept (the longest text, ties broken on
id) and the other members are dropped and counted.
"""

from dataclasses import dataclass

from src.contracts import Record
from src.processing.minhash import band_keys, bands_for, jaccard, shingles, signature

DEFAULT_THRESHOLD = 0.85


@dataclass(frozen=True)
class Cluster:
    member_ids: tuple[str, ...]  # sorted record ids
    representative_id: str  # kept: longest text, tie-break on id
    group_ids: tuple[str, ...]  # distinct groups touched (to be merged)
    sources: tuple[str, ...]  # distinct source names ("crm_postgres", "support_mailbox", ...)
    min_similarity: float  # weakest confirmed link inside the cluster


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Smallest id becomes the root: deterministic merged ids.
            low, high = sorted((ra, rb))
            self.parent[high] = low


def find_near_duplicates(records: list[Record], threshold: float = DEFAULT_THRESHOLD) -> list[Cluster]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    bands, rows = bands_for(threshold)
    by_id = {r.id: r for r in records}
    shingle_sets: dict[str, frozenset[str]] = {}
    buckets: dict[tuple[int, bytes], list[str]] = {}
    for rec in sorted(records, key=lambda r: r.id):
        sh = shingles(rec.text)
        if not sh:
            continue
        shingle_sets[rec.id] = sh
        for key in band_keys(signature(sh), bands, rows):
            buckets.setdefault(key, []).append(rec.id)

    uf = _UnionFind()
    weakest: dict[str, float] = {}
    checked: set[tuple[str, str]] = set()
    for ids in buckets.values():
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                similarity = jaccard(shingle_sets[a], shingle_sets[b])
                if similarity >= threshold:
                    uf.union(a, b)
                    weakest[a] = min(weakest.get(a, 1.0), similarity)
                    weakest[b] = min(weakest.get(b, 1.0), similarity)

    members: dict[str, list[str]] = {}
    for rid in weakest:
        members.setdefault(uf.find(rid), []).append(rid)

    clusters: list[Cluster] = []
    for ids in members.values():
        ids.sort()
        representative = min(ids, key=lambda rid: (-len(by_id[rid].text), rid))
        clusters.append(
            Cluster(
                member_ids=tuple(ids),
                representative_id=representative,
                group_ids=tuple(sorted({by_id[rid].group_id for rid in ids})),
                sources=tuple(sorted({rid.split(":", 1)[0] for rid in ids})),
                min_similarity=round(min(weakest[rid] for rid in ids), 4),
            )
        )
    return sorted(clusters, key=lambda c: c.member_ids)


def merge_groups(records: list[Record], clusters: list[Cluster]) -> dict[str, str]:
    """group_id -> merged group_id for every group touched by a cluster."""
    uf = _UnionFind()
    for cluster in clusters:
        first, *rest = cluster.group_ids
        for other in rest:
            uf.union(first, other)
    return {g: uf.find(g) for g in uf.parent}
