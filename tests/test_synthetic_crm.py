"""The synthetic CRM must be deterministic, classifiable and leakage-safe."""

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from src.config import load_taxonomy
from src.dataset.leakage import _jaccard, _shingles
from src.processing.domain import classify_domain
from src.synthetic.crm import POOL_DOMAINS, CrmDataset, generate_crm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data() -> CrmDataset:
    return generate_crm(seed=42)


def _texts(ds: CrmDataset) -> dict[str, str]:
    out: dict[str, str] = {f"claims:{c.claim_id}": c.description for c in ds.claims}
    out |= {f"claim_notes:{n.note_id}": n.body for n in ds.claim_notes}
    out |= {f"policies:{p.policy_id}": p.coverage_summary for p in ds.policies if p.coverage_summary}
    out |= {f"invoices:{i.invoice_id}": i.customer_note for i in ds.invoices if i.customer_note}
    return out


def _group(ds: CrmDataset, key: str) -> str:
    table, _, pk = key.partition(":")
    if table == "claim_notes":
        return f"claim:{next(n.claim_id for n in ds.claim_notes if n.note_id == int(pk))}"
    prefix = {"claims": "claim", "policies": "policy", "invoices": "invoice"}[table]
    return f"{prefix}:{pk}"


def test_deterministic(data: CrmDataset) -> None:
    again = generate_crm(seed=42)
    assert again == data
    assert generate_crm(seed=7) != data


def test_volumes_and_languages(data: CrmDataset) -> None:
    assert len(data.customers) == 200 and len(data.claims) == 150
    assert len(_texts(data)) == len(data.text_origin) == 600
    langs = Counter(lang for _, lang in data.text_origin.values())
    assert set(langs) == {"nl", "fr", "en"}


def test_every_text_is_classified_into_its_pool_domain(data: CrmDataset) -> None:
    taxonomy: dict[str, Any] = load_taxonomy(str(ROOT / "src" / "taxonomy.yaml"))
    texts = _texts(data)
    wrong = [
        key for key, (pool, _) in data.text_origin.items()
        if classify_domain(texts[key], taxonomy).domain != POOL_DOMAINS[pool]
    ]
    assert len(wrong) <= 0.01 * len(texts), wrong[:10]


def test_no_near_duplicates_across_groups(data: CrmDataset) -> None:
    texts = _texts(data)
    shingles = {k: _shingles(t) for k, t in texts.items()}
    by_pool: dict[tuple[str, str], list[str]] = {}
    for key, origin in data.text_origin.items():
        by_pool.setdefault(origin, []).append(key)
    worst = 0.0
    for keys in by_pool.values():
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                if _group(data, a) != _group(data, b):
                    worst = max(worst, _jaccard(shingles[a], shingles[b]))
    assert worst < 0.9, worst
