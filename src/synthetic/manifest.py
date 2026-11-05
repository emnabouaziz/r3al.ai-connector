"""Shared ground-truth PII manifest (data/generated/pii_manifest.json).

Several generators (CRM rows, documents) inject PII into free text. Each one
owns the entries tagged with its ``origin`` and replaces only those, so
re-running one generator never erases the other's ground truth.
THIS FILE IS A PII INDEX: data/generated/ is git-ignored.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.report.ground_truth_scan import normalise

LEGACY_ORIGIN = "crm"  # entries written before origins existed


def update_manifest(
    path: str | Path, origin: str, seed: int, values: dict[tuple[str, str], set[str]]
) -> int:
    """Replace ``origin``'s entries. values: (type, exact value) -> source uris.

    Returns the number of entries written for ``origin``.
    """
    target = Path(path)
    existing: dict[str, Any] = {}
    if target.is_file():
        existing = json.loads(target.read_text(encoding="utf-8"))
    kept = [v for v in existing.get("values", []) if v.get("origin", LEGACY_ORIGIN) != origin]
    new = [
        {"type": t, "value": v, "normalised": normalise(v), "sources": sorted(srcs), "origin": origin}
        for (t, v), srcs in sorted(values.items())
    ]
    seeds = dict(existing.get("seeds", {}))
    seeds[origin] = seed
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "seeds": dict(sorted(seeds.items())),
        "values": sorted(kept + new, key=lambda e: (e.get("origin", LEGACY_ORIGIN), e["type"], e["value"])),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(new)
