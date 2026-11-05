"""Anonymisation recall against ground truth: literal search, NO REGEX.

The pattern-based scan (pii_scan.py) reuses the anonymiser's own patterns,
so it cannot see PII those patterns do not cover: its "0 findings" is
circular. This scan is independent of every pattern. The synthetic
generator writes a manifest of every PII value it injected into free text;
here each output line is normalised (casefold, keep letters and digits
only) and every manifest value is searched as a literal substring.
A hit means the anonymiser missed that value.

Normalising away all separators makes the search formatting-independent:
"BE68 5390 0754 7034", "BE68539007547034" and a value broken across a line
break or across the user/assistant boundary all match. Strings of one
output line are concatenated without separator for the same reason.

Misses carry a sha256 prefix of the normalised value, never the value, so
the quality report stays safe to commit.
"""

import hashlib
import json
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HASH_PREFIX_LEN = 12


def normalise(text: str) -> str:
    """Casefold and keep letters and digits only (no regex)."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def value_hash(normalised_value: str) -> str:
    return hashlib.sha256(normalised_value.encode("utf-8")).hexdigest()[:HASH_PREFIX_LEN]


@dataclass
class RecallResult:
    total_injected: dict[str, int]  # per type: distinct values that entered the pipeline
    found_in_output: dict[str, int]  # per type: distinct values LEAKED into the output
    recall: dict[str, float]  # 1 - found / injected, per type
    overall_recall: float
    misses: list[dict[str, Any]] = field(default_factory=list)  # {type, file, line, value_hash}

    @property
    def leaked(self) -> int:
        return sum(self.found_in_output.values())

    def as_report(self) -> dict[str, Any]:
        return {
            "status": "measured",
            "total_injected": dict(sorted(self.total_injected.items())),
            "found_in_output": dict(sorted(self.found_in_output.items())),
            "recall": dict(sorted(self.recall.items())),
            "overall_recall": self.overall_recall,
            "misses": self.misses,
        }


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _line_texts(line: str) -> list[str]:
    """Normalised texts of one output line.

    Message contents are joined with no separator, so a value the builder
    cut between the user and assistant turns is still found. Role names and
    other fields are searched separately: interleaving "user"/"assistant"
    between the contents would break exactly that case.
    """
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return [normalise(line)]
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        contents = [m.get("content", "") for m in obj["messages"] if isinstance(m, dict)]
        rest = {k: v for k, v in obj.items() if k != "messages"}
        return [normalise("".join(c for c in contents if isinstance(c, str))), normalise(" ".join(_strings(rest)))]
    return [normalise("".join(_strings(obj)))]


def scan_against_manifest(
    output_paths: list[str],
    manifest_path: str,
    ingested_sources: Collection[str] | None = None,
) -> RecallResult:
    """For each output line, normalise the text the same way as the
    manifest (lowercase, strip separators and whitespace) and search for
    each manifest value as a literal substring. NO REGEX ANYWHERE.
    A hit means the anonymiser missed that value.
    misses record a sha256 prefix of the value, NEVER the value itself,
    so the report can be committed.

    ``ingested_sources``: source uris ("table:pk") that entered the
    pipeline. A manifest value counts only if one of its sources is among
    them, so recall is measured over what was actually ingested. None means
    every manifest value counts.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    allowed = set(ingested_sources) if ingested_sources is not None else None

    candidates: dict[tuple[str, str], str] = {}  # (type, normalised) -> hash
    for entry in manifest["values"]:
        sources = entry.get("sources", [])
        if allowed is not None and not any(s in allowed for s in sources):
            continue  # never entered the pipeline: cannot be a miss
        norm = normalise(entry["value"])
        if norm:
            candidates[(entry["type"], norm)] = value_hash(norm)

    leaked: set[tuple[str, str]] = set()
    misses: list[dict[str, Any]] = []
    for path in output_paths:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                texts = _line_texts(line)
                for (pii_type, norm), digest in candidates.items():
                    if any(norm in text for text in texts):
                        leaked.add((pii_type, norm))
                        misses.append(
                            {"type": pii_type, "file": Path(path).name, "line": lineno, "value_hash": digest}
                        )

    injected: dict[str, int] = {}
    for pii_type, _ in candidates:
        injected[pii_type] = injected.get(pii_type, 0) + 1
    found = {t: sum(1 for lt, _ in leaked if lt == t) for t in injected}
    recall = {t: round(1.0 - found[t] / n, 6) for t, n in injected.items()}
    total = sum(injected.values())
    overall = round(1.0 - len(leaked) / total, 6) if total else 1.0
    misses.sort(key=lambda m: (m["file"], m["line"], m["type"], m["value_hash"]))
    return RecallResult(injected, found, recall, overall, misses)


def no_ground_truth_report(reason: str) -> dict[str, Any]:
    return {"status": "no_ground_truth", "note": reason}
