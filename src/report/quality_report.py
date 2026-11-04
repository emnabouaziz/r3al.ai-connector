"""Quality report: out/quality_report.json and out/quality_report.md.

Numbers and ids only, never content. Output is fully deterministic (sorted
keys, no timestamps) so reruns are byte-identical.
"""

import json
from collections import Counter as Tally
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.contracts import Record
from src.counters import Counters
from src.dataset.splitter import SplitConfig, SplitResult
from src.fileio import write_text_atomic

JSON_NAME = "quality_report.json"
MD_NAME = "quality_report.md"


def build_report(
    *,
    counters: Counters,
    split_result: SplitResult,
    split_config: SplitConfig,
    tagged: list[Record],
    kept: int,
    n_train_examples: int,
    n_eval_examples: int,
    domains: list[str],
    languages: list[str],
    stub_stages: list[str],
) -> dict[str, Any]:
    tally: Tally[tuple[str, str]] = Tally((str(r.domain), str(r.language)) for r in tagged)
    matrix = {d: {lang: tally[(d, lang)] for lang in languages} for d in domains}
    unknown_language = {d: tally[(d, "unknown")] for d in domains}

    per_cell: dict[str, dict[str, Any]] = {d: {} for d in domains}
    per_domain: dict[str, dict[str, Any]] = {}
    for (d, lang), stats in split_result.per_cell.items():
        per_cell.setdefault(d, {})[lang] = dict(stats)
    for d in domains:
        cells = per_cell.get(d, {}).values()
        per_domain[d] = {
            "train": sum(c["train"] for c in cells),
            "eval": sum(c["eval"] for c in cells),
            "eval_groups": sum(c["eval_groups"] for c in cells),
            "capped": split_result.capped_domains.get(d, 0),
            "measurable": d not in split_result.unmeasurable_domains,
        }

    ingested = counters.ingested_total
    dropped = counters.dropped_total
    split_cfg = asdict(split_config)
    split_cfg["domains"] = list(split_config.domains)
    split_cfg["languages"] = list(split_config.languages)
    return {
        "unmeasurable_domains": list(split_result.unmeasurable_domains),
        "unmeasurable_cells": [f"{d}/{lang}" for d, lang in split_result.unmeasurable_cells],
        "invariant": {
            "ingested": ingested,
            "kept": kept,
            "dropped": dropped,
            "holds": ingested == kept + dropped,
        },
        "summary": {"train_examples": n_train_examples, "eval_examples": n_eval_examples},
        "ingested_per_source": dict(counters.ingested_per_source),
        "dropped": {s: dict(r) for s, r in counters.dropped.items()},
        "duplicates": {
            "exact": counters.duplicates_exact,
            "near": counters.duplicates_near,
            "superseded": counters.superseded,
        },
        "pii_replaced": dict(counters.pii_replaced),
        "domain_language_matrix": matrix,
        "unknown_language_per_domain": unknown_language,
        "split": {
            "config": split_cfg,
            "per_domain": per_domain,
            "per_cell": per_cell,
            "capped_domains": dict(split_result.capped_domains),
        },
        "stub_stages": list(stub_stages),
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Data quality report", ""]

    unmeasurable = report["unmeasurable_domains"]
    if unmeasurable:
        lines += [
            "> **WARNING - unmeasurable domains:** " + ", ".join(unmeasurable),
            "> Their eval floor is not met; per-domain metrics for them are not reliable.",
            "> Cells below floor: " + (", ".join(report["unmeasurable_cells"]) or "none"),
        ]
    else:
        lines.append("> All domains measurable: every eval floor is met.")
    if report["stub_stages"]:
        lines += ["", "> Stub stages (not yet implemented): " + ", ".join(report["stub_stages"])]

    inv = report["invariant"]
    lines += [
        "",
        "## Invariant",
        "",
        f"ingested ({inv['ingested']}) == kept ({inv['kept']}) + dropped ({inv['dropped']}): "
        f"**{'OK' if inv['holds'] else 'FAILED'}**",
        "",
        "## Ingestion",
        "",
        _table(["source", "records"], [[s, n] for s, n in sorted(report["ingested_per_source"].items())]),
        "",
        "## Drops",
        "",
    ]
    drop_rows = [[s, r, n] for s, reasons in sorted(report["dropped"].items())
                 for r, n in sorted(reasons.items())]
    lines.append(_table(["stage", "reason", "count"], drop_rows) if drop_rows else "No drops.")

    dup = report["duplicates"]
    lines += [
        "",
        "## Duplicates",
        "",
        _table(["exact", "near", "superseded"], [[dup["exact"], dup["near"], dup["superseded"]]]),
        "",
        "## PII replaced",
        "",
        _table(["type", "count"], [[t, n] for t, n in sorted(report["pii_replaced"].items())])
        if report["pii_replaced"] else "None.",
        "",
        "## Domain x language (tagged records, before split)",
        "",
    ]
    matrix: dict[str, dict[str, int]] = report["domain_language_matrix"]
    langs = list(next(iter(matrix.values())).keys()) if matrix else []
    unknown = report["unknown_language_per_domain"]
    lines.append(_table(
        ["domain", *langs, "unknown", "total"],
        [[d, *(row[lang] for lang in langs), unknown.get(d, 0), sum(row.values()) + unknown.get(d, 0)]
         for d, row in matrix.items()],
    ))

    split = report["split"]
    cfg = split["config"]
    lines += [
        "",
        "## Split per domain",
        "",
        f"eval_ratio={cfg['eval_ratio']}, eval_floor_per_domain={cfg['eval_floor_per_domain']} "
        f"groups, eval_floor_per_cell={cfg['eval_floor_per_cell']} groups, "
        f"max_share_per_domain={cfg['max_share_per_domain']} (train only)",
        "",
        _table(
            ["domain", "train", "eval", "eval groups", "capped", "measurable"],
            [[d, s["train"], s["eval"], s["eval_groups"], s["capped"], "yes" if s["measurable"] else "**NO**"]
             for d, s in split["per_domain"].items()],
        ),
        "",
        "## Split per cell",
        "",
        _table(
            ["domain", "language", "train", "eval", "eval groups", "floor", "floor met"],
            [[d, lang, c["train"], c["eval"], c["eval_groups"], c["floor"],
              "yes" if c["floor_met"] else "**NO**"]
             for d, cells in split["per_cell"].items() for lang, c in cells.items()],
        ),
        "",
        "## Capped domains (train only)",
        "",
        _table(["domain", "records removed"], [[d, n] for d, n in split["capped_domains"].items()])
        if split["capped_domains"] else "None.",
        "",
    ]
    return "\n".join(lines)


def write_quality_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    json_path, md_path = out_dir / JSON_NAME, out_dir / MD_NAME
    write_text_atomic(json_path, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_text_atomic(md_path, render_markdown(report))
    return json_path, md_path
