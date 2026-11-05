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
from src.dataset.splitter import STATUS_MEASURABLE, SplitConfig, SplitResult
from src.fileio import write_text_atomic

JSON_NAME = "quality_report.json"
MD_NAME = "quality_report.md"

CAP_RATIONALE = (
    "Capping is a sampling decision, not a quality rejection: valid records are removed "
    "from TRAIN only so that no domain exceeds max_share_per_domain of the training set; "
    "eval is never capped."
)


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
    needs_review: list[dict[str, Any]] | None = None,
    near_duplicate_clusters: int = 0,
    build_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tally: Tally[tuple[str, str]] = Tally((str(r.domain), str(r.language)) for r in tagged)
    matrix = {d: {lang: tally[(d, lang)] for lang in languages} for d in domains}
    unknown_language = {d: tally[(d, "unknown")] for d in domains}

    per_cell: dict[str, dict[str, Any]] = {d: {} for d in domains}
    for (d, lang), stats in split_result.per_cell.items():
        per_cell.setdefault(d, {})[lang] = dict(stats)
    per_domain = {d: dict(split_result.per_domain[d]) for d in domains if d in split_result.per_domain}

    ingested = counters.ingested_total
    dropped = counters.dropped_total
    capped = counters.capped_total
    split_cfg = asdict(split_config)
    split_cfg["domains"] = list(split_config.domains)
    split_cfg["languages"] = list(split_config.languages)
    return {
        "domain_status": {d: split_result.domain_status.get(d, "ABSENT") for d in domains},
        "invariant": {
            "ingested": ingested,
            "kept": kept,
            "dropped": dropped,
            "capped": capped,
            "holds": ingested == kept + dropped + capped,
        },
        "summary": {
            "train_examples": n_train_examples,
            "eval_examples": n_eval_examples,
            "eval_share": round(n_eval_examples / kept, 4) if kept else 0.0,
        },
        "ingested_per_source": dict(counters.ingested_per_source),
        "dropped": {
            "total": dropped,
            "per_stage": {s: dict(r) for s, r in counters.dropped.items()},
            "per_source": {s: dict(r) for s, r in sorted(counters.dropped_per_source.items())},
        },
        "capped": {"total": capped, "per_domain": dict(counters.capped), "rationale": CAP_RATIONALE},
        "duplicates": {
            "exact": counters.duplicates_exact,
            "near": counters.duplicates_near,
            "near_clusters": near_duplicate_clusters,
            "superseded": counters.superseded,
        },
        "cleaning": dict(sorted(counters.cleaned.items())),
        "build": _build_section_data(counters, build_summary or {}),
        "needs_review": sorted(
            needs_review or [], key=lambda item: (item.get("kind", ""), item.get("items", []))
        ),
        "pii_replaced": dict(counters.pii_replaced),
        "pii_candidates_rejected": dict(counters.pii_candidates_rejected),
        "domain_language_matrix": matrix,
        "unknown_language_per_domain": unknown_language,
        "split": {
            "config": split_cfg,
            "floor_scaling_applied": split_result.floor_scaling_applied,
            "effective_eval_floor_per_domain": split_result.effective_eval_floor_per_domain,
            "effective_eval_floor_per_cell": split_result.effective_eval_floor_per_cell,
            "eval_budget": split_result.eval_budget,
            "per_domain": per_domain,
            "per_cell": per_cell,
        },
        "stub_stages": list(stub_stages),
        # Filled by run.py after the ground-truth scan; explicit until then.
        "anonymisation_recall": {
            "status": "not_computed",
            "note": "Recall is measured by run.py after the outputs are written.",
        },
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _pii_table(replaced: dict[str, int], rejected: dict[str, int]) -> str:
    types = sorted(set(replaced) | set(rejected))
    if not types:
        return "None."
    rows: list[list[Any]] = []
    for t in types:
        # PERSON comes from NER only: there is no validator, hence no rejection count.
        rows.append([t, replaced.get(t, 0), "n/a (NER)" if t == "PERSON" else rejected.get(t, 0)])
    return _table(["type", "replaced", "candidates rejected"], rows)


def _recall_section(recall: dict[str, Any]) -> list[str]:
    lines = ["## Anonymisation recall (ground truth)", ""]
    if recall.get("status") != "measured":
        return lines + [f"**Not measured** ({recall.get('status')}): {recall.get('note', '')}", ""]
    lines += [
        "Literal search of every PII value injected by the synthetic generator, independent "
        "of the anonymiser's patterns (no regex). Leaked = still present in the output.",
        "",
        _table(
            ["type", "injected", "leaked", "recall"],
            [[t, n, recall["found_in_output"].get(t, 0), f"{recall['recall'][t]:.2%}"]
             for t, n in recall["total_injected"].items()],
        ),
        "",
        f"Overall recall: **{recall['overall_recall']:.2%}**",
        "",
    ]
    if recall["misses"]:
        lines += [
            "Misses (value hash only):",
            "",
            _table(["type", "file", "line", "value hash"],
                   [[m["type"], m["file"], m["line"], m["value_hash"]] for m in recall["misses"]]),
            "",
        ]
    return lines


def _build_section_data(counters: Counters, summary: dict[str, Any]) -> dict[str, Any]:
    observed = counters.dropped.get("build", {})
    reasons = sorted({*summary.get("reasons", []), *observed})
    per_kind = summary.get("per_kind", {})
    candidates = sum(int(k.get("candidates", 0)) for k in per_kind.values())
    accepted = sum(int(k.get("accepted", 0)) for k in per_kind.values())
    per_builder: dict[str, Any] = {}
    for name, stats in summary.get("per_builder", {}).items():
        cand, acc = int(stats.get("candidates", 0)), int(stats.get("accepted", 0))
        per_builder[name] = {
            "records": int(stats.get("records", 0)),
            "candidates": cand,
            "accepted": acc,
            "acceptance_rate": round(acc / cand, 4) if cand else 0.0,
            "rejections": dict(stats.get("rejections", {})),
        }
    return {
        "candidates": candidates,
        "accepted": accepted,
        "acceptance_rate": round(accepted / candidates, 4) if candidates else 0.0,
        "duplicate_examples": int(summary.get("duplicate_examples", 0)),
        "per_kind": per_kind,
        "per_builder": per_builder,
        "rejections": {r: int(observed.get(r, 0)) for r in reasons},
        "system_prompt_language": summary.get("system_prompt_language", ""),
    }


BUILDER_DESCRIPTIONS = {
    "tickets": "conversations -> extracted customer/agent pairs",
    "documents": "reference knowledge -> questions grounded in a section",
    "facts": "CRM case facts -> extracted pairs + templated answers grounded in the fields",
}


def _build_section(build: dict[str, Any]) -> list[str]:
    lines = [
        "## Example construction",
        "",
        f"**{build['candidates']} candidates -> {build['accepted']} accepted "
        f"({build['acceptance_rate']:.1%})**, then {build['duplicate_examples']} duplicate examples removed. "
        "Rejecting is the point: a rejected candidate would teach the model something wrong.",
        "",
    ]
    per_builder = build.get("per_builder", {})
    silent = [name for name, b in per_builder.items() if b["records"] and not b["accepted"]]
    if silent:
        lines += [
            "> **WARNING - builder(s) with a 0% acceptance rate: " + ", ".join(silent) + ".** Their source "
            "reaches the build stage but yields no example at all.",
            "",
        ]
    if per_builder:
        lines += [
            _table(
                ["builder", "what it builds", "records", "candidates", "accepted", "acceptance rate"],
                [[name, BUILDER_DESCRIPTIONS.get(name, ""), b["records"], b["candidates"], b["accepted"],
                  f"**{b['acceptance_rate']:.1%}**" if b["records"] and not b["accepted"] else
                  f"{b['acceptance_rate']:.1%}"]
                 for name, b in per_builder.items()],
            ),
            "",
        ]
        for name, b in per_builder.items():
            lines += [
                f"Rejections - {name} builder (every reason it can report, zeros included):",
                "",
                _table(["reason", "count"],
                       [[r, n] for r, n in sorted(b["rejections"].items(), key=lambda x: (-x[1], x[0]))]),
                "",
            ]
    else:
        lines += [
            _table(["reason", "count"],
                   [[r, n] for r, n in sorted(build["rejections"].items(), key=lambda x: (-x[1], x[0]))]),
            "",
        ]
    lines += [f"System prompt language: {build['system_prompt_language'] or 'n/a'}.", ""]
    return lines


def _review_section(items: list[dict[str, Any]]) -> list[str]:
    lines = ["## Needs client review", ""]
    if not items:
        return lines + ["Nothing escalated: the pipeline did not meet a decision it refuses to take.", ""]
    lines += [
        "The pipeline deliberately does NOT decide these items; they are escalated to whoever "
        "owns the data.",
        "",
    ]
    for item in items:
        lines.append(f"- **{item.get('kind', 'item')}**: {', '.join(f'`{x}`' for x in item.get('items', []))}")
        if item.get("detail"):
            lines.append(f"  {item['detail']}")
    return lines + [""]


def _status_cell(status: str) -> str:
    return status if status == STATUS_MEASURABLE else f"**{status}**"


def render_markdown(report: dict[str, Any]) -> str:
    split = report["split"]
    cfg = split["config"]
    lines: list[str] = ["# Data quality report", ""]

    scaling = split["floor_scaling_applied"]
    if scaling is not None:
        lines += [
            f"> **WARNING - eval floors scaled down by a factor of {scaling}** to respect "
            f"eval_ratio_max={cfg['eval_ratio_max']}. Effective floors: "
            f"{split['effective_eval_floor_per_domain']} examples per domain, "
            f"{split['effective_eval_floor_per_cell']} per cell "
            f"(configured: {cfg['eval_floor_per_domain']} / {cfg['eval_floor_per_cell']}).",
            "",
        ]
    not_measurable = {d: s for d, s in report["domain_status"].items() if s != STATUS_MEASURABLE}
    if not_measurable:
        lines.append(
            "> **WARNING - domains not fully measurable:** "
            + ", ".join(f"{d} ({s})" for d, s in not_measurable.items())
        )
    else:
        lines.append("> All domains MEASURABLE: every eval floor is met.")
    if report["stub_stages"]:
        lines += ["", "> Stub stages (not yet implemented): " + ", ".join(report["stub_stages"])]
    lines += ["", *_review_section(report["needs_review"]), *_build_section(report["build"])]

    inv = report["invariant"]
    summary = report["summary"]
    lines += [
        "",
        "## Invariant",
        "",
        f"ingested ({inv['ingested']}) == kept ({inv['kept']}) + dropped ({inv['dropped']}) "
        f"+ capped ({inv['capped']}): **{'OK' if inv['holds'] else 'FAILED'}**",
        "",
        f"kept = train ({summary['train_examples']}) + eval ({summary['eval_examples']}); "
        f"eval share = {summary['eval_share']:.1%} (ceiling {cfg['eval_ratio_max']:.0%})",
        "",
        "## Ingestion",
        "",
        _table(["source", "records"], [[s, n] for s, n in sorted(report["ingested_per_source"].items())]),
        "",
        f"## Dropped (quality) - total {report['dropped']['total']}",
        "",
    ]
    drop_rows = [[s, r, n] for s, reasons in sorted(report["dropped"]["per_stage"].items())
                 for r, n in sorted(reasons.items())]
    lines.append(_table(["stage", "reason", "count"], drop_rows) if drop_rows else "No drops.")
    source_rows = [
        [src, report["ingested_per_source"].get(src, 0), reason, n]
        for src, reasons in sorted(report["dropped"]["per_source"].items())
        for reason, n in sorted(reasons.items())
    ]
    if source_rows:
        lines += ["", "Per source:", "", _table(["source", "ingested", "reason", "count"], source_rows)]

    capped = report["capped"]
    lines += ["", f"## Capped (sampling decision) - total {capped['total']}", "", capped["rationale"], ""]
    lines.append(
        _table(["domain", "examples removed from train"], [[d, n] for d, n in sorted(capped["per_domain"].items())])
        if capped["per_domain"] else "Nothing capped."
    )

    dup = report["duplicates"]
    lines += [
        "",
        "## Duplicates",
        "",
        _table(["exact", "near", "near clusters", "superseded"],
               [[dup["exact"], dup["near"], dup["near_clusters"], dup["superseded"]]]),
        "",
        "near = records dropped because they are near-identical (MinHash + exact Jaccard >= the "
        "configured threshold, computed after anonymisation) to another record; one representative "
        "per cluster is kept and the groups of all members are merged, so near-duplicates never "
        "straddle the train/eval split. Typical cause: the same e-mail thread exported both by the "
        "ticket system (CSV) and by the mailbox (.eml).",
        "",
        "## Cleaning",
        "",
        "Messages affected by each cleaning operation (removed text never reaches the dataset):",
        "",
        _table(["operation", "messages"], [[op, n] for op, n in report["cleaning"].items()])
        if report["cleaning"] else "Nothing removed.",
        "",
        "## PII",
        "",
        "replaced = masked with a placeholder; candidates rejected = regex matches that failed "
        "their validator and were left untouched (pattern precision).",
        "",
        _pii_table(report["pii_replaced"], report["pii_candidates_rejected"]),
        "",
        *_recall_section(report["anonymisation_recall"]),
        "## Domain x language (examples, before split)",
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

    lines += [
        "",
        "## Split per domain",
        "",
        f"All values in examples. min_train_per_domain={cfg['min_train_per_domain']}, "
        f"eval floor per domain={split['effective_eval_floor_per_domain']}, "
        f"per cell={split['effective_eval_floor_per_cell']}, eval_ratio={cfg['eval_ratio']}, "
        f"eval_ratio_max={cfg['eval_ratio_max']}, max_share_per_domain={cfg['max_share_per_domain']} (train only).",
        "",
        "Statuses: ABSENT (no records), INSUFFICIENT_DATA (fewer than min_train examples), "
        "PARTIALLY_MEASURABLE (eval floor not reached), MEASURABLE (eval floor reached).",
        "",
        _table(
            ["domain", "status", "total", "train", "eval", "eval floor", "capped"],
            [[d, _status_cell(report["domain_status"][d]), s["total"], s["train"], s["eval"],
              s["eval_floor"], s["capped"]]
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
    ]
    return "\n".join(lines)


def write_quality_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    json_path, md_path = out_dir / JSON_NAME, out_dir / MD_NAME
    write_text_atomic(json_path, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    write_text_atomic(md_path, render_markdown(report))
    return json_path, md_path
