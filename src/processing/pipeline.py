"""End-to-end pipeline.

ingest -> parse -> clean -> dedup_exact -> anonymize -> dedup_near -> tag
-> split -> build -> verify -> report

Error policy:
  - an exception on ONE item never fails the run: it is dropped with a
    reason code and counted;
  - an exception on the RUN (invalid config, unreachable source, detected
    leakage, broken invariant) fails it.
Logs carry ids, counts and exception class names only, never content.
"""

import hashlib
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import src.connectors.builtin  # noqa: F401  (registers every connector)
from src.connectors.base import SourceConnector
from src.config import ConfigError, load_taxonomy
from src.contracts import DropReason, ItemRef, RawDocument, Record
from src.counters import Counters
from src.dataset.builder import record_to_example, write_jsonl
from src.dataset.leakage import assert_no_group_overlap, assert_no_near_duplicate_across_split
from src.dataset.splitter import SplitConfig, SplitResult, split
from src.processing.domain import classify_domain
from src.processing.language import detect_language
from src.registry import build as build_connector
from src.report.quality_report import JSON_NAME, MD_NAME, build_report, write_quality_report

logger = logging.getLogger(__name__)

STUB_STAGES = ["anonymize", "dedup_near"]
TRAIN_NAME = "train.jsonl"
EVAL_NAME = "eval.jsonl"
OUTPUT_NAMES = (TRAIN_NAME, EVAL_NAME, JSON_NAME, MD_NAME)

T = TypeVar("T")
U = TypeVar("U")


class SourceUnavailable(RuntimeError):
    """A configured source failed its connection test."""


@dataclass
class PipelineResult:
    train: list[dict[str, Any]]
    eval: list[dict[str, Any]]
    split: SplitResult
    counters: Counters
    report: dict[str, Any]
    drops: list[DropReason] = field(default_factory=list)
    outputs: dict[str, Path] = field(default_factory=dict)


class _Run:
    """Mutable state of one run: counters + drop log."""

    def __init__(self) -> None:
        self.counters = Counters()
        self.drops: list[DropReason] = []

    def drop(self, record_id: str, stage: str, reason: str, detail: str = "") -> None:
        self.counters.drop(stage, reason)
        self.drops.append(DropReason(record_id, stage, reason, detail))

    def per_item(
        self, stage: str, items: Iterable[T], fn: Callable[[T], U | None], item_id: Callable[[T], str]
    ) -> list[U]:
        """Apply ``fn`` to each item; an exception drops that item only.

        ``fn`` returns None when it has already dropped the item itself.
        """
        out: list[U] = []
        for item in items:
            try:
                result = fn(item)
            except Exception as exc:  # noqa: BLE001 - per-item isolation is the contract
                rid = item_id(item)
                logger.warning("item_failed stage=%s id=%s error=%s", stage, rid, type(exc).__name__)
                self.drop(rid, stage, f"{stage}_error", type(exc).__name__)
                continue
            if result is not None:
                out.append(result)
        return out


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping")
    return value


def _require(section: dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"missing config key '{where}.{key}'")
    return section[key]


def _normalise_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _stage_log(stage: str, n_in: int, n_out: int, run: _Run, extra: str = "") -> None:
    logger.info(
        "stage=%s in=%d out=%d dropped=%d%s", stage, n_in, n_out, n_in - n_out,
        f" {extra}" if extra else "",
    )


def run_pipeline(config: dict[str, Any]) -> PipelineResult:
    # --- Config (run-level errors) -------------------------------------------
    seed = int(_require(config, "seed", "root"))
    domains = list(_require(config, "domains", "root"))
    languages = list(_require(config, "languages", "root"))
    sources = _require(config, "sources", "root")
    if not isinstance(sources, list):
        raise ConfigError("'sources' must be a list")
    disabled = [s.get("name", "?") for s in sources if isinstance(s, dict) and s.get("enabled") is False]
    sources = [s for s in sources if not (isinstance(s, dict) and s.get("enabled") is False)]
    if not sources:
        raise ConfigError("no enabled source in 'sources'")
    if disabled:
        logger.info("sources_disabled names=%s", ",".join(disabled))
    processing = _section(config, "processing")
    taxonomy = load_taxonomy(str(_require(processing, "taxonomy_path", "processing")))
    min_lang_chars = int(processing.get("language_min_chars", 40))
    threshold = float(processing.get("domain_threshold", 0.15))
    margin = float(processing.get("domain_margin", 0.05))
    min_chars = int(_section(config, "clean").get("min_chars", 20))
    pii_types = [str(t) for t in _section(config, "anonymize").get("pii_types", [])]
    near_threshold = float(_section(config, "verify").get("near_duplicate_threshold", 0.9))
    system_prompt = str(_require(_section(config, "build"), "system_prompt", "build"))
    out_dir = Path(str(_require(_section(config, "output"), "dir", "output")))
    split_section = _section(config, "split")
    split_cfg = SplitConfig(
        eval_ratio=float(split_section.get("eval_ratio", 0.2)),
        eval_floor_per_domain=int(split_section.get("eval_floor_per_domain", 25)),
        eval_floor_per_cell=int(split_section.get("eval_floor_per_cell", 5)),
        max_share_per_domain=float(split_section.get("max_share_per_domain", 0.35)),
        seed=seed,
        domains=tuple(domains),
        languages=tuple(languages),
    )
    connectors = [build_connector(s) for s in sources]

    # A failed run must never leave stale outputs that look valid.
    for name in OUTPUT_NAMES:
        (out_dir / name).unlink(missing_ok=True)

    run = _Run()

    # --- ingest + parse ------------------------------------------------------
    records: list[Record] = []
    for connector in connectors:
        if not connector.test_connection():
            raise SourceUnavailable(f"source '{connector.name}' failed its connection test")
        refs = sorted(connector.list_items(), key=lambda r: r.item_id)
        source = connector.name

        def fetch(ref: ItemRef, _c: SourceConnector = connector) -> RawDocument:
            return _c.fetch(ref)

        def ref_id(ref: ItemRef, _s: str = source) -> str:
            return f"{_s}:{ref.item_id}"

        raws = run.per_item("ingest", refs, fetch, ref_id)
        # Fetch failures were ingested (listed) then dropped.
        run.counters.ingest(source, len(refs) - len(raws))
        _stage_log("ingest", len(refs), len(raws), run, f"source={source}")

        def parse(raw: RawDocument, _s: str = source) -> list[Record]:
            # FAKE: identity parse, one RawDocument -> one Record.
            text = raw.content.decode("utf-8") if isinstance(raw.content, bytes) else raw.content
            meta = {k: v for k, v in raw.metadata.items() if k != "content"}
            rec = Record(
                id=f"{raw.id}#0",
                text=text,
                group_id=str(raw.metadata.get("thread_id") or raw.id),
                source_id=raw.id,
                source_type=raw.source_type,
                metadata=meta,
            )
            run.counters.ingest(_s, 1)
            return [rec]

        def parse_or_count(raw: RawDocument, _s: str = source) -> list[Record]:
            try:
                return parse(raw)
            except Exception:
                run.counters.ingest(_s, 1)  # counted as ingested, then dropped
                raise

        parsed = run.per_item("parse", raws, parse_or_count, lambda r: r.id)
        flat = [r for group in parsed for r in group]
        _stage_log("parse", len(raws), len(flat), run, f"source={source}")
        records += flat

    records.sort(key=lambda r: r.id)

    # --- clean ---------------------------------------------------------------
    def clean(rec: Record) -> Record | None:
        rec.text = rec.text.strip()
        if not rec.text:
            run.drop(rec.id, "clean", "empty_text")
            return None
        if len(rec.text) < min_chars:
            run.drop(rec.id, "clean", "too_short", f"chars={len(rec.text)}")
            return None
        return rec

    n = len(records)
    records = run.per_item("clean", records, clean, lambda r: r.id)
    _stage_log("clean", n, len(records), run)

    # --- dedup_exact ---------------------------------------------------------
    seen: dict[str, str] = {}

    def dedup_exact(rec: Record) -> Record | None:
        key = hashlib.sha256(_normalise_for_hash(rec.text).encode("utf-8")).hexdigest()
        first = seen.get(key)
        if first is not None:
            run.drop(rec.id, "dedup", "duplicate_exact", f"duplicate_of={first}")
            run.counters.duplicates_exact += 1
            return None
        seen[key] = rec.id
        return rec

    n = len(records)
    records = run.per_item("dedup", records, dedup_exact, lambda r: r.id)
    _stage_log("dedup_exact", n, len(records), run, f"duplicates_exact={run.counters.duplicates_exact}")

    # --- anonymize (STUB) ----------------------------------------------------
    for t in pii_types:
        run.counters.pii_replaced.setdefault(t, 0)

    def anonymize(rec: Record) -> Record:
        # STUB: identity; real anonymisation replaces PII and fills the counts.
        rec.pii_counts = {t: 0 for t in pii_types}
        return rec

    n = len(records)
    records = run.per_item("anonymize", records, anonymize, lambda r: r.id)
    _stage_log("anonymize", n, len(records), run, "stub=true")

    # --- dedup_near (STUB) ---------------------------------------------------
    _stage_log("dedup_near", len(records), len(records), run, "stub=true")

    # --- tag -----------------------------------------------------------------
    def tag(rec: Record) -> Record | None:
        lang, conf = detect_language(rec.text, min_chars=min_lang_chars)
        rec.language = lang
        rec.metadata["language_confidence"] = round(conf, 4)
        score = classify_domain(rec.text, taxonomy, threshold=threshold, margin=margin)
        rec.metadata["domain_score"] = score.score
        rec.metadata["domain_terms"] = score.matched_terms
        if score.domain is None:
            run.drop(rec.id, "tag", "domain_unresolved", f"score={score.score}")
            return None
        rec.domain = score.domain
        return rec

    n = len(records)
    records = run.per_item("tag", records, tag, lambda r: r.id)
    _stage_log("tag", n, len(records), run)
    tagged = list(records)

    # --- split ---------------------------------------------------------------
    split_result = split(records, split_cfg)
    for rec in split_result.capped_records:
        run.drop(rec.id, "build", "train_domain_cap", f"domain={rec.domain}")
    logger.info(
        "stage=split train=%d eval=%d capped=%d unmeasurable_domains=%s",
        len(split_result.train), len(split_result.eval), len(split_result.capped_records),
        ",".join(split_result.unmeasurable_domains) or "-",
    )

    # --- build ---------------------------------------------------------------
    def to_example(rec: Record) -> dict[str, Any]:
        return record_to_example(rec, system_prompt)

    train = run.per_item("build", split_result.train, to_example, lambda r: r.id)
    eval_ = run.per_item("build", split_result.eval, to_example, lambda r: r.id)
    _stage_log("build", len(split_result.train) + len(split_result.eval), len(train) + len(eval_), run)

    # --- verify (BLOCKING) ---------------------------------------------------
    assert_no_group_overlap(split_result.train, split_result.eval)
    pairs = assert_no_near_duplicate_across_split(
        split_result.train, split_result.eval, threshold=near_threshold
    )
    kept = len(train) + len(eval_)
    run.counters.check_invariant(kept)
    logger.info(
        "stage=verify group_overlap=0 near_duplicates=%d invariant=ok ingested=%d kept=%d dropped=%d",
        len(pairs), run.counters.ingested_total, kept, run.counters.dropped_total,
    )

    # --- report + outputs ----------------------------------------------------
    report = build_report(
        counters=run.counters,
        split_result=split_result,
        split_config=split_cfg,
        tagged=tagged,
        kept=kept,
        n_train_examples=len(train),
        n_eval_examples=len(eval_),
        domains=domains,
        languages=languages,
        stub_stages=STUB_STAGES,
    )
    write_jsonl(out_dir / TRAIN_NAME, train)
    write_jsonl(out_dir / EVAL_NAME, eval_)
    json_path, md_path = write_quality_report(report, out_dir)
    logger.info("stage=report out_dir=%s files=%d", out_dir, len(OUTPUT_NAMES))

    return PipelineResult(
        train=train,
        eval=eval_,
        split=split_result,
        counters=run.counters,
        report=report,
        drops=run.drops,
        outputs={
            "train": out_dir / TRAIN_NAME,
            "eval": out_dir / EVAL_NAME,
            "report_json": json_path,
            "report_md": md_path,
        },
    )
