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
from collections import Counter
import json
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
from src.dataset.builder import (
    BUILDER_NAMES,
    KIND_CHUNK,
    build_record,
    candidate_to_record,
    example_line,
    make_builders,
    record_kind,
    write_jsonl,
)
from src.dataset.builders import BuildSettings, ChunkBuilder, FactsBuilder, TicketThreadBuilder
from src.dataset.question_gen import DOCUMENT_TITLE_KEY, document_title
from src.dataset.system_prompt import from_config as system_prompts_from_config
from src.dataset.leakage import assert_no_group_overlap, assert_no_near_duplicate_across_split
from src.dataset.splitter import SplitConfig, SplitResult, split
from src.processing.anonymize import anonymise, load_ner
from src.processing.chunking import DEFAULT_MAX_CHARS, chunk_text
from src.processing.clean import MESSAGE_SEPARATOR, clean_text, visible_text
from src.processing.dedup_near import DEFAULT_THRESHOLD, find_near_duplicates, merge_groups
from src.processing.parsers import get_parser
from src.processing.domain import classify_domain
from src.processing.language import detect_language
from src.registry import build as build_connector
from src.report.quality_report import JSON_NAME, MD_NAME, build_report, write_quality_report

logger = logging.getLogger(__name__)

STUB_STAGES: list[str] = []
TRAIN_NAME = "train.jsonl"
EVAL_NAME = "eval.jsonl"
OUTPUT_NAMES = (TRAIN_NAME, EVAL_NAME, JSON_NAME, MD_NAME)
STAGING_DIRNAME = ".staging"  # outputs wait here until the PII scans pass

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
    outputs: dict[str, Path] = field(default_factory=dict)  # staged paths until promoted
    out_dir: Path = Path(".")
    staging_dir: Path = Path(".")
    # Source uris ("table:pk") of every document that entered the pipeline.
    ingested_sources: set[str] = field(default_factory=set)


def promote_outputs(result: PipelineResult) -> dict[str, Path]:
    """Move the staged outputs into out/ (only after every scan passed)."""
    result.out_dir.mkdir(parents=True, exist_ok=True)
    promoted: dict[str, Path] = {}
    for key, staged in result.outputs.items():
        target = result.out_dir / staged.name
        staged.replace(target)
        promoted[key] = target
    try:
        result.staging_dir.rmdir()
    except OSError:
        pass  # not empty (foreign files): leave it
    result.outputs = promoted
    return promoted


class _Run:
    """Mutable state of one run: counters + drop log."""

    def __init__(self) -> None:
        self.counters = Counters()
        self.drops: list[DropReason] = []

    def drop(self, record_id: str, stage: str, reason: str, detail: str = "") -> None:
        # Record / document ids are "<source name>:<item id>[#chunk]".
        self.counters.drop(stage, reason, source=record_id.split(":", 1)[0])
        self.drops.append(DropReason(record_id, stage, reason, detail))

    def per_item(
        self,
        stage: str,
        items: Iterable[T],
        fn: Callable[[T], U | None],
        item_id: Callable[[T], str],
        error_reason: str | None = None,
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
                self.drop(rid, stage, error_reason or f"{stage}_error", type(exc).__name__)
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
    max_chunk_chars = int(_section(config, "parse").get("max_chunk_chars", DEFAULT_MAX_CHARS))
    near_threshold_dedup = float(_section(config, "dedup").get("near_threshold", DEFAULT_THRESHOLD))
    anonymize_cfg = _section(config, "anonymize")
    pii_types = [str(t) for t in anonymize_cfg.get("pii_types", [])]
    # Loaded once per process; a missing model is a run-level failure.
    # Taxonomy terms are domain vocabulary: never masked as a person on their own.
    domain_vocabulary = frozenset(
        str(term)
        for by_language in taxonomy["domains"].values()
        for terms in by_language.values()
        for term in terms
    )
    ner = load_ner(str(anonymize_cfg.get("ner_model", "xx_ent_wiki_sm")), domain_vocabulary)
    near_threshold = float(_section(config, "verify").get("near_duplicate_threshold", 0.9))
    build_section = _section(config, "build")
    prompts = system_prompts_from_config(build_section)  # run-level error if insurer_name missing
    build_settings = BuildSettings(
        min_question_chars=int(build_section.get("min_question_chars", 30)),
        min_answer_chars=int(build_section.get("min_answer_chars", 40)),
        max_placeholder_share=float(build_section.get("max_placeholder_share", 0.12)),
        min_chunk_chars=int(build_section.get("min_chunk_chars", 80)),
        max_answer_chars=int(build_section.get("max_answer_chars", 900)),
    )
    builders = make_builders(build_settings)
    out_dir = Path(str(_require(_section(config, "output"), "dir", "output")))
    split_section = _section(config, "split")
    split_cfg = SplitConfig(
        eval_ratio=float(split_section.get("eval_ratio", 0.2)),
        eval_floor_per_domain=int(split_section.get("eval_floor_per_domain", 25)),
        eval_floor_per_cell=int(split_section.get("eval_floor_per_cell", 5)),
        max_share_per_domain=float(split_section.get("max_share_per_domain", 0.35)),
        min_train_per_domain=int(split_section.get("min_train_per_domain", 20)),
        eval_ratio_max=float(split_section.get("eval_ratio_max", 0.30)),
        seed=seed,
        domains=tuple(domains),
        languages=tuple(languages),
    )
    connectors = [build_connector(s) for s in sources]

    staging_dir = out_dir / STAGING_DIRNAME
    # A failed run must never leave stale outputs that look valid.
    for name in OUTPUT_NAMES:
        (out_dir / name).unlink(missing_ok=True)
        (staging_dir / name).unlink(missing_ok=True)
    ingested_sources: set[str] = set()
    # Items the pipeline refuses to decide, escalated to the client (deduplicated).
    review_items: dict[str, dict[str, Any]] = {}

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
        ingested_sources.update(raw.source_uri for raw in raws)
        # Fetch failures were ingested (listed) then dropped.
        run.counters.ingest(source, len(refs) - len(raws))
        _stage_log("ingest", len(refs), len(raws), run, f"source={source}")

        def parse(raw: RawDocument, _s: str = source) -> list[Record]:
            # Generic for every source: dispatch on mime_type. One record per document;
            # chunking happens after cleaning (clean stage).
            for item in raw.metadata.get("needs_review", []):
                review_items.setdefault(json.dumps(item, sort_keys=True), item)
            if raw.metadata.get("superseded"):
                run.counters.ingest(_s, 1)
                run.counters.superseded += 1
                run.drop(raw.id, "parse", "superseded", f"by={raw.metadata.get('superseded_by', '')}")
                return []
            parser = get_parser(raw.mime_type)
            text = parser.extract(raw.content)
            if not text.strip() and parser.empty_reason is not None:
                run.counters.ingest(_s, 1)
                run.drop(raw.id, "parse", parser.empty_reason)
                return []
            meta = {k: v for k, v in raw.metadata.items() if k not in ("content", "needs_review")}
            run.counters.ingest(_s, 1)
            return [
                Record(
                    id=f"{raw.id}#0",
                    text=text,
                    # group_id = the document / thread, never the chunk.
                    group_id=str(raw.metadata.get("thread_id") or raw.id),
                    source_id=raw.id,
                    source_type=raw.source_type,
                    metadata=meta,
                )
            ]

        def parse_or_count(raw: RawDocument, _s: str = source) -> list[Record]:
            try:
                return parse(raw)
            except Exception:
                run.counters.ingest(_s, 1)  # counted as ingested, then dropped
                raise

        parsed = run.per_item("parse", raws, parse_or_count, lambda r: r.id)
        flat = [r for group in parsed for r in group]
        # One document can yield several chunks: report documents and records separately.
        documents_kept = sum(1 for group in parsed if group)
        logger.info(
            "stage=parse source=%s documents_in=%d documents_dropped=%d records_out=%d",
            source, len(raws), len(raws) - documents_kept, len(flat),
        )
        records += flat

    records.sort(key=lambda r: r.id)

    # --- clean, then chunk -----------------------------------------------------
    def clean(rec: Record) -> list[Record] | None:
        original = rec.text.strip()
        if not original:
            run.drop(rec.id, "clean", "empty_text")
            return None
        if len(original) < min_chars:
            run.drop(rec.id, "clean", "too_short", f"chars={len(original)}")
            return None
        result = clean_text(original)
        for operation, count in result.operations.items():
            run.counters.clean_op(operation, count)
        if result.auto_acknowledgement:
            run.drop(rec.id, "clean", "auto_acknowledgement")
            return None
        text = result.text.strip()
        visible = visible_text(text)
        if len(visible) < min_chars:
            run.drop(rec.id, "clean", "too_short_after_clean", f"chars={len(visible)}")
            return None
        # Chunk AFTER cleaning, so boilerplate never decides chunk boundaries.
        # Only reference text is chunked: a thread's or a case's messages are aligned
        # with their roles and become pairs in the build stage.
        is_chunk = record_kind(rec) == KIND_CHUNK
        chunks = chunk_text(text, max_chunk_chars) if is_chunk else [text]
        # The document's title travels with every chunk: questions about a section need it.
        title = document_title(text) if is_chunk else None
        if title is not None:
            rec.metadata[DOCUMENT_TITLE_KEY] = title
        run.counters.ingest(rec.id.split(":", 1)[0], len(chunks) - 1)  # 1 record -> N chunks
        return [
            Record(
                id=f"{rec.source_id}#{i}",
                text=chunk,
                group_id=rec.group_id,
                source_id=rec.source_id,
                source_type=rec.source_type,
                metadata={**rec.metadata, "chunk": i, "chunks": len(chunks)},
            )
            for i, chunk in enumerate(chunks)
        ]

    n = len(records)
    cleaned = run.per_item("clean", records, clean, lambda r: r.id)
    records = [r for chunked in cleaned for r in chunked]
    # Chunking expands records: report records dropped and chunks produced separately.
    logger.info(
        "stage=clean in=%d dropped=%d chunks_out=%d operations=%s",
        n, n - len(cleaned), len(records),
        ",".join(f"{op}:{c}" for op, c in sorted(run.counters.cleaned.items())) or "-",
    )

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

    # --- anonymize -----------------------------------------------------------
    for t in pii_types:
        run.counters.pii_replaced.setdefault(t, 0)

    def anonymize(rec: Record) -> Record:
        # On exception the record is dropped by per_item: never a raw fallback.
        # Message by message, so no entity span can cross a message boundary
        # (a text without separators is a single message).
        results = [anonymise(part, ner) for part in rec.text.split(MESSAGE_SEPARATOR)]
        rec.text = MESSAGE_SEPARATOR.join(r.text for r in results)
        replaced: Counter[str] = Counter()
        for result in results:
            replaced.update(result.replaced)
            for pii_type, count in result.candidates_rejected.items():
                run.counters.reject_pii_candidate(pii_type, count)
        # Structured field values (case bundles) go through the same anonymiser:
        # whatever a connector selects, PII never reaches a builder in clear.
        fields = rec.metadata.get("fields")
        if isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(value, str) and value:
                    field_result = anonymise(value, ner)
                    fields[key] = field_result.text
                    replaced.update(field_result.replaced)
        rec.pii_counts = {t: 0 for t in pii_types} | dict(replaced)
        for pii_type, count in replaced.items():
            run.counters.add_pii(pii_type, count)
        return rec

    n = len(records)
    records = run.per_item(
        "anonymize", records, anonymize, lambda r: r.id, error_reason="anonymisation_failed"
    )
    _stage_log(
        "anonymize", n, len(records), run,
        "replaced=" + ",".join(f"{t}:{c}" for t, c in sorted(run.counters.pii_replaced.items())),
    )

    # --- dedup_near (after anonymisation: see dedup_near module docstring) ----
    clusters = find_near_duplicates(records, near_threshold_dedup)
    group_map = merge_groups(records, clusters)
    dropped_ids: set[str] = set()
    for cluster in clusters:
        for member in cluster.member_ids:
            if member != cluster.representative_id:
                dropped_ids.add(member)
                run.counters.duplicates_near += 1
                run.drop(member, "dedup", "near_duplicate", f"representative={cluster.representative_id}")
        if len(cluster.sources) > 1:
            item = {
                "kind": "cross_source_near_duplicate",
                "items": list(cluster.member_ids),
                "detail": (
                    f"Near-identical texts from different sources ({', '.join(cluster.sources)}), "
                    f"similarity >= {cluster.min_similarity}. One was kept; confirm they describe "
                    "the same case."
                ),
            }
            review_items.setdefault(json.dumps(item, sort_keys=True), item)
    n = len(records)
    records = [r for r in records if r.id not in dropped_ids]
    for rec in records:
        rec.group_id = group_map.get(rec.group_id, rec.group_id)
    _stage_log(
        "dedup_near", n, len(records), run,
        f"clusters={len(clusters)} duplicates_near={run.counters.duplicates_near} "
        f"groups_merged={sum(1 for g, m in group_map.items() if g != m)}",
    )

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

    # --- build: records -> question/answer examples (BEFORE the split) --------
    # Accounting unit = candidate: a record yielding N candidates counts N
    # units (N-1 added to ingested); every rejected candidate is a build drop.
    build_stats: dict[str, dict[str, int]] = {}
    build_rejections: dict[str, dict[str, int]] = {}  # kind -> reason -> count

    def build_one(rec: Record) -> list[Record]:
        kind = record_kind(rec)
        candidates, rejections = build_record(rec, builders)
        units = len(candidates) + sum(rejections.values())
        if units == 0:
            rejections, units = {"no_candidate": 1}, 1
        run.counters.ingest(rec.id.split(":", 1)[0], units - 1)
        for reason, count in sorted(rejections.items()):
            for _ in range(count):
                run.drop(rec.id, "build", reason)
        stats = build_stats.setdefault(kind, {"records": 0, "candidates": 0, "accepted": 0})
        stats["records"] += 1
        stats["candidates"] += units
        stats["accepted"] += len(candidates)
        per_reason = build_rejections.setdefault(kind, {})
        for reason, count in rejections.items():
            per_reason[reason] = per_reason.get(reason, 0) + count
        return [candidate_to_record(c, f"{rec.id}/qa{k}", kind) for k, c in enumerate(candidates)]

    n = len(records)
    examples = [e for built in run.per_item("build", records, build_one, lambda r: r.id) for e in built]

    # Example-level near-duplicates (templated replies repeat across threads):
    # keep one, so the same pair can never sit in both train and eval.
    duplicate_ids = {
        member
        for cluster in find_near_duplicates(examples, near_threshold_dedup)
        for member in cluster.member_ids
        if member != cluster.representative_id
    }
    kind_of = {e.id: str(e.metadata.get("kind", KIND_CHUNK)) for e in examples}
    for example_id in sorted(duplicate_ids):
        run.drop(example_id, "build", "duplicate_example")
        per_reason = build_rejections.setdefault(kind_of[example_id], {})
        per_reason["duplicate_example"] = per_reason.get("duplicate_example", 0) + 1
    examples = [e for e in examples if e.id not in duplicate_ids]
    logger.info(
        "stage=build records_in=%d candidates=%d accepted=%d duplicate_examples=%d examples_out=%d",
        n, sum(s["candidates"] for s in build_stats.values()),
        sum(s["accepted"] for s in build_stats.values()), len(duplicate_ids), len(examples),
    )
    tagged = list(examples)

    # --- split (on examples, by group_id) -------------------------------------
    split_result = split(examples, split_cfg)
    # Capping is a sampling decision, never a drop.
    for domain, n in split_result.capped_per_domain.items():
        run.counters.cap(domain, n)
    logger.info(
        "stage=split train=%d eval=%d capped=%d floor_scaling=%s statuses=%s",
        len(split_result.train), len(split_result.eval), len(split_result.capped_records),
        split_result.floor_scaling_applied if split_result.floor_scaling_applied is not None else "none",
        ",".join(f"{d}:{s}" for d, s in sorted(split_result.domain_status.items())),
    )

    # --- emit: the client's chat format -------------------------------------
    def to_line(example: Record) -> dict[str, Any]:
        return example_line(example, prompts)

    train = run.per_item("build", split_result.train, to_line, lambda r: r.id)
    eval_ = run.per_item("build", split_result.eval, to_line, lambda r: r.id)

    # --- verify (BLOCKING) ---------------------------------------------------
    assert_no_group_overlap(split_result.train, split_result.eval)
    pairs = assert_no_near_duplicate_across_split(
        split_result.train, split_result.eval, threshold=near_threshold
    )
    kept = len(train) + len(eval_)
    run.counters.check_invariant(kept)
    logger.info(
        "stage=verify group_overlap=0 near_duplicates=%d invariant=ok ingested=%d kept=%d dropped=%d capped=%d",
        len(pairs), run.counters.ingested_total, kept, run.counters.dropped_total,
        run.counters.capped_total,
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
        needs_review=list(review_items.values()),
        near_duplicate_clusters=len(clusters),
        build_summary={
            "per_kind": {k: dict(v) for k, v in sorted(build_stats.items())},
            "reasons": sorted({*TicketThreadBuilder.REASONS, *ChunkBuilder.REASONS, *FactsBuilder.REASONS,
                               "duplicate_example", "build_error", "no_candidate"}),
            # Per builder: every reason it can report (zeros shown) and what it rejected.
            "per_builder": {
                BUILDER_NAMES[kind]: {
                    **build_stats.get(kind, {"records": 0, "candidates": 0, "accepted": 0}),
                    "rejections": {
                        reason: build_rejections.get(kind, {}).get(reason, 0)
                        for reason in sorted({*builder.REASONS, "duplicate_example",
                                              *build_rejections.get(kind, {})})
                    },
                }
                for kind, builder in sorted(builders.items(), key=lambda kv: BUILDER_NAMES[kv[0]])
            },
            "duplicate_examples": len(duplicate_ids),
            "system_prompt_language": prompts.mode,
        },
    )
    # Written to staging; run.py promotes them to out/ once both PII scans pass.
    write_jsonl(staging_dir / TRAIN_NAME, train)
    write_jsonl(staging_dir / EVAL_NAME, eval_)
    json_path, md_path = write_quality_report(report, staging_dir)
    logger.info("stage=report staging_dir=%s files=%d", staging_dir, len(OUTPUT_NAMES))

    return PipelineResult(
        train=train,
        eval=eval_,
        split=split_result,
        counters=run.counters,
        report=report,
        drops=run.drops,
        outputs={
            "train": staging_dir / TRAIN_NAME,
            "eval": staging_dir / EVAL_NAME,
            "report_json": json_path,
            "report_md": md_path,
        },
        out_dir=out_dir,
        staging_dir=staging_dir,
        ingested_sources=ingested_sources,
    )
