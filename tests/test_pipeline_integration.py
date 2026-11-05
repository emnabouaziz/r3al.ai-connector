"""Full pipeline on FakeConnector: outputs, schema, invariants, idempotency."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import run as run_cli
from src.config import load_config
from src.connectors.fake import FakeConnector
from src.contracts import ItemRef, RawDocument
from src.dataset.leakage import LeakageError
from src.dataset.splitter import SplitResult
from src.dataset.splitter import split as real_split
from src.processing.pipeline import OUTPUT_NAMES, PipelineResult, promote_outputs, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["claims", "policies", "billing", "onboarding", "complaints", "compliance"]
LANGS = ["nl", "fr", "en"]


def make_config(out_dir: Path) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg["output"]["dir"] = str(out_dir)
    cfg["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    return cfg


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory) -> PipelineResult:
    return run_pipeline(make_config(tmp_path_factory.mktemp("out")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assert_schema(example: dict[str, Any]) -> None:
    assert list(example) == ["messages", "meta"]
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    for m in example["messages"]:
        assert list(m) == ["role", "content"]
        assert isinstance(m["content"], str) and m["content"]
    meta = example["meta"]
    assert list(meta) == ["domain", "language", "source_ids"]
    assert meta["domain"] in DOMAINS
    assert meta["language"] in LANGS + ["unknown"]
    assert meta["source_ids"] and all(isinstance(s, str) for s in meta["source_ids"])


def test_outputs_exist_and_match_schema(result: PipelineResult) -> None:
    for path in result.outputs.values():
        assert path.is_file()
    train = read_jsonl(result.outputs["train"])
    eval_ = read_jsonl(result.outputs["eval"])
    assert train and eval_
    for example in train + eval_:
        assert_schema(example)
    assert len(train) == len(result.train) and len(eval_) == len(result.eval)


def test_no_group_shared_between_train_and_eval(result: PipelineResult) -> None:
    train_groups = {r.group_id for r in result.split.train}
    eval_groups = {r.group_id for r in result.split.eval}
    assert train_groups and eval_groups
    assert not train_groups & eval_groups
    # Same check from the written files, via source ids.
    train_sources = {s for e in read_jsonl(result.outputs["train"]) for s in e["meta"]["source_ids"]}
    eval_sources = {s for e in read_jsonl(result.outputs["eval"]) for s in e["meta"]["source_ids"]}
    assert not train_sources & eval_sources


def test_counters_and_invariant(result: PipelineResult) -> None:
    c = result.counters
    assert c.ingested_per_source == {"fake_demo": 60}
    assert c.dropped["clean"] == {"empty_text": 1, "too_short": 1}
    assert c.dropped["dedup"] == {"duplicate_exact": 3, "near_duplicate": 2}
    assert c.duplicates_near == 2
    assert c.dropped["tag"] == {"domain_unresolved": 1}
    assert c.duplicates_exact == 3
    kept = len(result.train) + len(result.eval)
    assert c.ingested_total == kept + c.dropped_total + c.capped_total
    assert result.report["invariant"]["holds"] is True
    assert len(result.drops) == c.dropped_total
    # Capping is never a drop.
    assert "train_domain_cap" not in str(c.dropped)
    assert c.capped == result.split.capped_per_domain


def test_eval_share_and_statuses(result: PipelineResult) -> None:
    kept = len(result.train) + len(result.eval)
    assert len(result.eval) / kept <= 0.30 + 1e-9
    statuses = result.report["domain_status"]
    assert sorted(statuses) == sorted(DOMAINS)
    assert set(statuses.values()) <= {"ABSENT", "INSUFFICIENT_DATA", "PARTIALLY_MEASURABLE", "MEASURABLE"}
    train_domains = {e["meta"]["domain"] for e in result.train}
    for e in result.eval:
        assert e["meta"]["domain"] in train_domains


def test_report_separates_dropped_and_capped_and_shows_absent_domain(tmp_path: Path) -> None:
    from src.counters import Counters
    from src.dataset.splitter import SplitConfig, split
    from src.report.quality_report import build_report, render_markdown
    from tests.test_splitter import make_dataset

    records = make_dataset({"claims": 300, "policies": 50, "billing": 50, "complaints": 50})
    cfg = SplitConfig()
    res = split(records, cfg)
    counters = Counters()
    counters.ingest("s", len(records) + 2)
    counters.drop("tag", "domain_unresolved", 2)
    for d, n in res.capped_per_domain.items():
        counters.cap(d, n)
    report = build_report(
        counters=counters, split_result=res, split_config=cfg, tagged=records,
        kept=len(res.train) + len(res.eval), n_train_examples=len(res.train),
        n_eval_examples=len(res.eval), domains=DOMAINS, languages=LANGS, stub_stages=[],
    )
    assert report["invariant"]["holds"] is True
    assert report["dropped"]["total"] == 2
    assert report["capped"]["total"] == len(res.capped_records) > 0
    md = render_markdown(report)
    assert "## Dropped (quality) - total 2" in md
    assert f"## Capped (sampling decision) - total {report['capped']['total']}" in md
    assert "| compliance | **ABSENT** |" in md and "| onboarding | **ABSENT** |" in md
    for d in DOMAINS:
        for lang in LANGS:
            assert f"| {d} | {lang} |" in md


def test_report_shows_floor_scaling(tmp_path: Path) -> None:
    from src.counters import Counters
    from src.dataset.splitter import SplitConfig, split
    from src.report.quality_report import build_report, render_markdown
    from tests.test_splitter import make_dataset

    records = make_dataset({d: 40 for d in DOMAINS})  # type: ignore[misc]
    cfg = SplitConfig(min_train_per_domain=5, max_share_per_domain=1.0)
    res = split(records, cfg)
    counters = Counters()
    counters.ingest("s", len(records))
    report = build_report(
        counters=counters, split_result=res, split_config=cfg, tagged=records,
        kept=len(res.train) + len(res.eval), n_train_examples=len(res.train),
        n_eval_examples=len(res.eval), domains=DOMAINS, languages=LANGS, stub_stages=[],
    )
    assert report["split"]["floor_scaling_applied"] == res.floor_scaling_applied is not None
    assert "eval floors scaled down by a factor of" in render_markdown(report)


def test_near_duplicates_keep_one_representative(result: PipelineResult) -> None:
    # The fake source re-sends 2 messages ("RE: " + original) in the same thread:
    # dedup_near keeps one record per pair, and no group straddles the split.
    groups: dict[str, set[str]] = {}
    for side, recs in (("train", result.split.train), ("eval", result.split.eval)):
        for r in recs:
            groups.setdefault(r.group_id, set()).add(side)
    assert all(len(sides) == 1 for sides in groups.values())
    near = [d for d in result.drops if d.reason == "near_duplicate"]
    assert len(near) == 2
    # The split now holds examples ("<record id>/qa<k>"): check by originating record.
    kept_records = {r.id.split("/qa")[0] for r in result.split.train + result.split.eval + result.split.capped_records}
    for drop in near:
        representative = drop.detail.removeprefix("representative=")
        assert representative in kept_records and drop.record_id not in kept_records


def test_report_shows_all_18_cells(result: PipelineResult) -> None:
    report = json.loads(result.outputs["report_json"].read_text(encoding="utf-8"))
    matrix = report["domain_language_matrix"]
    assert sorted(matrix) == sorted(DOMAINS)
    assert all(sorted(row) == sorted(LANGS) for row in matrix.values())
    per_cell = report["split"]["per_cell"]
    assert sum(len(cells) for cells in per_cell.values()) >= 18
    md = result.outputs["report_md"].read_text(encoding="utf-8")
    assert md.startswith("# Data quality report")
    for d in DOMAINS:
        for lang in LANGS:
            assert f"| {d} | {lang} |" in md
    assert "Stub stages" not in md  # every stage is implemented now
    assert "## Needs client review" in md and "## Cleaning" in md


def _hashes(out_dir: Path) -> dict[str, str]:
    return {n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest() for n in OUTPUT_NAMES}


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "a")
    promote_outputs(run_pipeline(cfg))
    first = _hashes(tmp_path / "a")
    promote_outputs(run_pipeline(cfg))
    assert _hashes(tmp_path / "a") == first
    promote_outputs(run_pipeline(make_config(tmp_path / "b")))
    assert _hashes(tmp_path / "b") == first


def test_one_failing_item_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = FakeConnector.fetch

    def flaky(self: FakeConnector, ref: ItemRef) -> RawDocument:
        if ref.item_id == "doc-005":
            raise RuntimeError("simulated source failure")
        return original(self, ref)

    monkeypatch.setattr(FakeConnector, "fetch", flaky)
    res = run_pipeline(make_config(tmp_path))
    assert res.counters.dropped["ingest"] == {"ingest_error": 1}
    assert res.counters.ingested_total == 60
    assert res.report["invariant"]["holds"] is True


def test_leakage_fails_the_run_and_leaves_no_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def leaky_split(records: Any, cfg: Any) -> SplitResult:
        res = real_split(records, cfg)
        res.eval.append(res.train[0])  # same group on both sides
        return res

    monkeypatch.setattr("src.processing.pipeline.split", leaky_split)
    with pytest.raises(LeakageError):
        run_pipeline(make_config(tmp_path))
    assert not any((tmp_path / n).exists() for n in OUTPUT_NAMES)
    assert not any((tmp_path / ".staging" / n).exists() for n in OUTPUT_NAMES)


def _write_cli_config(tmp_path: Path) -> Path:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(tmp_path / "out")
    raw["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _write_cli_config(tmp_path)
    assert run_cli.main(["--config", str(cfg_path)]) == 0
    assert (tmp_path / "out" / "train.jsonl").is_file()

    def leaky_split(records: Any, cfg: Any) -> SplitResult:
        res = real_split(records, cfg)
        res.eval.append(res.train[0])
        return res

    monkeypatch.setattr("src.processing.pipeline.split", leaky_split)
    assert run_cli.main(["--config", str(cfg_path)]) == 2


# --- Example construction (real Q/A, no half-split) ----------------------------------------


def test_source_ids_are_real_raw_document_ids(result: PipelineResult) -> None:
    connector = FakeConnector({"seed": 42})
    connector.name = "fake_demo"
    real_ids = {connector.fetch(ref).id for ref in connector.list_items()}
    lines = read_jsonl(result.outputs["train"]) + read_jsonl(result.outputs["eval"])
    assert lines
    for line in lines:
        assert line["meta"]["source_ids"] and set(line["meta"]["source_ids"]) <= real_ids


def test_no_example_is_a_text_cut_in_half(result: PipelineResult) -> None:
    from src.connectors.fake import REPLIES

    replies = set(REPLIES.values())
    for line in read_jsonl(result.outputs["train"]) + read_jsonl(result.outputs["eval"]):
        user, assistant = line["messages"][1]["content"], line["messages"][2]["content"]
        # The answer is a whole agent reply, the question a whole customer message.
        assert any(reply in assistant for reply in replies)
        assert not any(reply[:40] in user for reply in replies)


def test_system_prompt_matches_the_example_language(result: PipelineResult) -> None:
    marker = {"nl": "Nederlands", "fr": "français", "en": "English"}
    for line in read_jsonl(result.outputs["train"]) + read_jsonl(result.outputs["eval"]):
        system = line["messages"][0]["content"]
        assert marker[line["meta"]["language"]] in system and "Example Insurance SA" in system


def test_report_has_the_build_rejection_table(result: PipelineResult) -> None:
    build = result.report["build"]
    for reason in ("no_agent_reply", "auto_acknowledgement", "refers_to_attachment", "answer_too_short",
                   "question_too_short", "answer_is_a_question", "placeholder_heavy", "no_question_anchor",
                   "chunk_too_short", "answer_not_grounded"):
        assert reason in build["rejections"]
    assert build["candidates"] == build["accepted"] + sum(
        n for r, n in build["rejections"].items() if r != "duplicate_example")
    md = result.outputs["report_md"].read_text(encoding="utf-8")
    assert "## Example construction" in md and "| refers_to_attachment |" in md
