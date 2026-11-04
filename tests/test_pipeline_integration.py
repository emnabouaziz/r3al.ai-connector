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
from src.processing.pipeline import OUTPUT_NAMES, PipelineResult, run_pipeline

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
    assert c.dropped["dedup"] == {"duplicate_exact": 3}
    assert c.dropped["tag"] == {"domain_unresolved": 1}
    assert c.duplicates_exact == 3
    kept = len(result.train) + len(result.eval)
    assert c.ingested_total == kept + c.dropped_total
    assert result.report["invariant"]["holds"] is True
    assert len(result.drops) == c.dropped_total


def test_near_duplicates_stay_in_their_thread(result: PipelineResult) -> None:
    # The 2 near duplicates survive (stub) but share a group with their original.
    groups: dict[str, set[str]] = {}
    for side, recs in (("train", result.split.train), ("eval", result.split.eval)):
        for r in recs:
            groups.setdefault(r.group_id, set()).add(side)
    assert all(len(sides) == 1 for sides in groups.values())
    sizes = [sum(r.group_id == g for r in result.split.train + result.split.eval) for g in groups]
    assert sizes.count(2) == 2


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
    assert "anonymize" in md and "dedup_near" in md


def _hashes(out_dir: Path) -> dict[str, str]:
    return {n: hashlib.sha256((out_dir / n).read_bytes()).hexdigest() for n in OUTPUT_NAMES}


def test_rerun_is_byte_identical(tmp_path: Path) -> None:
    cfg = make_config(tmp_path / "a")
    run_pipeline(cfg)
    first = _hashes(tmp_path / "a")
    run_pipeline(cfg)
    assert _hashes(tmp_path / "a") == first
    run_pipeline(make_config(tmp_path / "b"))
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
