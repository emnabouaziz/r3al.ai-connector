"""Ground-truth recall scan: literal, formatting-independent, never leaks values."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import run as run_cli
from src.processing.anonymize import AnonymisationResult
from src.report.ground_truth_scan import normalise, scan_against_manifest, value_hash

ROOT = Path(__file__).resolve().parents[1]
IBAN = "BE68 5390 0754 7034"
NAME = "Lotte Maes"


def _manifest(path: Path, entries: list[dict[str, Any]]) -> str:
    values = [{**e, "normalised": normalise(e["value"])} for e in entries]
    path.write_text(json.dumps({"generated_at": "test", "seed": 0, "values": values}), encoding="utf-8")
    return str(path)


def _jsonl(path: Path, users: list[str]) -> str:
    lines = [
        json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": "ok"}],
                    "meta": {"domain": "billing", "language": "nl", "source_ids": ["s"]}}, ensure_ascii=False)
        for u in users
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def manifest(tmp_path: Path) -> str:
    return _manifest(tmp_path / "manifest.json", [
        {"type": "IBAN", "value": IBAN, "sources": ["invoices:1"]},
        {"type": "PERSON", "value": NAME, "sources": ["claims:2"]},
        {"type": "PHONE", "value": "+32 478 12 34 56", "sources": ["claims:3"]},
    ])


def test_iban_written_without_spaces_is_detected(tmp_path: Path, manifest: str) -> None:
    out = _jsonl(tmp_path / "train.jsonl", ["clean line", "storten op BE68539007547034 aub"])
    result = scan_against_manifest([out], manifest)
    assert result.found_in_output == {"IBAN": 1, "PERSON": 0, "PHONE": 0}
    assert result.recall["IBAN"] == 0.0
    assert [(m["type"], m["line"]) for m in result.misses] == [("IBAN", 2)]


def test_value_split_across_line_break_is_detected(tmp_path: Path, manifest: str) -> None:
    out = _jsonl(tmp_path / "train.jsonl", ["Groeten, Lotte\nMaes", "bel +32 478\n12 34 56"])
    result = scan_against_manifest([out], manifest)
    assert result.found_in_output["PERSON"] == 1 and result.found_in_output["PHONE"] == 1


def test_value_split_across_user_and_assistant_is_detected(tmp_path: Path, manifest: str) -> None:
    line = json.dumps({"messages": [{"role": "user", "content": "Groeten, Lotte"},
                                    {"role": "assistant", "content": "Maes. Wij bekijken het."}]})
    out = tmp_path / "eval.jsonl"
    out.write_text(line + "\n", encoding="utf-8")
    assert scan_against_manifest([str(out)], manifest).found_in_output["PERSON"] == 1


def test_misses_contain_hash_never_the_value(tmp_path: Path, manifest: str) -> None:
    out = _jsonl(tmp_path / "train.jsonl", [f"{NAME} {IBAN} +32 478 12 34 56"])
    result = scan_against_manifest([out], manifest)
    assert len(result.misses) == 3
    dumped = json.dumps(result.as_report())
    for raw in (IBAN, "BE68539007547034", NAME, "lottemaes", "32478123456", "Maes"):
        assert raw.lower() not in dumped.lower()
    assert {m["value_hash"] for m in result.misses} == {
        value_hash(normalise(v)) for v in (IBAN, NAME, "+32 478 12 34 56")
    }


def test_value_absent_from_ingested_texts_is_not_counted(tmp_path: Path, manifest: str) -> None:
    out = _jsonl(tmp_path / "train.jsonl", [f"rekening {IBAN}"])
    # Only invoices:1 and claims:2 entered the pipeline; claims:3 (the phone) did not.
    result = scan_against_manifest([out], manifest, ingested_sources={"invoices:1", "claims:2"})
    assert result.total_injected == {"IBAN": 1, "PERSON": 1}
    assert result.recall == {"IBAN": 0.0, "PERSON": 1.0}
    assert result.overall_recall == 0.5


def test_placeholders_and_clean_output_give_full_recall(tmp_path: Path, manifest: str) -> None:
    out = _jsonl(tmp_path / "train.jsonl", ["Groeten, [PERSON]. Rekening [IBAN], tel [PHONE]."])
    result = scan_against_manifest([out], manifest)
    assert result.leaked == 0 and result.overall_recall == 1.0 and result.misses == []


# --- run.py wiring: staging, promotion, absent manifest -------------------------------


def _cli_config(tmp_path: Path, manifest: str | None) -> Path:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(tmp_path / "out")
    raw["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    if manifest is not None:
        raw["verify"]["pii_manifest"] = manifest
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _fake_manifest(tmp_path: Path) -> str:
    # "Jan Peeters" signs several FakeConnector documents.
    sources = [f"memory://fake_demo/doc-{i:03d}" for i in range(60)]
    return _manifest(tmp_path / "manifest.json", [{"type": "PERSON", "value": "Jan Peeters", "sources": sources}])


OUTPUTS = ("train.jsonl", "eval.jsonl", "quality_report.json", "quality_report.md")


def test_absent_manifest_skips_scan_and_succeeds(tmp_path: Path) -> None:
    cfg = _cli_config(tmp_path, str(tmp_path / "missing.json"))
    assert run_cli.main(["--config", str(cfg)]) == 0
    report = json.loads((tmp_path / "out" / "quality_report.json").read_text(encoding="utf-8"))
    assert report["anonymisation_recall"]["status"] == "no_ground_truth"
    assert "Anonymisation recall (ground truth)" in (tmp_path / "out" / "quality_report.md").read_text(
        encoding="utf-8"
    )


def test_outputs_promoted_to_out_on_success(tmp_path: Path) -> None:
    cfg = _cli_config(tmp_path, _fake_manifest(tmp_path))
    assert run_cli.main(["--config", str(cfg)]) == 0
    assert all((tmp_path / "out" / name).is_file() for name in OUTPUTS)
    assert not (tmp_path / "out" / ".staging").exists()
    report = json.loads((tmp_path / "out" / "quality_report.json").read_text(encoding="utf-8"))
    assert report["anonymisation_recall"]["recall"] == {"PERSON": 1.0}


def test_outputs_stay_in_staging_on_ground_truth_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_names(text: str, ner_model: Any) -> AnonymisationResult:
        return AnonymisationResult(text=text)  # anonymiser that misses everything

    monkeypatch.setattr("src.processing.pipeline.anonymise", no_names)
    cfg = _cli_config(tmp_path, _fake_manifest(tmp_path))
    assert run_cli.main(["--config", str(cfg)]) == 4
    staging = tmp_path / "out" / ".staging"
    assert all((staging / name).is_file() for name in OUTPUTS)
    assert not any((tmp_path / "out" / name).exists() for name in OUTPUTS)
    report = json.loads((staging / "quality_report.json").read_text(encoding="utf-8"))
    assert report["anonymisation_recall"]["found_in_output"] == {"PERSON": 1}
    assert "Peeters" not in json.dumps(report)
