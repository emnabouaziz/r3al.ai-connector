"""Final blocking PII guard over the output files."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import run as run_cli
from src.report.pii_scan import Finding, scan_output_files

ROOT = Path(__file__).resolve().parents[1]
PLANTED_IBAN = "BE68 5390 0754 7034"


def _example(user: str) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are a support assistant."},
                {"role": "user", "content": user},
                {"role": "assistant", "content": "We will look into it."},
            ],
            "meta": {"domain": "billing", "language": "nl", "source_ids": ["crm:invoices:1"]},
        },
        ensure_ascii=False,
    )


def _write(path: Path, users: list[str]) -> str:
    path.write_text("".join(_example(u) + "\n" for u in users), encoding="utf-8")
    return str(path)


def test_planted_iban_is_found_without_leaking_the_value(tmp_path: Path) -> None:
    path = _write(tmp_path / "train.jsonl", [
        "Mijn rekening is [IBAN].",
        f"Gelieve te storten op {PLANTED_IBAN}, dank u.",
    ])
    findings = scan_output_files([path])
    assert findings == [Finding(path, 2, "IBAN")]
    rendered = " ".join(str(f) + repr(f) for f in findings)
    assert "5390" not in rendered and "BE68" not in rendered and "storten" not in rendered


def test_every_validated_type_is_detected(tmp_path: Path) -> None:
    path = _write(tmp_path / "eval.jsonl", [
        "RRN 85.07.30-033.28",
        "mail jan.peeters@example.be",
        "bel +32 478 12 34 56",
        "adres Kerkstraat 12, 9000 Gent",
    ])
    types = [(f.line, f.pii_type) for f in scan_output_files([path])]
    assert types == [(1, "NATIONAL_ID"), (2, "EMAIL"), (3, "PHONE"), (4, "ADDRESS")]


def test_clean_and_rejected_candidates_produce_no_finding(tmp_path: Path) -> None:
    path = _write(tmp_path / "train.jsonl", [
        "[PERSON] belde op [PHONE] over [IBAN].",
        "Ongeldig nummer BE68 5390 0754 7035 en polis 2024 123456.",
    ])
    assert scan_output_files([path]) == []


def test_run_exits_3_and_keeps_outputs_in_staging_when_pii_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.processing.anonymize import engine

    real = engine.anonymise

    def leaky(text: str, ner_model: Any) -> engine.AnonymisationResult:
        # Simulates an anonymiser bug: a valid IBAN slips through every record.
        result = real(text, ner_model)
        return engine.AnonymisationResult(f"{result.text} {PLANTED_IBAN}", result.replaced)

    monkeypatch.setattr("src.processing.pipeline.anonymise", leaky)
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["output"]["dir"] = str(tmp_path / "out")
    raw["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(raw), encoding="utf-8")

    assert run_cli.main(["--config", str(cfg)]) == 3
    staging = tmp_path / "out" / ".staging"
    # A failed scan never destroys the run: everything is kept for inspection...
    assert (staging / "train.jsonl").is_file() and (staging / "eval.jsonl").is_file()
    # ...and nothing is promoted to out/.
    assert not (tmp_path / "out" / "train.jsonl").exists()
