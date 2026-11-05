"""Files connector: streaming listing, fetch, supersede, and end-to-end pipeline."""

import logging
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from docx import Document

from src.config import load_config
from src.connectors.files import FilesConfigError, FilesConnector
from src.processing.parsers import DOCX_MIME, PDF_MIME
from src.processing.pipeline import run_pipeline
from src.synthetic.documents import generate_documents

ROOT = Path(__file__).resolve().parents[1]


def connector(root: Path, **extra: Any) -> FilesConnector:
    c = FilesConnector({"root": str(root), **extra})
    c.name = "knowledge_docs"
    return c


def _procedure_docx(path: Path, version: int, deadline: int, modified: datetime) -> None:
    doc = Document()
    doc.add_heading(f"Procédure dégât des eaux - version {version}", level=0)
    doc.add_heading("1. Déclarer le sinistre", level=1)
    doc.add_paragraph(
        f"La déclaration de sinistre doit nous parvenir dans les {deadline} jours suivant la découverte "
        "du dégât des eaux. Un expert évalue ensuite les dégâts et l'indemnisation est versée dans les "
        "30 jours suivant l'accord, après déduction de la franchise prévue au contrat."
    )
    doc.save(str(path))
    os.utime(path, (modified.timestamp(), modified.timestamp()))


@pytest.fixture
def same_language_pair(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    root.mkdir()
    _procedure_docx(root / "procedure_water_damage_v2.docx", 2, 5, datetime(2025, 3, 1, tzinfo=timezone.utc))
    _procedure_docx(root / "procedure_water_damage_v3.docx", 3, 8, datetime(2025, 9, 15, tzinfo=timezone.utc))
    return root


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("generated")
    generate_documents(root, seed=42)
    return root


# --- Listing and fetch -------------------------------------------------------------


def test_list_items_returns_a_lazy_iterator(generated: Path) -> None:
    items = connector(generated).list_items()
    assert isinstance(items, Iterator) and not isinstance(items, list)
    first = next(items)
    assert first.uri == f"file:{first.item_id}"
    assert set(first.hints) == {"mtime", "size"}


def test_listing_is_recursive_sorted_and_filtered(tmp_path: Path, generated: Path) -> None:
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "b.pdf").write_bytes((generated / "gdpr_privacy_notice_nl.pdf").read_bytes())
    (root / "sub" / "a.docx").write_bytes((generated / "tariffs_2026.docx").read_bytes())
    (root / "notes.txt").write_text("ignored", encoding="utf-8")
    assert [r.item_id for r in connector(root).list_items()] == ["b.pdf", "sub/a.docx"]


def test_fetch_returns_bytes_mime_and_metadata(generated: Path) -> None:
    c = connector(generated)
    refs = {r.item_id: r for r in c.list_items()}
    pdf = c.fetch(refs["policy_conditions_auto_fr.pdf"])
    docx = c.fetch(refs["tariffs_2026.docx"])
    assert isinstance(pdf.content, bytes) and pdf.mime_type == PDF_MIME and docx.mime_type == DOCX_MIME
    assert pdf.id == "knowledge_docs:policy_conditions_auto_fr.pdf"
    assert pdf.metadata["filename"] == "policy_conditions_auto_fr.pdf"
    assert pdf.metadata["size"] == len(pdf.content)
    assert pdf.metadata["mtime"].startswith("2025-01-15")
    assert pdf.metadata["thread_id"] == "doc:policy_conditions_auto_fr.pdf"


def test_test_connection_and_config_validation(tmp_path: Path) -> None:
    assert connector(tmp_path).test_connection()
    assert not connector(tmp_path / "missing").test_connection()
    with pytest.raises(FilesConfigError):
        FilesConnector({})
    with pytest.raises(FilesConfigError):
        FilesConnector({"root": str(tmp_path), "extensions": [".xlsx"]})


# --- Supersede ------------------------------------------------------------------------


def test_same_language_older_version_is_superseded(same_language_pair: Path) -> None:
    c = connector(same_language_pair)
    docs = {r.item_id: c.fetch(r) for r in c.list_items()}
    old, new = docs["procedure_water_damage_v2.docx"], docs["procedure_water_damage_v3.docx"]
    assert old.metadata["superseded"] is True
    assert old.metadata["superseded_by"] == "procedure_water_damage_v3.docx"
    assert "superseded" not in new.metadata


def test_newer_version_with_older_mtime_is_not_trusted(same_language_pair: Path) -> None:
    v3 = same_language_pair / "procedure_water_damage_v3.docx"
    past = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    os.utime(v3, (past, past))
    c = connector(same_language_pair)
    assert all("superseded" not in c.fetch(r).metadata for r in c.list_items())


def test_cross_language_pair_is_not_matched_and_is_logged(
    generated: Path, caplog: pytest.LogCaptureFixture
) -> None:
    c = connector(generated)
    with caplog.at_level(logging.INFO, logger="src.connectors.files"):
        docs = {r.item_id: c.fetch(r) for r in c.list_items()}
    assert "superseded" not in docs["procedure_water_damage_v2.docx"].metadata
    assert "superseded" not in docs["procedure_waterschade_v3.docx"].metadata
    unmatched = [m for m in caplog.messages if m.startswith("supersede_unmatched")]
    assert len(unmatched) == 2


# --- End to end through the pipeline ---------------------------------------------------


def _config(tmp_path: Path, docs_root: Path) -> dict[str, Any]:
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    cfg["sources"].append({"name": "knowledge_docs", "type": "files", "config": {"root": str(docs_root)}})
    return cfg


def test_pipeline_drops_superseded_and_scanned_documents(tmp_path: Path, same_language_pair: Path,
                                                         generated: Path) -> None:
    for name in ("internal_claims_note_scanned.pdf", "policy_conditions_liability_en.pdf"):
        (same_language_pair / name).write_bytes((generated / name).read_bytes())
    result = run_pipeline(_config(tmp_path, same_language_pair))

    c = result.counters
    assert c.superseded == 1
    assert c.dropped["parse"] == {"superseded": 1, "pdf_no_text_layer": 1}
    assert c.dropped_per_source["knowledge_docs"]["superseded"] == 1
    assert result.report["invariant"]["holds"] is True

    kept = result.split.train + result.split.eval + result.split.capped_records
    sources = {r.source_id for r in kept}
    assert "knowledge_docs:procedure_water_damage_v3.docx" in sources
    assert "knowledge_docs:procedure_water_damage_v2.docx" not in sources


def test_chunks_of_one_document_share_one_group_and_one_side(tmp_path: Path, generated: Path) -> None:
    root = tmp_path / "one"
    root.mkdir()
    name = "policy_conditions_liability_en.pdf"
    (root / name).write_bytes((generated / name).read_bytes())
    result = run_pipeline(_config(tmp_path, root))

    source_id = f"knowledge_docs:{name}"
    sides = {
        side: [r for r in recs if r.source_id == source_id]
        for side, recs in (("train", result.split.train), ("eval", result.split.eval),
                           ("capped", result.split.capped_records))
    }
    chunks = [r for recs in sides.values() for r in recs]
    assert len(chunks) >= 6
    assert {r.group_id for r in chunks} == {f"doc:{name}"}
    assert sum(1 for recs in sides.values() if recs) == 1  # never split across sides
