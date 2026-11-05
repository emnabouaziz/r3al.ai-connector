"""Email connector + .eml parser on the generated export, and the end-to-end pairing."""

import csv
from collections.abc import Iterator
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import pytest

from src.config import load_config
from src.connectors.email import EmailConfigError, EmailConnector, normalise_subject, thread_group_id
from src.contracts import RawDocument
from src.processing.clean import MESSAGE_SEPARATOR
from src.processing.parsers import THREAD_CSV_MIME, THREAD_EML_MIME
from src.processing.parsers.eml import extract_text
from src.processing.pipeline import run_pipeline
from src.synthetic.tickets import generate_tickets

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ["assurance.example.be"]


@pytest.fixture(scope="module")
def export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tickets")
    generate_tickets(root, seed=42)
    return root


def connector(root: Path) -> EmailConnector:
    c = EmailConnector({"csv_path": str(root / "tickets_export.csv"), "eml_dir": str(root / "emails"),
                        "agent_domains": AGENTS})
    c.name = "support_mailbox"
    return c


@pytest.fixture(scope="module")
def docs(export: Path) -> dict[str, RawDocument]:
    c = connector(export)
    return {ref.item_id: c.fetch(ref) for ref in c.list_items()}


# --- Generation ----------------------------------------------------------------------


def test_generation_is_byte_reproducible(tmp_path: Path, export: Path) -> None:
    generate_tickets(tmp_path, seed=42)
    for path in [export / "tickets_export.csv", *sorted((export / "emails").glob("*.eml"))]:
        twin = tmp_path / path.relative_to(export)
        assert twin.read_bytes() == path.read_bytes(), path.name


def test_export_shape(export: Path) -> None:
    rows = list(csv.DictReader((export / "tickets_export.csv").open(encoding="utf-8", newline="")))
    assert list(rows[0]) == ["ticket_id", "thread_id", "created_at", "channel", "direction",
                             "from_address", "to_address", "subject", "body"]
    assert 100 <= len({r["thread_id"] for r in rows}) <= 130 and 330 <= len(rows) <= 450
    assert {r["direction"] for r in rows} == {"inbound", "outbound"}


# --- Connector -------------------------------------------------------------------------


def test_connection_and_config(tmp_path: Path, export: Path) -> None:
    assert connector(export).test_connection()
    broken = EmailConnector({"csv_path": str(export / "tickets_export.csv"), "eml_dir": str(tmp_path / "none"),
                             "agent_domains": AGENTS})
    assert not broken.test_connection()
    with pytest.raises(EmailConfigError):
        EmailConnector({"csv_path": "x.csv"})


def test_one_item_per_thread_not_per_message(export: Path) -> None:
    items = connector(export).list_items()
    assert isinstance(items, Iterator)
    refs = list(items)
    csv_threads = {r["thread_id"] for r in csv.DictReader((export / "tickets_export.csv").open(encoding="utf-8"))}
    assert sum(r.item_id.startswith("csv:") for r in refs) == len(csv_threads)
    assert sum(r.item_id.startswith("eml:") for r in refs) == 15
    assert len(list((export / "emails").glob("*.eml"))) > 15  # several messages per eml thread


def test_thread_document_keeps_order_and_directions(docs: dict[str, RawDocument]) -> None:
    doc = next(d for d in docs.values() if d.metadata["message_count"] >= 4)
    messages = doc.metadata["messages"]
    assert [m["created_at"] for m in messages] == sorted(m["created_at"] for m in messages)
    assert {m["direction"] for m in messages} == {"inbound", "outbound"}
    assert isinstance(doc.content, str) and doc.content.count(MESSAGE_SEPARATOR) == len(messages) - 1
    assert doc.mime_type in (THREAD_CSV_MIME, THREAD_EML_MIME)
    assert "@" not in str(doc.metadata)  # no addresses in metadata


def test_csv_and_eml_exports_of_a_thread_get_one_group_id(docs: dict[str, RawDocument]) -> None:
    csv_groups = {d.metadata["thread_id"] for k, d in docs.items() if k.startswith("csv:")}
    eml_groups = [d.metadata["thread_id"] for k, d in docs.items() if k.startswith("eml:")]
    assert len(eml_groups) == 15
    assert all(group in csv_groups for group in eml_groups)
    # ...and distinct conversations never share one.
    assert len(csv_groups) == sum(1 for k in docs if k.startswith("csv:"))


def test_group_id_ignores_reply_prefixes_and_trailing_noise() -> None:
    body = "Goedemiddag,\n\nGraag een offerte voor een nieuwe polis voor mijn wagen.\n\n"
    assert normalise_subject("Antw: RE: Fwd:  Offerte  auto") == "offerte auto"
    assert thread_group_id("Offerte auto", body + "Groeten,\nJan") == thread_group_id(
        "Re: Offerte auto", body + "Met vriendelijke groeten,\nJan Peeters\n+32 478 12 34 56"
    )
    assert thread_group_id("Offerte auto", body) != thread_group_id("Offerte woning", body)


def test_thread_without_agent_reply(docs: dict[str, RawDocument]) -> None:
    no_reply = [d for d in docs.values() if not d.metadata["has_agent_reply"] and d.content.strip()
                and "automati" not in d.content]
    assert len(no_reply) >= 2
    assert all({m["direction"] for m in d.metadata["messages"]} == {"inbound"} for d in no_reply)


def test_mojibake_in_csv_is_repaired(export: Path, docs: dict[str, RawDocument]) -> None:
    raw = (export / "tickets_export.csv").read_text(encoding="utf-8")
    assert "Ã" in raw  # the export really is damaged...
    assert not any("Ã" in str(d.content) for d in docs.values())  # ...and repaired on read


# --- .eml parser ----------------------------------------------------------------------------


def _eml(build: Any) -> bytes:
    msg = EmailMessage()
    msg["From"] = "a@example.be"
    msg["Subject"] = "t"
    build(msg)
    return bytes(msg)


def test_html_only_eml_still_yields_text() -> None:
    content = _eml(lambda m: m.set_content(
        "<html><body><p>Beste klant,</p><p>Uw polis &amp; het bijvoegsel zijn verwerkt.<br>Tot snel.</p>"
        "<style>p{}</style></body></html>", subtype="html"))
    assert extract_text(content) == "Beste klant,\n\nUw polis & het bijvoegsel zijn verwerkt.\nTot snel."


def test_plain_part_is_preferred_and_charset_decoded() -> None:
    def build(m: EmailMessage) -> None:
        m.set_content("Je déménage à Liège, merci.", charset="iso-8859-1")
        m.add_alternative("<p>HTML version</p>", subtype="html")
    assert extract_text(_eml(build)) == "Je déménage à Liège, merci."


def test_eml_without_text_part_yields_empty() -> None:
    content = _eml(lambda m: m.set_content(b"%PDF-1.4", maintype="application", subtype="pdf"))
    assert extract_text(content) == ""


def test_generated_html_only_reply_is_read(docs: dict[str, RawDocument]) -> None:
    eml_docs = [d for k, d in docs.items() if k.startswith("eml:")]
    assert all(d.content and all(part.strip() for part in str(d.content).split("\x1e")) for d in eml_docs)


# --- End to end ------------------------------------------------------------------------------


def test_pipeline_pairs_exports_drops_noise_and_never_splits(tmp_path: Path, export: Path) -> None:
    cfg: dict[str, Any] = load_config(str(ROOT / "config.yaml"))
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    cfg["sources"].append({"name": "support_mailbox", "type": "email", "config": {
        "csv_path": str(export / "tickets_export.csv"), "eml_dir": str(export / "emails"),
        "agent_domains": AGENTS}})
    result = run_pipeline(cfg)

    mailbox = result.counters.dropped_per_source["support_mailbox"]
    assert mailbox["near_duplicate"] == 15  # one survivor per CSV/.eml pair
    assert mailbox["auto_acknowledgement"] == 2
    assert mailbox["empty_text"] == 1
    assert result.counters.cleaned["quoted_history"] > 0 and result.counters.cleaned["disclaimer"] > 0

    kept = result.split.train + result.split.eval + result.split.capped_records
    by_group: dict[str, set[str]] = {}
    for rec in kept:
        if rec.source_id.startswith("support_mailbox:"):
            by_group.setdefault(rec.group_id, set()).add(rec.source_id.split(":")[1])
    assert all(exports == {"csv"} or exports == {"eml"} for exports in by_group.values())
    sides: dict[str, set[str]] = {}
    for side, recs in (("train", result.split.train), ("eval", result.split.eval)):
        for rec in recs:
            sides.setdefault(rec.group_id, set()).add(side)
    assert all(len(s) == 1 for s in sides.values())
    assert result.report["invariant"]["holds"] is True
