"""E-mail cleaning: each operation removes its block and is counted."""

from pathlib import Path
from typing import Any

from src.config import load_config
from src.processing.clean import MESSAGE_SEPARATOR, clean_text
from src.processing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]

DISCLAIMER_NL = (
    "Dit bericht is vertrouwelijk en uitsluitend bestemd voor de geadresseerde.\n"
    "Indien u dit bericht per vergissing ontving, verwittig dan de afzender\n"
    "en verwijder het bericht onmiddellijk uit uw systeem.\n"
    "Elke verspreiding of kopie zonder voorafgaande toelating is verboden.\n"
    "Verzekeringen NV, Koningsstraat 1, 1000 Brussel, RPR Brussel.\n"
    "Verzekeringsonderneming toegelaten onder codenummer 0000."
)
BODY = "Wij hebben de wijziging verwerkt. U ontvangt binnen vijf werkdagen een bijvoegsel."


def test_quoted_history_is_removed_and_counted() -> None:
    text = (
        f"{BODY}\n\nOp 02/10/2025 09:00 schreef Jan Peeters <jan@example.be>:\n> Graag een adreswijziging.\n>\n"
        "> Groeten, Jan"
    )
    result = clean_text(text)
    assert result.text == BODY
    assert result.operations == {"quoted_history": 1}


def test_french_and_english_attribution_lines_are_removed() -> None:
    for line in ("Le 02/10/2025 09:00, Sophie Dubois <s@example.be> a écrit :",
                 "On 02/10/2025 09:00, Emma Wilson <e@example.be> wrote:"):
        assert clean_text(f"{BODY}\n\n{line}\n> old text").text == BODY


def test_signature_block_is_removed_and_counted() -> None:
    result = clean_text(f"{BODY}\n\nMet vriendelijke groeten,\nInge Vermeulen\nKlantendienst\nVerzekeringen NV")
    assert result.text == BODY
    assert result.operations == {"signature": 1}
    dash = clean_text(f"{BODY}\n\n--\nInge Vermeulen | Klantendienst\nT 02 555 01 23")
    assert dash.text == BODY and dash.operations == {"signature": 1}


def test_inline_sign_off_and_long_tail_are_not_signatures() -> None:
    inline = f"{BODY}\n\nMet vriendelijke groeten, Jan Peeters."
    assert clean_text(inline).text == inline  # CRM-style closer on one line is content
    long_tail = f"Cordialement,\n" + "\n".join(f"ligne {i} du message" for i in range(7))
    assert clean_text(long_tail).operations == {}


def test_disclaimer_is_removed_before_the_signature() -> None:
    outbound = f"Beste klant,\n\n{BODY}\n\nMet vriendelijke groeten,\nInge Vermeulen\nKlantendienst\n\n{DISCLAIMER_NL}"
    result = clean_text(outbound)
    assert result.text == f"Beste klant,\n\n{BODY}"
    assert result.operations == {"disclaimer": 1, "signature": 1}


def test_auto_acknowledgement_alone_is_flagged() -> None:
    ack = "Dit is een automatisch antwoord. Wij hebben uw bericht goed ontvangen.\n\nReferentie: TCK-1"
    result = clean_text(ack)
    # A record made only of automatic replies is flagged: the pipeline drops it.
    assert result.auto_acknowledgement is True
    assert result.operations == {"auto_acknowledgement": 1}


def test_auto_acknowledgement_inside_a_thread_is_kept_for_the_builder() -> None:
    # Inside a thread the auto-reply stays in its slot: the example builder rejects
    # that pair with its own reason ('auto_acknowledgement') instead of the whole thread.
    thread = MESSAGE_SEPARATOR.join([
        "Goedemiddag,\n\nGraag een offerte voor een nieuwe polis.",
        "Ceci est une réponse automatique. Nous avons bien reçu votre message.",
        f"Beste klant,\n\n{BODY}",
    ])
    result = clean_text(thread)
    assert result.auto_acknowledgement is False
    assert len(result.messages) == 3 and "automatique" in result.messages[1]
    assert result.operations == {"auto_acknowledgement": 1}


def test_messages_are_cleaned_one_by_one() -> None:
    # The customer's signature must not swallow the agent reply that follows it.
    thread = MESSAGE_SEPARATOR.join([
        "Goedemiddag,\n\nGraag een offerte.\n\nGroeten,\nJan Peeters\n+32 478 12 34 56",
        f"Beste Jan Peeters,\n\n{BODY}\n\nKind regards,\nHelen Carter\nCustomer service\n\n{DISCLAIMER_NL}",
    ])
    result = clean_text(thread)
    # Message slots are preserved (aligned with their directions in metadata).
    assert result.messages == ["Goedemiddag,\n\nGraag een offerte.", f"Beste Jan Peeters,\n\n{BODY}"]
    assert result.text == MESSAGE_SEPARATOR.join(result.messages)
    assert result.operations == {"disclaimer": 1, "signature": 2}


def test_emptied_message_keeps_its_slot() -> None:
    thread = MESSAGE_SEPARATOR.join(["Graag een offerte voor een nieuwe polis.", "--\nInge\nKlantendienst"])
    assert clean_text(thread).messages == ["Graag een offerte voor een nieuwe polis.", ""]


def test_text_without_email_noise_is_untouched() -> None:
    text = "1. Exclusions\nThe policy does not cover intentional acts.\n\nProduit | Prime\nAuto | 412 EUR"
    result = clean_text(text)
    assert result.text == text and result.operations == {}


def test_too_short_after_clean_is_dropped_with_its_own_reason(tmp_path: Path) -> None:
    csv_path = tmp_path / "tickets.csv"
    csv_path.write_text(
        "ticket_id,thread_id,created_at,channel,direction,from_address,to_address,subject,body\n"
        'TCK-1,THR-1,2025-10-01T09:00:00+02:00,email,inbound,a@example.be,s@assurance.example.be,Vraag,'
        '"Met vriendelijke groeten,\nJan Peeters\n+32 478 12 34 56\nKerkstraat 1, 9000 Gent"\n',
        encoding="utf-8",
    )
    (tmp_path / "emails").mkdir()
    cfg: dict[str, Any] = load_config(str(ROOT / "config.yaml"))
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    cfg["sources"].append({"name": "support_mailbox", "type": "email", "config": {
        "csv_path": str(csv_path), "eml_dir": str(tmp_path / "emails"),
        "agent_domains": ["assurance.example.be"]}})
    result = run_pipeline(cfg)
    assert result.counters.dropped_per_source["support_mailbox"] == {"too_short_after_clean": 1}
    assert result.counters.cleaned["signature"] >= 1
