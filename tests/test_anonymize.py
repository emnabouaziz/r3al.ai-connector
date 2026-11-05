"""Two-layer anonymiser: engine behaviour with the real pinned NER model."""

import re
from pathlib import Path
from typing import Any

import pytest

from src.processing.anonymize import AnonymisationResult, NerSpan, anonymise, load_ner
from src.processing.anonymize.ner import SpacyNer
from src.processing.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ner() -> SpacyNer:
    return load_ner("xx_ent_wiki_sm")


class NoNer:
    def find(self, text: str) -> list[NerSpan]:
        return []


def test_model_is_loaded_once() -> None:
    assert load_ner("xx_ent_wiki_sm") is load_ner("xx_ent_wiki_sm")


def test_dutch_claim_note_masks_name_iban_phone_and_keeps_sentence(ner: SpacyNer) -> None:
    text = (
        "Beste, kunt u de vergoeding voor de waterschade storten op rekening BE68 5390 0754 7034? "
        "U kunt mij altijd bellen op +32 478 12 34 56. Met vriendelijke groeten, Jan Peeters."
    )
    result = anonymise(text, ner)
    assert result.text == (
        "Beste, kunt u de vergoeding voor de waterschade storten op rekening [IBAN]? "
        "U kunt mij altijd bellen op [PHONE]. Met vriendelijke groeten, [PERSON]."
    )
    assert result.replaced == {"IBAN": 1, "PERSON": 1, "PHONE": 1}


def test_french_and_english_structured_pii(ner: SpacyNer) -> None:
    text = (
        "Mon numéro de registre national est le 85.07.30-033.28. Écrivez-moi à "
        "sophie.dubois@example.be. Bien à vous, Sophie Dubois, Rue de la Station 12, 5000 Namur."
    )
    result = anonymise(text, ner)
    assert result.text == (
        "Mon numéro de registre national est le [NATIONAL_ID]. Écrivez-moi à "
        "[EMAIL]. Bien à vous, [PERSON], [ADDRESS]."
    )
    landline = anonymise("Phone call with Emma Brown, call back on 02/345.67.89 please.", ner)
    assert landline.text == "Phone call with [PERSON], call back on [PHONE] please."


def test_already_anonymised_text_is_unchanged(ner: SpacyNer) -> None:
    once = anonymise("Groeten, An Janssens (+32 478 12 34 56, BE68 5390 0754 7034).", ner)
    twice = anonymise(once.text, ner)
    assert twice.text == once.text
    assert twice.replaced == {}


def test_placeholders_are_never_corrupted(ner: SpacyNer) -> None:
    text = "[PERSON] belde op [PHONE] over [IBAN] en [ADDRESS]."
    assert anonymise(text, ner).text == text


def test_invalid_iban_is_left_alone_and_counted(ner: SpacyNer) -> None:
    text = "Gelieve te storten op BE68 5390 0754 7035 aub."
    result = anonymise(text, ner)
    assert "BE68 5390 0754 7035" in result.text
    assert "IBAN" not in result.replaced
    assert result.candidates_rejected["IBAN"] == 1


def test_policy_numbers_amounts_and_dates_are_not_phones(ner: SpacyNer) -> None:
    text = "Polis 2024 123456, bedrag 1.234,56 euro, schade van 01/03/2025."
    result = anonymise(text, NoNer())
    assert result.text == text
    assert "PHONE" not in result.replaced


def test_wide_match_is_trimmed_to_the_valid_value() -> None:
    result = anonymise("Rekening BE68 5390 0754 7034 2024 graag.", NoNer())
    assert result.text == "Rekening [IBAN] 2024 graag."


def test_structured_pii_runs_before_ner() -> None:
    class GreedyNer:
        """Pretends the whole text is one PERSON entity."""

        def find(self, text: str) -> list[NerSpan]:
            return [NerSpan(0, len(text), "PERSON")]

    result = anonymise("IBAN BE68 5390 0754 7034 tel +32 478 12 34 56", GreedyNer())
    assert result.replaced["IBAN"] == 1 and result.replaced["PHONE"] == 1
    assert "[IBAN]" in result.text and "[PHONE]" in result.text
    assert not re.search(r"\[[A-Z_]*\[", result.text)


def test_record_header_name_is_masked(ner: SpacyNer) -> None:
    # Regression: the statistical model missed this header in production.
    result = anonymise("Polisinformatie voor Koen Wouters:\n\nDe dekking start op 01/02/2025.", ner)
    assert result.text.startswith("Polisinformatie voor [PERSON]:")


def test_domain_words_are_not_masked_as_persons() -> None:
    # Regression: sentence-initial domain keywords were tagged PER and masked,
    # which made 13 production records unclassifiable.
    ner = load_ner("xx_ent_wiki_sm", frozenset({"aanrijdingsformulier", "ombudsman", "dossier"}))
    text = "Aanrijdingsformulier ontvangen en gecontroleerd. Madame, Monsieur, merci."
    result = anonymise(text, ner)
    assert "Aanrijdingsformulier" in result.text and "Madame" in result.text
    # ...but a span with a real name in it is still masked.
    assert anonymise("Dossier van Lotte Maes ontvangen.", ner).text.count("[PERSON]") == 1


def test_located_entity_without_digit_is_not_masked(ner: SpacyNer) -> None:
    result = anonymise("De storm heeft veel schade veroorzaakt in Gent en Antwerpen.", ner)
    assert "Gent" in result.text and "Antwerpen" in result.text


def test_deterministic(ner: SpacyNer) -> None:
    text = "Telefonisch contact met Lotte Maes: vergoeding naar BE68 5390 0754 7034."
    assert anonymise(text, ner) == anonymise(text, ner)


def test_failing_record_is_dropped_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.config import load_config
    from src.processing.anonymize import engine

    real = engine.anonymise

    def flaky(text: str, ner_model: Any) -> AnonymisationResult:
        if "Leuven" in text:  # a few fake documents mention Leuven
            raise RuntimeError("simulated NER crash")
        return real(text, ner_model)

    monkeypatch.setattr("src.processing.pipeline.anonymise", flaky)
    cfg = load_config(str(ROOT / "config.yaml"))
    cfg["output"]["dir"] = str(tmp_path)
    cfg["processing"]["taxonomy_path"] = str(ROOT / "src" / "taxonomy.yaml")
    result = run_pipeline(cfg)
    failed = result.counters.dropped["anonymize"]["anonymisation_failed"]
    assert failed >= 1
    assert result.report["invariant"]["holds"] is True
    # The failing records never reach the output, not even un-anonymised.
    out = result.outputs["train"].read_text(encoding="utf-8") + result.outputs["eval"].read_text(
        encoding="utf-8"
    )
    assert "Leuven" not in out
