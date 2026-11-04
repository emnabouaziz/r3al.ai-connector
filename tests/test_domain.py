from pathlib import Path
from typing import Any

import pytest

from src.config import load_taxonomy
from src.processing.domain import classify_domain, normalize_tokens

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "src" / "taxonomy.yaml"


@pytest.fixture(scope="module")
def taxonomy() -> dict[str, Any]:
    return load_taxonomy(str(TAXONOMY_PATH))


# The safety net: one case per domain in each of the 3 languages.
CASES: list[tuple[str, str, str]] = [
    # claims
    ("claims", "nl", "Mijn auto werd aangereden. Ik wil een schadeaangifte doen en het "
                     "aanrijdingsformulier doorsturen."),
    ("claims", "fr", "Suite à l'accident, j'ai rempli le constat amiable. Quand l'expert "
                     "passera-t-il pour évaluer le sinistre ?"),
    ("claims", "en", "My basement was flooded last night. I would like to file a claim; when "
                     "can the loss adjuster come to assess the damage?"),
    # policies
    ("policies", "nl", "Is waterschade gedekt door mijn polis? Ik vind de uitsluitingen niet "
                       "terug in de algemene voorwaarden."),
    ("policies", "fr", "Pouvez-vous me dire si le vol est couvert par ma garantie et quelle "
                       "franchise s'applique selon les conditions générales ?"),
    ("policies", "en", "Does my coverage include storm damage, and which deductible applies "
                       "according to the terms and conditions?"),
    # billing
    ("billing", "nl", "Ik heb een betalingsherinnering ontvangen voor mijn premie, maar de "
                      "domiciliering staat toch actief? Kunt u de factuur controleren?"),
    ("billing", "fr", "J'ai reçu un rappel de paiement pour ma prime alors que la "
                      "domiciliation est active. Pouvez-vous vérifier la facture ?"),
    ("billing", "en", "I received a payment reminder for my premium although the direct debit "
                      "is active. Could you check the invoice?"),
    # onboarding
    ("onboarding", "nl", "Wij verhuizen volgende maand. Graag een adreswijziging en een "
                         "offerte voor een nieuwe polis voor ons appartement."),
    ("onboarding", "fr", "Je souhaite souscrire une assurance habitation : pouvez-vous "
                         "m'envoyer un devis ? Nous avons aussi un changement d'adresse."),
    ("onboarding", "en", "I would like a quote for a new policy for my car, and please "
                         "register my change of address."),
    # complaints
    ("complaints", "nl", "Ik ben zeer ontevreden over de behandeling van mijn dossier en dien "
                         "een klacht in. Indien nodig stap ik naar de ombudsman."),
    ("complaints", "fr", "Je suis très insatisfait du traitement de mon dossier. Je dépose une "
                         "plainte et je contacterai le médiateur si nécessaire."),
    ("complaints", "en", "I am very dissatisfied with how my case was handled. This is a "
                         "formal complaint and I will escalate to the ombudsman if needed."),
    # compliance
    ("compliance", "nl", "Op basis van de AVG wil ik mijn inzagerecht uitoefenen: welke "
                         "persoonsgegevens bewaart u over mij?"),
    ("compliance", "fr", "Conformément au RGPD, je souhaite exercer mon droit d'accès : "
                         "quelles données personnelles conservez-vous à mon sujet ?"),
    ("compliance", "en", "Under the GDPR I want to exercise my right of access: what personal "
                         "data do you hold about me?"),
]


def test_cases_cover_every_domain_and_language() -> None:
    pairs = {(d, lang) for d, lang, _ in CASES}
    assert len(pairs) == 18


@pytest.mark.parametrize(
    ("expected", "lang", "text"), CASES, ids=[f"{d}-{lang}" for d, lang, _ in CASES]
)
def test_domain_per_language(
    taxonomy: dict[str, Any], expected: str, lang: str, text: str
) -> None:
    result = classify_domain(text, taxonomy)
    assert result.domain == expected, result
    assert result.matched_terms


def test_ombudsman_is_complaints(taxonomy: dict[str, Any]) -> None:
    result = classify_domain("I contacted the ombudsman yesterday about my file.", taxonomy)
    assert result.domain == "complaints"
    assert "ombudsman" in result.matched_terms


def test_invoice_and_claim_close_weights_is_ambiguous(taxonomy: dict[str, Any]) -> None:
    result = classify_domain("I received an invoice and I want to file a claim.", taxonomy)
    assert result.domain is None
    assert result.score >= 0.15  # abstained on margin, not on threshold
    assert result.runner_up in {"billing", "claims"}
    assert result.score - result.runner_up_score < 0.05


@pytest.mark.parametrize(
    "text",
    ["team meeting on Tuesday", "The team meeting is moved to Tuesday afternoon, see you there."],
)
def test_off_topic_is_none(taxonomy: dict[str, Any], text: str) -> None:
    result = classify_domain(text, taxonomy)
    assert result.domain is None
    assert result.score == 0.0


def test_long_document_with_weak_hits_abstains(taxonomy: dict[str, Any]) -> None:
    filler = "the quick brown fox jumps over the lazy dog " * 100
    result = classify_domain(filler + " dossier dossier dossier", taxonomy)
    assert result.domain is None
    assert 0.0 < result.score < 0.15


def test_multiword_term_not_double_counted(taxonomy: dict[str, Any]) -> None:
    result = classify_domain("Les conditions générales et particulières.", taxonomy)
    assert result.matched_terms == ["conditions generales"]


def test_normalisation() -> None:
    assert normalize_tokens("Réclamations GÉNÉRALES d’accès") == [
        "reclamation",
        "generale",
        "d",
        "acce",
    ]


def test_deterministic(taxonomy: dict[str, Any]) -> None:
    text = CASES[0][2]
    assert len({repr(classify_domain(text, taxonomy)) for _ in range(5)}) == 1


def test_empty_text(taxonomy: dict[str, Any]) -> None:
    result = classify_domain("", taxonomy)
    assert result.domain is None and result.score == 0.0
