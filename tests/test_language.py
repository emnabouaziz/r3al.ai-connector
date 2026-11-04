import pytest

from src.processing.language import detect_language

DUTCH_200 = (
    "Goedemiddag, ik heb vorige week een schadeaangifte ingediend voor mijn wagen "
    "na een aanrijding op de parking. Kunt u mij laten weten wanneer de expert langskomt "
    "en hoe lang de behandeling van mijn dossier ongeveer zal duren? Alvast bedankt."
)
FRENCH = (
    "Bonjour, j'ai reçu un rappel concernant ma prime d'assurance habitation alors que "
    "la domiciliation est bien active sur mon compte. Pourriez-vous vérifier ma situation ?"
)
ENGLISH = (
    "Hello, I received a reminder about my home insurance premium even though the direct "
    "debit is active on my account. Could you please check my situation and get back to me?"
)
GERMAN = (
    "Guten Tag, ich habe letzte Woche eine Schadensmeldung für mein Auto eingereicht und "
    "warte immer noch auf eine Antwort von Ihrer Versicherung. Bitte melden Sie sich bald."
)


def test_dutch_200_chars_high_confidence() -> None:
    assert len(DUTCH_200) >= 200
    lang, conf = detect_language(DUTCH_200)
    assert lang == "nl"
    assert conf > 0.8


@pytest.mark.parametrize(("text", "expected"), [(FRENCH, "fr"), (ENGLISH, "en")])
def test_french_and_english(text: str, expected: str) -> None:
    lang, conf = detect_language(text)
    assert lang == expected
    assert conf > 0.8


def test_short_text_is_unknown() -> None:
    text = "Bedankt, groet!"
    assert len(text) == 15
    assert detect_language(text) == ("unknown", 0.0)


def test_empty_and_symbol_only_text_do_not_crash() -> None:
    assert detect_language("") == ("unknown", 0.0)
    assert detect_language("1234567890 - 1234567890 / 1234567890 + 1234567890") == (
        "unknown",
        0.0,
    )


def test_mixed_dutch_french_does_not_raise() -> None:
    text = (
        "Beste, ik stuur u de documenten door. Bonjour, je vous envoie les documents "
        "demandés pour mon dossier. Met vriendelijke groeten, cordialement."
    )
    lang, conf = detect_language(text)
    assert lang in {"nl", "fr", "en", "unknown"}
    assert 0.0 <= conf <= 1.0


def test_unsupported_language_is_unknown() -> None:
    assert detect_language(GERMAN) == ("unknown", 0.0)


def test_deterministic() -> None:
    results = {detect_language(DUTCH_200) for _ in range(5)}
    assert len(results) == 1
