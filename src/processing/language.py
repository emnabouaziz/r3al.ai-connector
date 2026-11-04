"""Language detection restricted to nl / fr / en."""

import logging
from typing import Final

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

from src.contracts import Language

logger = logging.getLogger(__name__)

# langdetect is probabilistic; a fixed seed makes it deterministic.
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES: Final[frozenset[str]] = frozenset({"nl", "fr", "en"})


def detect_language(text: str, min_chars: int = 40) -> tuple[Language, float]:
    """Return (language, confidence 0..1).
    Below min_chars, return ("unknown", 0.0) without calling the detector.
    Use langdetect with DetectorFactory.seed fixed (determinism).
    Restrict candidates to nl/fr/en; anything else -> "unknown".

    The detector's top candidate is used as-is: if it is not nl/fr/en the
    result is "unknown" with confidence 0.0, rather than promoting a
    lower-ranked supported language.
    """
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "unknown", 0.0

    try:
        candidates = detect_langs(stripped)
    except LangDetectException:
        # No usable features (digits, symbols only, ...). Not an error.
        return "unknown", 0.0

    if not candidates:
        return "unknown", 0.0

    top = candidates[0]
    lang = str(top.lang)
    prob = float(top.prob)
    if lang == "nl":
        return "nl", prob
    if lang == "fr":
        return "fr", prob
    if lang == "en":
        return "en", prob
    return "unknown", 0.0
