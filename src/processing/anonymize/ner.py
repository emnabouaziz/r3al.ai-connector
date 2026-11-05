"""Layer 2: unstructured PII (person names, located addresses) via NER.

Model choice: spaCy ``xx_ent_wiki_sm`` (see README, "Anonymisation").
One small multilingual model covers Dutch, French and English, runs fully
offline from a pinned wheel, and is deterministic. Presidio was not chosen:
its recognisers sit on top of per-language spaCy models anyway and would
add three large models plus an orchestration layer for no gain here.

TRADE-OFF: RECALL OVER PRECISION.
Masking an innocent word costs a little training signal; leaking a
customer's name into a fine-tuning set is a GDPR incident that cannot be
undone once a model has been trained on it. So this layer is tuned toward
recall:
  - every PER entity is masked, whatever its length or position;
  - rule-based cues add names the statistical model misses: honorifics
    (Mevrouw, Monsieur, Mr ...), letter sign-offs ("Met vriendelijke
    groeten, X Y", "Cordialement, X Y", "Kind regards, X Y"), contact
    phrases ("contact met X Y", "call with X Y") and record headers
    ("Polisinformatie voor X Y:");
  - LOC / GPE spans are masked as [ADDRESS] only when they contain a digit:
    a bare city name ("Gent") is not identifying on its own and masking it
    would destroy meaning in every claim description.

The only precision guard is EDGE TRIMMING. The small multilingual model tags
many capitalised sentence-initial words as PER ("Madame", "Mijn",
"Aanrijdingsformulier"). Leading / trailing words of a PER span that cannot
be names (nl/fr/en stop words, salutations, domain vocabulary passed in by
the caller) are trimmed off; a span disappears only when EVERY word is such
a non-name word. Any other word keeps the span masked, so the guard never
turns an uncertain span into a leak. Without it, masking the domain keywords
themselves made 13 production records unclassifiable.

The model is loaded ONCE per process (``load_ner`` is cached) and reused.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.fr.stop_words import STOP_WORDS as FR_STOP_WORDS
from spacy.lang.nl.stop_words import STOP_WORDS as NL_STOP_WORDS
from spacy.language import Language as SpacyLanguage

from src.processing.domain import normalize_tokens


@dataclass(frozen=True)
class NerSpan:
    start: int
    end: int
    pii_type: str  # "PERSON" or "ADDRESS"


class NerModel(Protocol):
    def find(self, text: str) -> list[NerSpan]: ...


_NAME_WORD = r"[A-ZÀ-Ý][a-zà-ÿ'’-]+"
_PARTICLE = r"(?:van|de|der|den|du|le|la|von|di|d'|De|Van)"
_NAME = rf"{_NAME_WORD}(?:[ \t]+(?:{_PARTICLE}[ \t]+)*{_NAME_WORD}){{0,2}}"

_CUE_RULES: tuple[re.Pattern[str], ...] = (
    # Honorifics, NL / FR / EN.
    re.compile(
        rf"\b(?:Mevrouw|Mevr\.|Meneer|Dhr\.|De heer|Madame|Monsieur|Mme|Mlle|Mr\.?|Mrs\.?|Ms\.?|Dr\.?)[ \t]+(?P<name>{_NAME})"
    ),
    # Letter sign-offs followed by a capitalised name.
    re.compile(
        rf"(?i:groeten|groet|regards|cordialement|bien à vous|salutations|merci d'avance|"
        rf"alvast bedankt|thanks in advance|sincerely),[ \t]*(?P<name>{_NAME})"
    ),
    # Contact phrases.
    re.compile(rf"(?i:contact met|contact avec|call with|contact with)[ \t]+(?P<name>{_NAME})"),
    # Record headers: "Polisinformatie voor X Y:", "Informations de contrat pour X Y :".
    re.compile(rf"(?i:\bvoor|\bpour|\bfor)[ \t]+(?P<name>{_NAME})(?=[ \t]*:)"),
    # Self-identification: "Mijn naam is X Y", "Je m'appelle X Y", "Je soussigné(e) X Y", "My name is X Y".
    re.compile(
        rf"(?i:mijn naam is|ik ben|je m'appelle|je m’appelle|je soussign[ée]e?|soussign[ée]e?|my name is)"
        rf"[ \t]+(?P<name>{_NAME})"
    ),
    # Letter greetings: "Beste X Y,", "Bonjour X Y,", "Dear X Y,".
    re.compile(rf"(?i:\bbeste|\bgeachte|\bdear|\bbonjour|\bhello|\bhi)[ \t]+(?P<name>{_NAME})(?=[ \t]*,)"),
)

_SALUTATIONS = frozenset({
    "beste", "geachte", "mevrouw", "meneer", "heer", "hallo", "goedemiddag", "goedemorgen",
    "groeten", "groet", "vriendelijke", "bedankt", "alvast",
    "bonjour", "bonsoir", "madame", "monsieur", "chère", "cher", "équipe", "salutations",
    "cordialement", "merci",
    "hello", "hi", "dear", "team", "regards", "kind", "best", "thanks", "sincerely",
    # Sentence openers of support replies that the model tags as PER ("Excuses voor de vertraging").
    "excuses", "helaas", "graag", "inderdaad", "uiteraard", "natuurlijk", "ondertussen", "welkom",
    "désolé", "désolée", "malheureusement", "effectivement", "toutes", "bienvenue", "excusez",
    "apologies", "unfortunately", "sorry", "indeed", "regrettably", "welcome",
})


# Capitalised words that are never first names. Kept explicit on purpose: generic
# stop-word lists contain real first names ("An", "Els", "Tom" are Dutch stop words or
# close to them), so they only ever trim LOWERCASE words.
_CAPITALISED_NON_NAMES = frozenset({
    "oui", "non", "ja", "nee", "yes", "no",
    "mijn", "uw", "wij", "ik", "het", "kunt", "kan", "pouvez", "vous", "nous", "votre", "notre",
    "your", "our", "can", "could", "the", "last", "prochain", "dernière",
    "assurances", "assurance", "verzekeringen", "verzekering", "insurance",
})


def _non_name_vocabulary(extra: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    """(any-case vocabulary, capitalised-word vocabulary)."""
    def norm(words: set[str]) -> frozenset[str]:
        return frozenset(tok for w in words for tok in normalize_tokens(w))

    strong = {*_SALUTATIONS, *_CAPITALISED_NON_NAMES, *extra}
    return norm({*NL_STOP_WORDS, *FR_STOP_WORDS, *EN_STOP_WORDS, *strong}), norm(strong)


def rule_based_person_spans(text: str) -> list[NerSpan]:
    spans: list[NerSpan] = []
    for rule in _CUE_RULES:
        for m in rule.finditer(text):
            spans.append(NerSpan(m.start("name"), m.end("name"), "PERSON"))
    return spans


class SpacyNer:
    """spaCy NER + recall-oriented cue rules + non-name edge trimming. Deterministic."""

    def __init__(
        self,
        nlp: SpacyLanguage,
        non_name_words: frozenset[str] = frozenset(),
        capitalised_non_names: frozenset[str] = frozenset(),
    ) -> None:
        self._nlp = nlp
        self._non_name = non_name_words
        self._capitalised_non_name = capitalised_non_names

    def _is_non_name(self, word: str) -> bool:
        # "l'Ombudsman" -> ["l", "ombudsman"]: single letters are elisions.
        tokens = [t for t in normalize_tokens(word) if len(t) > 1]
        if not tokens:
            return True
        core = word.lstrip("'’\"(").split("'")[-1].split("’")[-1]
        vocabulary = self._capitalised_non_name if core[:1].isupper() else self._non_name
        return all(t in vocabulary for t in tokens)

    def _trim(self, text: str, start: int, end: int) -> tuple[int, int] | None:
        words = [(start + m.start(), start + m.end()) for m in re.finditer(r"\S+", text[start:end])]
        while words and self._is_non_name(text[words[0][0]:words[0][1]]):
            words.pop(0)
        while words and self._is_non_name(text[words[-1][0]:words[-1][1]]):
            words.pop()
        # A name always carries a capital: "l'envoi d'offres commerciales" is not a person.
        if not any(text[s:e].lstrip("'’\"(").split("'")[-1][:1].isupper() for s, e in words):
            return None
        return (words[0][0], words[-1][1]) if words else None

    def find(self, text: str) -> list[NerSpan]:
        spans: list[NerSpan] = []
        for ent in self._nlp(text).ents:
            if ent.label_ in ("PER", "PERSON"):
                kept = self._trim(text, ent.start_char, ent.end_char)
                if kept is not None:
                    spans.append(NerSpan(kept[0], kept[1], "PERSON"))
            elif ent.label_ in ("LOC", "GPE") and any(ch.isdigit() for ch in ent.text):
                spans.append(NerSpan(ent.start_char, ent.end_char, "ADDRESS"))
        # Cue spans get the same edge trimming: "Dear Team," must not become [PERSON].
        for cue in rule_based_person_spans(text):
            kept = self._trim(text, cue.start, cue.end)
            if kept is not None:
                spans.append(NerSpan(kept[0], kept[1], "PERSON"))
        return sorted(set(spans), key=lambda s: (s.start, s.end, s.pii_type))


@lru_cache(maxsize=4)
def load_ner(model_name: str, extra_non_name_words: frozenset[str] = frozenset()) -> SpacyNer:
    """Load the pinned spaCy model once per process (per vocabulary). No network access.

    ``extra_non_name_words``: domain vocabulary (e.g. taxonomy terms) that must
    never be masked as a person on its own.
    """
    nlp = spacy.load(model_name)
    any_case, capitalised = _non_name_vocabulary(extra_non_name_words)
    return SpacyNer(nlp, any_case, capitalised)
