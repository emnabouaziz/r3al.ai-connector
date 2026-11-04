"""Weighted multilingual keyword classification into the 6 support domains.

Pure and deterministic: the taxonomy is passed in, nothing is read here.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Final, get_args

from src.contracts import Domain

KNOWN_DOMAINS: Final[tuple[str, ...]] = get_args(Domain)

# Texts shorter than this many tokens are scored as if they had this length,
# so a two-word snippet cannot reach an inflated score.
LENGTH_FLOOR_TOKENS: Final[int] = 20

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})


def _strip_plural(token: str) -> str:
    """Remove a simple trailing plural 's' (claims -> claim, sinistres -> sinistre)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_tokens(text: str) -> list[str]:
    """Lowercase, strip accents, split on non-alphanumerics, drop simple plurals."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_APOSTROPHES).lower())
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return [_strip_plural(tok) for tok in _TOKEN_RE.findall(ascii_text)]


@dataclass
class DomainScore:
    domain: Domain | None
    score: float
    matched_terms: list[str] = field(default_factory=list)  # for explainability in the report
    runner_up: Domain | None = None
    runner_up_score: float = 0.0


# normalised term tokens -> {domain: (weight, term as written in the taxonomy)}
_TermIndex = dict[tuple[str, ...], dict[str, tuple[float, str]]]


def _build_index(taxonomy: dict[str, Any]) -> tuple[_TermIndex, int]:
    domains = taxonomy.get("domains")
    if not isinstance(domains, dict):
        raise ValueError("taxonomy must contain a 'domains' mapping")

    index: _TermIndex = {}
    for domain, by_language in domains.items():
        if domain not in KNOWN_DOMAINS:
            raise ValueError(f"unknown domain in taxonomy: {domain}")
        if not isinstance(by_language, dict):
            raise ValueError(f"domain '{domain}' must map languages to term weights")
        for terms in by_language.values():
            for term, weight in (terms or {}).items():
                key = tuple(normalize_tokens(str(term)))
                if not key:
                    continue
                per_domain = index.setdefault(key, {})
                # Same normalised term listed in several languages: keep the max weight.
                candidate = (float(weight), str(term))
                previous = per_domain.get(domain)
                if previous is None or candidate > previous:
                    per_domain[domain] = candidate
    max_len = max((len(k) for k in index), default=0)
    return index, max_len


def classify_domain(
    text: str, taxonomy: dict[str, Any], threshold: float = 0.15, margin: float = 0.05
) -> DomainScore:
    """Weighted multilingual keyword scoring.
    Abstention rules, mandatory:
      - best score < threshold           -> domain=None
      - (best - runner_up) < margin      -> domain=None (ambiguous)
    domain=None is NOT an error: the record is dropped downstream with
    reason='domain_unresolved' and counted. A declared gap beats an
    invented label that pollutes per-domain evaluation.

    Matching is greedy longest-first over normalised tokens, so a multi-word
    term consumes its tokens and is not double-counted with its parts.
    Score = sum(weights) / sqrt(max(n_tokens, LENGTH_FLOOR_TOKENS)), which
    keeps long documents from passing the threshold on weak hits alone.
    """
    index, max_len = _build_index(taxonomy)
    tokens = normalize_tokens(text)

    raw: dict[str, float] = {}
    matched: dict[str, set[str]] = {}
    i = 0
    while i < len(tokens):
        for length in range(min(max_len, len(tokens) - i), 0, -1):
            key = tuple(tokens[i : i + length])
            hit = index.get(key)
            if hit is not None:
                for domain, (weight, label) in hit.items():
                    raw[domain] = raw.get(domain, 0.0) + weight
                    matched.setdefault(domain, set()).add(label)
                i += length
                break
        else:
            i += 1

    if not raw:
        return DomainScore(domain=None, score=0.0)

    norm = math.sqrt(max(len(tokens), LENGTH_FLOOR_TOKENS))
    # Sort by score desc, then by domain name for a deterministic tie-break.
    ranked = sorted(((s / norm, d) for d, s in raw.items()), key=lambda x: (-x[0], x[1]))
    best_score, best = ranked[0]
    runner_score, runner = ranked[1] if len(ranked) > 1 else (0.0, None)

    best_domain = _as_domain(best)
    runner_domain = _as_domain(runner) if runner is not None else None
    terms = sorted(matched[best])

    resolved = best_score >= threshold and (best_score - runner_score) >= margin
    return DomainScore(
        domain=best_domain if resolved else None,
        score=round(best_score, 6),
        matched_terms=terms,
        runner_up=runner_domain,
        runner_up_score=round(runner_score, 6),
    )


def _as_domain(name: str) -> Domain:
    for d in KNOWN_DOMAINS:
        if d == name:
            return d  # type: ignore[return-value]
    raise ValueError(f"unknown domain: {name}")
