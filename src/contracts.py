"""Data contracts shared by every pipeline stage."""

from dataclasses import dataclass, field
from typing import Any, Literal

Domain = Literal["claims", "policies", "billing", "onboarding", "complaints", "compliance"]
Language = Literal["nl", "fr", "en", "unknown"]


@dataclass(frozen=True)
class ItemRef:
    """Pointer to one item inside a source, produced by ``list_items``."""

    source_type: str
    item_id: str  # stable identifier WITHIN the source
    uri: str  # path, URI, or "table:pk"
    hints: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawDocument:
    """Unparsed payload fetched from a source."""

    id: str  # "{source_name}:{item_id}", globally unique
    source_type: str
    source_uri: str
    content: str | bytes
    mime_type: str
    metadata: dict[str, Any]


@dataclass
class Record:
    """Internal unit after parsing. One RawDocument -> 1..n Record."""

    id: str
    text: str
    group_id: str  # thread / document / cluster. Basis of the split.
    source_id: str  # originating RawDocument.id
    source_type: str
    language: Language = "unknown"
    domain: Domain | None = None
    pii_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DropReason:
    """Why a record was removed. Must never carry content or PII."""

    record_id: str
    stage: str  # "ingest" | "parse" | "clean" | "dedup" | "anonymize" | "tag" | "build"
    reason: str  # short snake_case code, NOT a free sentence
    detail: str = ""  # NEVER any content or PII
