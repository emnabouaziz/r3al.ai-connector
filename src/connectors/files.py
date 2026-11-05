"""Files connector: PDF / DOCX documents under a root directory.

One file = one RawDocument with the raw bytes; parsing and chunking happen
in the pipeline's parse stage (by mime_type). group_id is the document
(metadata ``thread_id = "doc:<relative path>"``), so every chunk of a
document stays on the same side of the train/eval split.

SUPERSEDE. Two versions of the same procedure with contradictory content
(a 5-day vs an 8-day declaration deadline) must not both be trained on.
A file "<base>_v<N>.<ext>" is marked superseded=True when a sibling
"<base>_v<M>.<ext>" with M > N exists, is not older, and has a similar
title. The pipeline then drops it (reason 'superseded') and increments
Counters.superseded.

Detection is deliberately limited to the SAME base filename, which in
practice means the same language. A French v2 and a Dutch v3 of the same
procedure have different filenames and titles; matching them would require
translation-level similarity that is not reliable, and a loose rule would
silently merge unrelated documents. Such versioned files are logged as
"supersede_unmatched" and the limitation is recorded in FAKES.md.
"""

import difflib
import logging
import os
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.connectors.base import SourceConnector
from src.contracts import ItemRef, RawDocument
from src.processing.parsers import MIME_BY_EXTENSION, get_parser
from src.registry import register

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"^(?P<base>.+?)_v(?P<version>\d+)$", re.IGNORECASE)
_VERSION_WORDS_RE = re.compile(r"\b(?:v|version|versie)\s*\d+\b|\d+", re.IGNORECASE)
DEFAULT_EXTENSIONS = (".pdf", ".docx")
TITLE_SIMILARITY_MIN = 0.8


class FilesConfigError(ValueError):
    """Invalid files connector configuration."""


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")


def _version(path: Path) -> tuple[str, int] | None:
    m = _VERSION_RE.match(path.stem)
    return (m.group("base").lower(), int(m.group("version"))) if m else None


def document_title(path: Path) -> str:
    """First non-empty line of the parsed document, without version numbers."""
    mime = MIME_BY_EXTENSION.get(path.suffix.lower())
    if mime is None:
        return ""
    text = get_parser(mime).extract(path.read_bytes())
    first = next((line.lstrip("#").strip() for line in text.splitlines() if line.strip()), "")
    return _VERSION_WORDS_RE.sub("", first).casefold().strip(" -()")


def title_similarity(a: Path, b: Path) -> float:
    return difflib.SequenceMatcher(None, document_title(a), document_title(b)).ratio()


@register
class FilesConnector(SourceConnector):
    source_type = "files"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        root = config.get("root")
        if not isinstance(root, str) or not root:
            raise FilesConfigError("'root' is required")
        self.root = Path(root)
        extensions = config.get("extensions", list(DEFAULT_EXTENSIONS))
        if not isinstance(extensions, list) or not extensions:
            raise FilesConfigError("'extensions' must be a non-empty list")
        self.extensions = tuple(str(e).lower() for e in extensions)
        # filename -> other versioned files it could not be matched with
        self._unmatched: dict[str, list[str]] = {}
        unknown = [e for e in self.extensions if e not in MIME_BY_EXTENSION]
        if unknown:
            raise FilesConfigError(f"no parser for extension(s): {', '.join(unknown)}")

    def test_connection(self) -> bool:
        ok = self.root.is_dir() and os.access(self.root, os.R_OK)
        if not ok:
            logger.error("files_root_unreadable source=%s", self.name)
        return ok

    def list_items(self) -> Iterator[ItemRef]:
        """Stream files recursively, in a deterministic order (sorted walk)."""
        for dirpath, dirnames, filenames in os.walk(self.root):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(dirpath) / filename
                if path.suffix.lower() not in self.extensions:
                    continue
                rel = path.relative_to(self.root).as_posix()
                stat = path.stat()
                yield ItemRef(
                    source_type=self.source_type,
                    item_id=rel,
                    uri=f"file:{rel}",
                    hints={"mtime": _iso_mtime(path), "size": stat.st_size},
                )

    def _newer_version(self, path: Path) -> Path | None:
        """A newer version of ``path`` in the same directory, or None."""
        own = _version(path)
        if own is None:
            return None
        base, version = own
        candidates = sorted(
            (
                (v[1], sibling)
                for sibling in path.parent.iterdir()
                if sibling.is_file()
                and sibling.suffix.lower() in self.extensions
                and (v := _version(sibling)) is not None
                and v[0] == base
                and v[1] > version
            ),
            reverse=True,
        )
        if not candidates:
            others = sorted(p.name for p in path.parent.iterdir() if p != path and _version(p) is not None)
            self._unmatched[path.name] = others
            logger.info(
                "supersede_unmatched file=%s version=%d other_versioned_files=%s "
                "note=cross-language versions are not matched",
                path.name, version, ",".join(sorted(others)) or "-",
            )
            return None
        for _, newer in candidates:
            if newer.stat().st_mtime < path.stat().st_mtime:
                logger.warning("supersede_rejected file=%s newer=%s reason=older_mtime", path.name, newer.name)
                continue
            similarity = title_similarity(path, newer)
            if similarity < TITLE_SIMILARITY_MIN:
                logger.warning(
                    "supersede_rejected file=%s newer=%s reason=title_mismatch similarity=%.2f",
                    path.name, newer.name, similarity,
                )
                continue
            return newer
        return None

    def fetch(self, ref: ItemRef) -> RawDocument:
        path = self.root / ref.item_id
        content = path.read_bytes()
        metadata: dict[str, Any] = {
            "filename": path.name,
            "mtime": _iso_mtime(path),
            "size": len(content),
            "thread_id": f"doc:{ref.item_id}",
        }
        newer = self._newer_version(path)
        others = self._unmatched.get(path.name, [])
        if others:
            # Escalate instead of guessing a cross-language version match.
            metadata["needs_review"] = [{
                "kind": "supersede_unmatched",
                "items": sorted([path.name, *others]),
                "detail": (
                    "Versioned documents with no same-language counterpart. They may be the same "
                    "procedure in different languages AND versions, with contradictory content (e.g. "
                    "different deadlines). Both are kept; the document owner must say which version "
                    "is current."
                ),
            }]
        if newer is not None:
            metadata["superseded"] = True
            metadata["superseded_by"] = newer.name
            logger.info("document_superseded file=%s by=%s", path.name, newer.name)
        return RawDocument(
            id=f"{self.name}:{ref.item_id}",
            source_type=self.source_type,
            source_uri=ref.uri,
            content=content,
            mime_type=MIME_BY_EXTENSION[path.suffix.lower()],
            metadata=metadata,
        )
