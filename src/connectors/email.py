"""Email connector: a ticket-system CSV export AND a mailbox .eml directory.

One ItemRef / RawDocument per THREAD (not per message). The thread's
messages are concatenated in chronological order, separated by
MESSAGE_SEPARATOR so the clean stage can treat each message on its own;
direction / timestamp / channel of every message are kept in metadata
(never addresses: no PII in metadata).

GROUP ID IS CONTENT-DERIVED. The same conversation exported twice (CSV row
ids vs mailbox Message-IDs) must land in ONE group, otherwise it would be
split between train and eval. The id is built from:
  - the normalised subject (Re:/Fwd:/Antw:/TR: prefixes stripped, casefolded);
  - a hash of the OPENING of the first inbound message, after the same
    cleaning the pipeline applies (signature / quote / disclaimer removal)
    and normalisation (casefold, letters and digits only).
Using the opening (GROUP_KEY_CHARS characters) makes the key robust to what
the two exports do differently: signatures, footers, quoted history, HTML.
Only hashes enter the id, so it never carries PII into logs or errors.

Thread reconstruction from .eml: the root of References (or In-Reply-To,
or the message's own Message-ID). Direction: inbound unless the sender's
domain is one of the configured agent domains.
"""

import csv
import hashlib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import ftfy

from src.connectors.base import SourceConnector
from src.contracts import ItemRef, RawDocument
from src.processing.clean import MESSAGE_SEPARATOR, clean_text
from src.processing.parsers import THREAD_CSV_MIME, THREAD_EML_MIME
from src.processing.parsers.eml import message_text, parse_message
from src.registry import register
from src.report.ground_truth_scan import normalise

logger = logging.getLogger(__name__)

GROUP_KEY_CHARS = 300
_SUBJECT_PREFIX_RE = re.compile(r"^\s*(?:re|fwd?|fw|antw|tr|aw|wg|sv)\s*(?:\[\d+\])?\s*:\s*", re.IGNORECASE)
_MESSAGE_ID_RE = re.compile(r"<[^<>\s]+>")
CSV_COLUMNS = ("ticket_id", "thread_id", "created_at", "channel", "direction",
               "from_address", "to_address", "subject", "body")


class EmailConfigError(ValueError):
    """Invalid email connector configuration."""


class BrokenItem(ValueError):
    """An export item that could not be read (dropped by the pipeline)."""


def csv_thread_uri(thread_id: str) -> str:
    return f"email:csv:{thread_id}"


def _short_hash(value: str, n: int) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:n]


def normalise_subject(subject: str) -> str:
    previous = None
    while previous != subject:
        previous, subject = subject, _SUBJECT_PREFIX_RE.sub("", subject)
    return " ".join(subject.split()).casefold()


def thread_group_id(subject: str, first_inbound_body: str) -> str:
    """Content-derived group id, identical for both exports of a thread."""
    opening = normalise(clean_text(ftfy.fix_text(first_inbound_body)).text)[:GROUP_KEY_CHARS]
    return f"mail:{_short_hash(normalise_subject(subject), 8)}-{_short_hash(opening, 12)}"


@dataclass(frozen=True)
class _Message:
    created_at: datetime
    direction: str  # "inbound" | "outbound"
    channel: str
    subject: str
    body: str
    sort_key: str  # deterministic tie-break


@dataclass
class _Thread:
    item_id: str
    mime_type: str
    messages: list[_Message]
    broken: str = ""  # reason when the item could not be read

    @property
    def uri(self) -> str:
        return f"email:{self.item_id}"


@register
class EmailConnector(SourceConnector):
    source_type = "email"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        csv_path, eml_dir = config.get("csv_path"), config.get("eml_dir")
        if not isinstance(csv_path, str) or not isinstance(eml_dir, str) or not csv_path or not eml_dir:
            raise EmailConfigError("'csv_path' and 'eml_dir' are both required")
        self.csv_path, self.eml_dir = Path(csv_path), Path(eml_dir)
        domains = config.get("agent_domains", [])
        if not isinstance(domains, list) or not domains:
            raise EmailConfigError("'agent_domains' must list the insurer's mail domains")
        self.agent_domains = tuple(str(d).lower() for d in domains)
        self._threads: dict[str, _Thread] = {}

    # --- connection ------------------------------------------------------------

    def test_connection(self) -> bool:
        ok = (
            self.csv_path.is_file() and os.access(self.csv_path, os.R_OK)
            and self.eml_dir.is_dir() and os.access(self.eml_dir, os.R_OK)
        )
        if not ok:
            logger.error("email_paths_unreadable source=%s", self.name)
        return ok

    # --- loading ---------------------------------------------------------------

    def _direction(self, sender: str) -> str:
        domain = sender.rsplit("@", 1)[-1].strip("> ").lower()
        return "outbound" if domain in self.agent_domains else "inbound"

    def _csv_threads(self) -> Iterator[_Thread]:
        rows_by_thread: dict[str, list[_Message]] = {}
        broken: list[_Thread] = []
        with self.csv_path.open(encoding="utf-8", newline="") as fh:
            for line_no, row in enumerate(csv.DictReader(fh), start=2):
                try:
                    if not row.get("thread_id"):
                        raise BrokenItem("missing thread_id")
                    message = _Message(
                        created_at=datetime.fromisoformat(row["created_at"]),
                        direction=row["direction"] if row["direction"] in ("inbound", "outbound") else "inbound",
                        channel=row.get("channel") or "email",
                        subject=row.get("subject") or "",
                        body=ftfy.fix_text(row.get("body") or ""),
                        sort_key=row.get("ticket_id") or str(line_no),
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning("email_csv_row_unreadable line=%d error=%s", line_no, type(exc).__name__)
                    broken.append(_Thread(f"csv-broken:{line_no}", THREAD_CSV_MIME, [], type(exc).__name__))
                    continue
                rows_by_thread.setdefault(row["thread_id"], []).append(message)
        for thread_id in sorted(rows_by_thread):
            yield _Thread(f"csv:{thread_id}", THREAD_CSV_MIME, rows_by_thread[thread_id])
        yield from broken

    def _eml_threads(self) -> Iterator[_Thread]:
        by_root: dict[str, list[_Message]] = {}
        broken: list[_Thread] = []
        for path in sorted(self.eml_dir.glob("*.eml")):
            try:
                message = parse_message(path.read_bytes())
                own_id = str(message.get("Message-ID", "")).strip() or f"<{path.name}>"
                references = _MESSAGE_ID_RE.findall(str(message.get("References", "")))
                in_reply_to = _MESSAGE_ID_RE.findall(str(message.get("In-Reply-To", "")))
                root = (references or in_reply_to or [own_id])[0]
                created = parsedate_to_datetime(str(message["Date"]))
                by_root.setdefault(root, []).append(
                    _Message(
                        created_at=created,
                        direction=self._direction(str(message.get("From", ""))),
                        channel="email",
                        subject=str(message.get("Subject", "")),
                        body=message_text(message),
                        sort_key=own_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the listing
                logger.warning("email_eml_unreadable file=%s error=%s", path.name, type(exc).__name__)
                broken.append(_Thread(f"eml-broken:{path.name}", THREAD_EML_MIME, [], type(exc).__name__))
        for root in sorted(by_root):
            yield _Thread(f"eml:{_short_hash(root, 12)}", THREAD_EML_MIME, by_root[root])
        yield from broken

    def _load(self) -> None:
        self._threads = {}
        for thread in (*self._csv_threads(), *self._eml_threads()):
            thread.messages.sort(key=lambda m: (m.created_at.astimezone(timezone.utc), m.sort_key))
            self._threads[thread.item_id] = thread

    # --- interface -------------------------------------------------------------

    def list_items(self) -> Iterator[ItemRef]:
        """One ItemRef per thread (CSV threads, then .eml threads)."""
        self._load()
        for item_id, thread in self._threads.items():
            yield ItemRef(
                source_type=self.source_type,
                item_id=item_id,
                uri=thread.uri,
                hints={"messages": len(thread.messages), "export": item_id.split(":", 1)[0]},
            )

    def fetch(self, ref: ItemRef) -> RawDocument:
        if ref.item_id not in self._threads:
            self._load()
        thread = self._threads[ref.item_id]
        if thread.broken:
            raise BrokenItem(thread.broken)
        messages = thread.messages
        first_inbound = next((m for m in messages if m.direction == "inbound"), messages[0])
        group_id = thread_group_id(messages[0].subject, first_inbound.body)
        return RawDocument(
            id=f"{self.name}:{ref.item_id}",
            source_type=self.source_type,
            source_uri=thread.uri,
            content=MESSAGE_SEPARATOR.join(m.body for m in messages),
            mime_type=thread.mime_type,
            metadata={
                "thread_id": group_id,
                "kind": "thread",  # builder dispatch: inbound/outbound messages -> Q/A pairs
                "export": ref.item_id.split(":", 1)[0],
                "message_count": len(messages),
                "has_agent_reply": any(m.direction == "outbound" for m in messages),
                "messages": [
                    {"direction": m.direction, "created_at": m.created_at.isoformat(), "channel": m.channel}
                    for m in messages
                ],
            },
        )
