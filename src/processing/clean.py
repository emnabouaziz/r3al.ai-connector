"""E-mail noise removal, applied to every record (no source branching).

A text may hold several messages joined by MESSAGE_SEPARATOR (the email
connector does this for a thread). Each message is cleaned on its own,
because a signature or a disclaimer is defined relative to the END of its
message. Texts without the separator (CRM rows, documents) are one message.

Named, counted operations, in this order:
  1. quoted_history      ">" lines and "On ... wrote:" / "Op ... schreef:" /
                         "Le ... a écrit :" attribution lines
  2. disclaimer          from a known legal-disclaimer opening to the end of
                         the message (removed BEFORE the signature, which is
                         otherwise followed by too many lines to qualify)
  3. signature           from a "--" line or a stand-alone sign-off line
                         ("Met vriendelijke groeten,", "Cordialement,",
                         "Kind regards,") when fewer than 6 lines follow.
                         A sign-off on the same line as text ("Groeten, Jan.")
                         is deliberately NOT a signature block.
  4. auto_acknowledgement  a message that is an automatic reply is counted;
                         inside a thread it is kept (the example builder
                         rejects that pair with its own reason); a record
                         made only of automatic replies is dropped with
                         reason 'auto_acknowledgement'.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

MESSAGE_SEPARATOR = "\n\n\x1e\n\n"
_SEPARATOR_CHAR = "\x1e"
SIGNATURE_MAX_LINES = 6

_ATTRIBUTION_RE = re.compile(
    r"^(?:On\s.+\swrote:|Op\s.+\sschreef.*:|Le\s.+\sa\s+écrit\s*:|-{2,}\s*Original Message\s*-{2,})\s*$",
    re.IGNORECASE,
)
_SIGNOFF_RE = re.compile(
    r"^(?:met\s+vriendelijke\s+groet(?:en)?|vriendelijke\s+groet(?:en)?|hartelijke\s+groeten|groeten"
    r"|cordialement|bien\s+(?:à|a)\s+vous|salutations(?:\s+distinguées)?|bien\s+cordialement"
    r"|kind\s+regards|best\s+regards|regards|sincerely)\s*,?\s*$",
    re.IGNORECASE,
)
DISCLAIMER_OPENINGS = (
    "dit bericht is vertrouwelijk",
    "ce message est confidentiel",
    "this message is confidential",
    "disclaimer:",
)
AUTO_ACK_MARKERS = (
    "dit is een automatisch antwoord",
    "ceci est une réponse automatique",
    "this is an automatic reply",
)


@dataclass
class CleanResult:
    text: str
    operations: dict[str, int] = field(default_factory=dict)  # operation -> messages affected
    auto_acknowledgement: bool = False  # every message was an automatic reply
    messages: list[str] = field(default_factory=list)  # cleaned messages, positions preserved


def split_messages(text: str) -> list[str]:
    return [m.strip("\n") for m in text.split(_SEPARATOR_CHAR)]


def strip_quoted_history(message: str) -> tuple[str, bool]:
    kept = [
        line for line in message.splitlines()
        if not line.lstrip().startswith(">") and not _ATTRIBUTION_RE.match(line.strip())
    ]
    cleaned = "\n".join(kept).strip()
    return cleaned, cleaned != message.strip()


def strip_disclaimer(message: str) -> tuple[str, bool]:
    lines = message.splitlines()
    for i, line in enumerate(lines):
        if line.strip().casefold().startswith(DISCLAIMER_OPENINGS):
            return "\n".join(lines[:i]).strip(), True
    return message.strip(), False


def strip_signature(message: str) -> tuple[str, bool]:
    lines = message.strip().splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("--", "-- ") or _SIGNOFF_RE.match(stripped):
            following = [x for x in lines[i + 1:] if x.strip()]
            if len(following) < SIGNATURE_MAX_LINES:
                return "\n".join(lines[:i]).strip(), True
    return message.strip(), False


def is_auto_acknowledgement(message: str) -> bool:
    folded = message.casefold()
    return any(marker in folded for marker in AUTO_ACK_MARKERS)


def clean_text(text: str) -> CleanResult:
    """Clean each message; message positions are preserved.

    A multi-message text keeps one slot per message (an emptied message
    becomes ""), joined by MESSAGE_SEPARATOR, so the builder can still align
    each message with its direction. Auto-acknowledgements are detected and
    counted; they are kept in a thread (the builder rejects the pair) and the
    record is flagged when it contains nothing else.
    """
    ops: Counter[str] = Counter()
    raw_messages = split_messages(text)
    cleaned: list[str] = []
    auto_acks = 0
    for message in raw_messages:
        if not message.strip():
            cleaned.append("")
            continue
        if is_auto_acknowledgement(message):
            ops["auto_acknowledgement"] += 1
            auto_acks += 1
        for name, step in (
            ("quoted_history", strip_quoted_history),
            ("disclaimer", strip_disclaimer),
            ("signature", strip_signature),
        ):
            message, removed = step(message)
            if removed:
                ops[name] += 1
        cleaned.append(message.strip())
    non_empty = sum(1 for m in raw_messages if m.strip())
    return CleanResult(
        text=MESSAGE_SEPARATOR.join(cleaned) if len(cleaned) > 1 else (cleaned[0] if cleaned else ""),
        operations=dict(sorted(ops.items())),
        auto_acknowledgement=non_empty > 0 and auto_acks == non_empty,
        messages=cleaned,
    )


def visible_text(text: str) -> str:
    """The text without message separators (for length checks)."""
    return "\n\n".join(m for m in split_messages(text) if m.strip()).strip()
