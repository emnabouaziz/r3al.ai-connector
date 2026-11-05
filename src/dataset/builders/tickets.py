"""Ticket threads -> question/answer pairs.

A thread is a sequence of messages with direction inbound / outbound (the
connector keeps them apart with MESSAGE_SEPARATOR; their directions are in
metadata["messages"]). Each outbound reply is paired with the LAST inbound
message before it that has not been answered yet: that inbound message is
`user`, the reply is `assistant`. A thread may yield several pairs; all
carry the thread's group_id, so they stay on one side of the split.

Every candidate that would teach the model something wrong is rejected and
counted - a dataset of 400 good pairs beats 900 where a third are noise.
"""

import re

from src.contracts import Record
from src.dataset.builders.base import BuildCandidate, BuildSettings, ExampleBuilder, RejectionTally, placeholder_share
from src.processing.clean import is_auto_acknowledgement, split_messages

ATTACHMENT_MARKERS = (
    "bijlage", "bijgevoegd", "bijgesloten",
    "ci-joint", "ci-jointe", "en annexe", "pièce jointe", "pièces jointes", "veuillez trouver",
    "attached", "attachment", "enclosed", "please find",
)
REQUEST_MARKERS = (
    "kunt u", "kan u", "wilt u", "gelieve", "bezorg ons", "stuur ons",
    "pouvez-vous", "pourriez-vous", "merci de nous", "veuillez nous",
    "could you", "can you", "would you", "please send", "please provide",
)
_REFERENCE_LINE = re.compile(r"^\s*(?:referentie|référence|reference|ref\.?)\s*:\s*\S+\s*$", re.IGNORECASE)
_GREETING_LINE = re.compile(
    r"^\s*(?:beste|geachte|dear|bonjour|bonsoir|hello|hi|madame|monsieur)\b[^\n]*,\s*$", re.IGNORECASE
)
_SENTENCES = re.compile(r"[^.!?]+[.!?]?")


def strip_reference_lines(text: str) -> str:
    """Drop ticket / mailbox reference lines: ids the model cannot know."""
    return "\n".join(line for line in text.splitlines() if not _REFERENCE_LINE.match(line)).strip()


def _content(answer: str) -> str:
    """The reply without its greeting line."""
    lines = answer.strip().splitlines()
    if lines and _GREETING_LINE.match(lines[0]):
        lines = lines[1:]
    return "\n".join(lines).strip()


def is_only_a_question(answer: str) -> bool:
    """True when every sentence of the reply asks the customer for something."""
    sentences = [s.strip() for s in _SENTENCES.findall(_content(answer)) if s.strip()]
    if not sentences:
        return False
    return all(s.endswith("?") or any(m in s.casefold() for m in REQUEST_MARKERS) for s in sentences)


def refers_to_attachment(answer: str) -> bool:
    folded = answer.casefold()
    return any(marker in folded for marker in ATTACHMENT_MARKERS)


def pair_rejection(question: str, answer: str, s: BuildSettings) -> str | None:
    """Rejection reason for an extracted (customer message, agent reply) pair, or None.

    Shared by every builder that extracts real exchanges (ticket threads, CRM cases).
    """
    if is_auto_acknowledgement(answer):
        return "auto_acknowledgement"
    if refers_to_attachment(answer):
        return "refers_to_attachment"
    if is_only_a_question(answer):
        return "answer_is_a_question"
    if len(_content(answer)) < s.min_answer_chars:
        return "answer_too_short"
    if len(question.strip()) < s.min_question_chars:
        return "question_too_short"
    # Measured on the answer too: it is what the model learns to produce.
    if max(placeholder_share(answer), placeholder_share(f"{question}\n{answer}")) > s.max_placeholder_share:
        return "placeholder_heavy"
    return None


PAIR_REASONS = (
    "auto_acknowledgement", "refers_to_attachment", "answer_is_a_question", "answer_too_short",
    "question_too_short", "placeholder_heavy",
)


class TicketThreadBuilder(ExampleBuilder):
    REASONS = ("no_agent_reply", "no_customer_message", "thread_structure_lost", *PAIR_REASONS)

    def __init__(self, settings: BuildSettings) -> None:
        super().__init__(settings)

    def build(self, records: list[Record]) -> tuple[list[BuildCandidate], dict[str, int]]:
        candidates: list[BuildCandidate] = []
        tally = RejectionTally()
        for rec in records:
            parts = [p.strip() for p in split_messages(rec.text)]
            directions = [str(m.get("direction")) for m in rec.metadata.get("messages", [])]
            if len(parts) != len(directions):
                tally.add("thread_structure_lost")
                continue
            if not any(d == "outbound" and p for p, d in zip(parts, directions)):
                tally.add("no_agent_reply")
                continue
            if rec.domain is None:
                raise ValueError("thread reached the builder without a domain")
            pending: str | None = None
            produced = 0
            for text, direction in zip(parts, directions):
                if not text:
                    continue
                if direction == "inbound":
                    pending = text
                    continue
                if pending is None:
                    continue  # agent follow-up without a new customer message
                question, answer = pending, strip_reference_lines(text)
                pending = None
                produced += 1
                reason = pair_rejection(question, answer, self.settings)
                if reason is not None:
                    tally.add(reason)
                    continue
                candidates.append(BuildCandidate(
                    user=question, assistant=answer, domain=rec.domain, language=rec.language,
                    source_ids=[rec.source_id], group_id=rec.group_id,
                ))
            if produced == 0:
                tally.add("no_customer_message")
        return candidates, tally.counts
