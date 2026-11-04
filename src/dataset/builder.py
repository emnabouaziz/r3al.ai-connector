"""Convert records into the client's chat fine-tuning format."""

import json
import re
from pathlib import Path
from typing import Any

from src.contracts import Record
from src.fileio import write_text_atomic

_WHITESPACE = re.compile(r"\s+")


class BuildError(ValueError):
    """A record cannot be turned into a training example."""


def _split_halves(text: str) -> tuple[str, str]:
    """Cut at the whitespace closest to the middle (FAKE shortcut, see FAKES.md)."""
    mid = len(text) // 2
    cuts = [m.start() for m in _WHITESPACE.finditer(text)]
    if not cuts:
        raise BuildError("text_has_no_split_point")
    cut = min(cuts, key=lambda i: (abs(i - mid), i))
    user, assistant = text[:cut].strip(), text[cut:].strip()
    if not user or not assistant:
        raise BuildError("empty_half")
    return user, assistant


def record_to_example(record: Record, system_prompt: str) -> dict[str, Any]:
    if record.domain is None:
        raise BuildError("record_without_domain")
    user, assistant = _split_halves(record.text)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {
            "domain": record.domain,
            "language": record.language,
            "source_ids": [record.source_id],
        },
    }


def to_chat_format(records: list[Record], system_prompt: str) -> list[dict[str, Any]]:
    """Exact format imposed by the client:
    {"messages":[{"role":"system","content":...},
                 {"role":"user","content":...},
                 {"role":"assistant","content":...}],
     "meta":{"domain":...,"language":...,"source_ids":[...]}}
    For the fake connector: user = first half of the text, assistant =
    second half (recorded in FAKES.md)."""
    return [record_to_example(r, system_prompt) for r in records]


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
    write_text_atomic(path, lines)
