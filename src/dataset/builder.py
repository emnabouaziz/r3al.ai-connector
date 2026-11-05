"""Example construction: dispatch by record KIND, then the client's chat format.

Dispatch uses the record's kind carried in metadata, never its source_type:
  thread -> tickets builder (conversations -> extracted pairs)
  case   -> facts builder (structured case facts -> grounded factual pairs)
  chunk  -> documents builder (reference knowledge -> grounded questions)

Built examples become "example records" (one per Q/A pair) so the existing
splitter and leakage checks work on them unchanged: group_id is the
thread / document, so every pair of a thread lands on the same side.

Output: the exact client schema, one JSON object per line
  {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}],
   "meta":{"domain":...,"language":...,"source_ids":[...]}}
source_ids are the real RawDocument ids the example was built from.
"""

import json
from pathlib import Path
from typing import Any

from src.contracts import Record
from src.dataset.builders import (
    BuildCandidate,
    BuildSettings,
    ChunkBuilder,
    ExampleBuilder,
    FactsBuilder,
    TicketThreadBuilder,
)
from src.dataset.question_gen import QuestionGenerator
from src.dataset.system_prompt import SystemPrompts
from src.fileio import write_text_atomic

KIND_KEY = "kind"
KIND_THREAD = "thread"
KIND_CHUNK = "chunk"
KIND_CASE = "case"
KINDS = (KIND_THREAD, KIND_CASE, KIND_CHUNK)
# Which builder handles which kind, as named in the report.
BUILDER_NAMES = {KIND_THREAD: "tickets", KIND_CHUNK: "documents", KIND_CASE: "facts"}


def record_kind(record: Record) -> str:
    kind = record.metadata.get(KIND_KEY)
    return kind if kind in KINDS else KIND_CHUNK


def make_builders(settings: BuildSettings, generator: QuestionGenerator | None = None) -> dict[str, ExampleBuilder]:
    return {
        KIND_THREAD: TicketThreadBuilder(settings),
        KIND_CASE: FactsBuilder(settings),
        KIND_CHUNK: ChunkBuilder(settings, generator),
    }


def build_record(record: Record, builders: dict[str, ExampleBuilder]) -> tuple[list[BuildCandidate], dict[str, int]]:
    return builders[record_kind(record)].build([record])


def candidate_to_record(candidate: BuildCandidate, example_id: str, kind: str) -> Record:
    return Record(
        id=example_id,
        text=f"{candidate.user}\n\n{candidate.assistant}",  # what dedup and leakage checks compare
        group_id=candidate.group_id,
        source_id=candidate.source_ids[0],
        source_type="example",
        language=candidate.language,
        domain=candidate.domain,
        metadata={"user": candidate.user, "assistant": candidate.assistant,
                  "source_ids": list(candidate.source_ids), KIND_KEY: kind},
    )


def to_chat_line(
    user: str, assistant: str, domain: str, language: str, source_ids: list[str], system_prompt: str
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": {"domain": domain, "language": language, "source_ids": list(source_ids)},
    }


def example_line(example: Record, prompts: SystemPrompts) -> dict[str, Any]:
    if example.domain is None:
        raise ValueError("example without a domain")
    return to_chat_line(
        str(example.metadata["user"]), str(example.metadata["assistant"]), example.domain,
        example.language, list(example.metadata["source_ids"]), prompts.for_language(example.language),
    )


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples)
    write_text_atomic(path, lines)
