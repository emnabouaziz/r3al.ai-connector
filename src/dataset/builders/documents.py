"""Chunks (document sections, CRM rows, any non-thread record) -> grounded Q/A.

THE SINGLE MOST IMPORTANT RULE: the answer to every generated question
must be CONTAINED IN THE CHUNK. A question the chunk cannot answer teaches
the model to hallucinate at training time: it learns to produce a
confident answer it was never shown. So:
  - no anchor (heading, FAQ question, title) -> 'no_question_anchor':
    we reject rather than invent a question;
  - the answer is the chunk's own leading text, and is re-checked
    ('answer_not_grounded': its normalised form must be a substring of the
    normalised chunk) - a guard that also binds any future generator;
  - too little answer text -> 'chunk_too_short';
  - mostly placeholders after anonymisation -> 'placeholder_heavy'.

A CRM claim note and a document chunk take this same path: a customer's
note has no topic anchor and no answer, so it is rejected, not paraphrased.
"""

from src.contracts import Record
from src.dataset.builders.base import BuildCandidate, BuildSettings, ExampleBuilder, RejectionTally, placeholder_share
from src.dataset.question_gen import QuestionGenerator, TemplatedQuestionGenerator
from src.report.ground_truth_scan import normalise


def is_grounded(answer: str, chunk: str) -> bool:
    norm = normalise(answer)
    return bool(norm) and norm in normalise(chunk)


class ChunkBuilder(ExampleBuilder):
    REASONS = ("no_question_anchor", "chunk_too_short", "answer_not_grounded", "placeholder_heavy")

    def __init__(self, settings: BuildSettings, generator: QuestionGenerator | None = None) -> None:
        super().__init__(settings)
        self.generator = generator or TemplatedQuestionGenerator(settings.max_answer_chars)

    def build(self, records: list[Record]) -> tuple[list[BuildCandidate], dict[str, int]]:
        candidates: list[BuildCandidate] = []
        tally = RejectionTally()
        for rec in records:
            pairs = self.generator.generate(rec)
            if not pairs:
                tally.add("no_question_anchor")
                continue
            if rec.domain is None:
                raise ValueError("chunk reached the builder without a domain")
            for question, answer in pairs:
                if not is_grounded(answer, rec.text):
                    tally.add("answer_not_grounded")
                elif len(answer) < self.settings.min_chunk_chars:
                    tally.add("chunk_too_short")
                elif max(placeholder_share(answer), placeholder_share(f"{question}\n{answer}")) > (
                    self.settings.max_placeholder_share
                ):
                    tally.add("placeholder_heavy")
                else:
                    candidates.append(BuildCandidate(
                        user=question, assistant=answer, domain=rec.domain, language=rec.language,
                        source_ids=[rec.source_id], group_id=rec.group_id,
                    ))
        return candidates, tally.counts
