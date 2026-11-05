"""Templated question generation: local, deterministic, grounded."""

import ast
from pathlib import Path

from src.contracts import Language, Record
from src.dataset import question_gen
from src.dataset.question_gen import (
    QuestionGenerator,
    TemplatedQuestionGenerator,
    document_title,
    find_sections,
    leading_text,
)
from src.dataset.system_prompt import SystemPromptError, SystemPrompts, from_config

ROOT = Path(__file__).resolve().parents[1]


def rec(text: str, language: Language = "nl", title: str | None = None) -> Record:
    return Record("k:doc#0", text, "g", "k:doc", "files", language=language, domain="policies",
                  metadata={"document_title": title} if title else {})


def test_templates_exist_for_each_language_and_use_the_heading() -> None:
    text = "2. Waarborg brand\nDe waarborg brand dekt brand, ontploffing, blikseminslag en rook."
    for lang, expected in (("nl", "Waarborg brand"), ("fr", "Waarborg brand"), ("en", "Waarborg brand")):
        ((question, answer),) = TemplatedQuestionGenerator().generate(rec(text, lang))  # type: ignore[arg-type]
        assert expected in question and answer.startswith("De waarborg brand dekt")
    ((q_fr, _),) = TemplatedQuestionGenerator().generate(rec(text, "fr"))
    assert q_fr.endswith(" ?")  # French typography


def test_document_title_gives_the_question_its_context() -> None:
    text = "1. Purpose\nThis policy sets the retention periods for personal data processed by the company."
    ((question, _),) = TemplatedQuestionGenerator().generate(rec(text, "en", title="Data retention policy (GDPR)"))
    assert "Data retention policy (GDPR)" in question and "Purpose" in question


def test_generation_is_deterministic() -> None:
    text = "3. Exclusions\nUitgesloten zijn opzettelijke daden en slijtage door gebrek aan onderhoud."
    assert TemplatedQuestionGenerator().generate(rec(text)) == TemplatedQuestionGenerator().generate(rec(text))


def test_no_anchor_gives_no_question() -> None:
    assert TemplatedQuestionGenerator().generate(rec("Goedemiddag,\n\nIk heb een vraag over mijn factuur.")) == []
    assert TemplatedQuestionGenerator().generate(rec("2. Scope\nText.", language="unknown")) == []


def test_sections_never_use_labels_or_placeholders_as_anchors() -> None:
    assert find_sections("Interne notitie:\nExpert aangesteld, bezoek gepland.") == []
    assert find_sections("Polisinformatie voor [PERSON]\n\nDe waarborg brand is inbegrepen.") == []


def test_leading_text_is_a_bounded_substring() -> None:
    body = "Eerste zin over de dekking. Tweede zin over de vrijstelling. Derde zin over uitsluitingen.\nNieuwe lijn."
    answer = leading_text(body, 60)
    assert answer in body and len(answer) <= 60 and answer.endswith(".")


def test_document_title_detection() -> None:
    assert document_title("Algemene voorwaarden - Woningverzekering\n1. Voorwerp") == (
        "Algemene voorwaarden - Woningverzekering")
    assert document_title("Goedemiddag,\n\nIk heb een vraag.") is None


def test_llm_extension_point_needs_no_pipeline_change() -> None:
    class FutureLocalModel(QuestionGenerator):
        def generate(self, chunk: Record) -> list[tuple[str, str]]:
            return [("Q?", chunk.text)]

    assert FutureLocalModel().generate(rec("x")) == [("Q?", "x")]


def test_question_generation_makes_no_network_call() -> None:
    tree = ast.parse(Path(question_gen.__file__).read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert not imported & {"requests", "httpx", "urllib", "http", "socket", "anthropic", "openai"}


def test_system_prompt_language_and_insurer_from_config() -> None:
    prompts = from_config({"insurer_name": "Example Insurance SA"})
    assert "Example Insurance SA" in prompts.for_language("fr") and "français" in prompts.for_language("fr")
    assert "Nederlands" in prompts.for_language("nl")
    english = SystemPrompts("Example Insurance SA", mode="english")
    assert english.for_language("nl") == english.for_language("en")
    for bad in ({}, {"insurer_name": "X", "system_prompt_language": "klingon"}):
        try:
            from_config(bad)
        except SystemPromptError:
            continue
        raise AssertionError(f"accepted {bad}")
