"""Chunks -> grounded Q/A: anchors, grounding, rejections."""

from src.contracts import Domain, Language, Record
from src.dataset.builders import BuildCandidate, BuildSettings, ChunkBuilder
from src.dataset.builders.documents import is_grounded
from src.dataset.question_gen import QuestionGenerator
from src.report.ground_truth_scan import normalise

POLICY_CHUNK = (
    "4. Exclusions\n"
    "The policy does not cover intentional acts, liability arising from the use of motor vehicles, property in "
    "your custody, or professional activities. Anything not excluded is covered within the sum insured.\n"
    "5. Deductible\n"
    "A deductible of 250 euros applies per event for property. No deductible applies to bodily injury."
)


def chunk(text: str, language: Language = "en", domain: Domain = "policies",
          title: str | None = "General terms - Family liability insurance") -> Record:
    metadata = {"document_title": title} if title else {}
    return Record("knowledge_docs:policy_conditions_liability_en.pdf#3", text, "doc:policy.pdf",
                  "knowledge_docs:policy_conditions_liability_en.pdf", "files", language=language,
                  domain=domain, metadata=metadata)


def build(rec: Record, generator: QuestionGenerator | None = None) -> tuple[list[BuildCandidate], dict[str, int]]:
    return ChunkBuilder(BuildSettings(), generator).build([rec])


def test_chunk_with_headings_yields_grounded_questions_per_section() -> None:
    candidates, rejections = build(chunk(POLICY_CHUNK))
    assert rejections == {}
    assert len(candidates) == 2
    exclusions, deductible = candidates
    assert "Exclusions" in exclusions.user and "General terms - Family liability insurance" in exclusions.user
    for c in candidates:
        assert normalise(c.assistant) in normalise(POLICY_CHUNK)  # the answer is IN the chunk
    # An answer never spills into the next section.
    assert "deductible" not in exclusions.assistant.lower()
    assert deductible.assistant.startswith("A deductible of 250 euros")
    assert exclusions.source_ids == ["knowledge_docs:policy_conditions_liability_en.pdf"]


def test_chunk_without_heading_is_rejected_not_invented() -> None:
    crm_note = ("Goedemiddag,\n\nMijn wagen werd gisteren aangereden op de parking. Ik wil graag een "
                "schadeaangifte doen en stuur u het ingevulde aanrijdingsformulier.\n\nMet vriendelijke groeten, "
                "[PERSON].")
    assert build(chunk(crm_note, "nl", "claims", title=None)) == ([], {"no_question_anchor": 1})


def test_field_label_is_not_a_question_anchor() -> None:
    agent_note = ("Interne notitie:\n\nExpert aangesteld voor de vaststelling van de schade, bezoek gepland op "
                  "12/03/2025. Status bijgewerkt in het systeem.")
    assert build(chunk(agent_note, "nl", "claims", title=None))[1] == {"no_question_anchor": 1}


def test_placeholder_heavy_chunk_is_rejected() -> None:
    text = "1. Example\n[PERSON] [ADDRESS] [PHONE] [IBAN] [EMAIL] [PERSON] [NATIONAL_ID] [PERSON] [PHONE] reported."
    assert build(chunk(text))[1] == {"placeholder_heavy": 1}


def test_short_section_is_rejected() -> None:
    assert build(chunk("1. Scope\nCovers the household."))[1] == {"chunk_too_short": 1}


def test_ungrounded_answer_is_rejected_whatever_the_generator() -> None:
    class Hallucinating(QuestionGenerator):
        def generate(self, chunk: Record) -> list[tuple[str, str]]:
            return [("What is the deductible?", "The deductible is 500 euros and includes flood damage at home.")]

    assert build(chunk(POLICY_CHUNK), Hallucinating())[1] == {"answer_not_grounded": 1}
    assert not is_grounded("totally different text", POLICY_CHUNK)
    assert is_grounded("A deductible of 250 euros applies per event", POLICY_CHUNK)


def test_faq_question_is_used_as_is() -> None:
    faq = ("Hoe vraag ik een offerte aan?\nEen offerte voor een nieuwe polis vraagt u online aan of via uw "
           "makelaar. Wij bezorgen u binnen twee werkdagen een prijsofferte.")
    (candidate,), _ = build(chunk(faq, "nl", "onboarding", title="Veelgestelde vragen van klanten"))
    assert candidate.user == "Hoe vraag ik een offerte aan?"
    assert candidate.assistant.startswith("Een offerte voor een nieuwe polis")
