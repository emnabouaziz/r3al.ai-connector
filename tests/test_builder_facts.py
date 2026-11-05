"""CRM case facts -> grounded Q/A (B1 extracted pairs, B2 templated factual pairs)."""

from typing import Any

from src.contracts import Domain, Language, Record
from src.dataset.builder import KIND_CASE, make_builders, record_kind
from src.dataset.builders import BuildSettings, FactsBuilder
from src.dataset.builders.facts import FACT_TEMPLATES, display_fields, facts_grounded, next_step_category
from src.processing.clean import MESSAGE_SEPARATOR

CUSTOMER = ("Goedemiddag,\n\nWanneer mag ik de vergoeding voor schadegeval CLM-2024-00013 verwachten? Het gaat "
            "om de waterschade in de keuken.")
AGENT = ("Beste [PERSON],\n\nDe vergoeding wordt uitbetaald zodra het verslag van de expert is goedgekeurd, meestal "
         "binnen tien werkdagen.")
INTERNAL = ("Interne notitie:\n\nVerslag van de expert ontvangen, de schade wordt geraamd op 2400,00 euro. Klant "
            "werd per e-mail geïnformeerd.")
CLAIM_FIELDS = {"claim_number": "CLM-2024-00013", "claim_type": "water_damage", "status": "in_progress",
                "incident_date": "2025-03-12"}


def case(notes: list[tuple[str, str]], fields: dict[str, Any] | None = None, case_type: str = "claim",
         language: Language = "nl", domain: Domain = "claims", record_id: str = "crm_postgres:claims:1#0") -> Record:
    return Record(
        id=record_id,
        text=MESSAGE_SEPARATOR.join(text for _, text in notes),
        group_id="claim:1",
        source_id=record_id.split("#")[0],
        source_type="sql",
        language=language,
        domain=domain,
        metadata={"kind": "case", "case_type": case_type, "fields": dict(fields or CLAIM_FIELDS),
                  "notes": [{"author_role": role, "created_at": None} for role, _ in notes]},
    )


def build(rec: Record, **kwargs: Any) -> tuple[list[Any], dict[str, int]]:
    return FactsBuilder(BuildSettings(), **kwargs).build([rec])


def test_case_kind_dispatches_to_the_facts_builder() -> None:
    rec = case([("customer", CUSTOMER)])
    assert record_kind(rec) == KIND_CASE
    assert isinstance(make_builders(BuildSettings())[record_kind(rec)], FactsBuilder)


def test_customer_note_followed_by_agent_note_yields_a_pair() -> None:
    candidates, rejections = build(case([("customer", CUSTOMER), ("agent", AGENT)]))
    assert rejections == {}
    extracted = [c for c in candidates if c.user == CUSTOMER]
    assert len(extracted) == 1 and extracted[0].assistant == AGENT


def test_only_internal_notes_yield_no_extracted_pair() -> None:
    candidates, _ = build(case([("internal", INTERNAL), ("internal", INTERNAL)]))
    assert all(c.user != INTERNAL and INTERNAL not in c.assistant for c in candidates)
    assert len(candidates) == 1  # only the templated factual pair


def test_in_progress_claim_in_dutch_yields_a_grounded_templated_pair() -> None:
    candidates, rejections = build(case([("internal", INTERNAL)]))
    assert rejections == {}
    (pair,) = candidates
    assert "CLM-2024-00013" in pair.user and "waterschade" in pair.user
    assert "CLM-2024-00013" in pair.assistant and "12/03/2025" in pair.assistant
    assert "behandel" in pair.assistant  # "in behandeling" / "Wij behandelen": the in_progress status
    # Every date / amount / reference in the answer is a field value...
    assert facts_grounded(pair.assistant, display_fields(CLAIM_FIELDS, "nl"))
    # ...and the internal note's own figures never leak into it.
    assert "2400" not in pair.assistant


def test_next_step_is_rewritten_never_copied() -> None:
    (pair,), _ = build(case([("internal", INTERNAL)]))
    assert next_step_category(INTERNAL) == "report_received"
    assert "Het verslag van de expert is binnen" in pair.assistant
    assert "Interne notitie" not in pair.assistant and "geraamd" not in pair.assistant


def test_state_without_template_is_rejected_and_counted() -> None:
    rejected = dict(CLAIM_FIELDS, status="rejected")  # deliberately no template: needs a human answer
    assert build(case([("internal", INTERNAL)], fields=rejected)) == ([], {"no_template_for_state": 1})
    assert ("claim", "rejected", "nl") not in FACT_TEMPLATES
    unknown_language = case([("internal", INTERNAL)], language="unknown")
    assert build(unknown_language)[1] == {"no_template_for_state": 1}


def test_missing_field_is_fields_insufficient() -> None:
    no_date = {k: v for k, v in CLAIM_FIELDS.items() if k != "incident_date"}
    assert build(case([("internal", INTERNAL)], fields=no_date))[1] == {"fields_insufficient": 1}
    unknown_type = dict(CLAIM_FIELDS, claim_type="meteorite")
    assert build(case([("internal", INTERNAL)], fields=unknown_type))[1] == {"fields_insufficient": 1}


def test_answer_asserting_a_date_absent_from_the_fields_is_rejected() -> None:
    lying = {("claim", "in_progress", "nl"): (
        ("Wat is de stand van zaken in mijn dossier {claim_number}?",
         "Uw dossier {claim_number} wordt afgesloten op 30/06/2025."),
    )}
    assert build(case([("internal", INTERNAL)]), templates=lying) == ([], {"answer_not_grounded": 1})


def test_placeholder_heavy_answer_is_rejected() -> None:
    hollow = {("claim", "in_progress", "nl"): (("Vraag {claim_number}?",
                                                  "[PERSON] [PERSON] [ADDRESS] [PHONE] {claim_number} ok"),)}
    assert build(case([("internal", INTERNAL)]), templates=hollow)[1] == {"placeholder_heavy": 1}


def test_all_pairs_of_a_case_share_its_group_id() -> None:
    candidates, _ = build(case([("customer", CUSTOMER), ("agent", AGENT), ("internal", INTERNAL),
                                ("customer", CUSTOMER + " Nog nieuws?"), ("agent", AGENT + " Tot snel.")]))
    assert len(candidates) == 3
    assert {c.group_id for c in candidates} == {"claim:1"}
    assert {tuple(c.source_ids) for c in candidates} == {("crm_postgres:claims:1",)}


def test_extracted_pairs_reuse_the_ticket_rejection_rules() -> None:
    attachment = "Beste [PERSON],\n\nIn bijlage vindt u het verslag van de expert en de berekening van de vergoeding."
    candidates, rejections = build(case([("customer", CUSTOMER), ("agent", attachment)]))
    assert rejections == {"refers_to_attachment": 1} and len(candidates) == 1  # only the factual pair


def test_invoice_and_policy_templates_ground_on_amounts_and_dates() -> None:
    invoice = {"invoice_number": "INV-2025-000063", "status": "overdue", "issue_date": "2025-01-10",
               "due_date": "2025-02-09", "amount": "845.00"}
    (pair,), _ = build(case([("customer", "Ik heb mijn factuur nooit ontvangen, maar nu wel een aanmaning.")],
                            fields=invoice, case_type="invoice", domain="billing",
                            record_id="crm_postgres:invoices:9#0"))
    assert "845,00" in pair.assistant and "09/02/2025" in pair.assistant
    policy = {"policy_number": "POL-100037", "product": "car", "status": "active", "start_date": "2023-09-25",
              "annual_premium": "970.56"}
    (pair,), _ = build(case([("document", "Samenvatting polis POL-100037 (autoverzekering): ...")], fields=policy,
                            case_type="policy", language="en", domain="policies",
                            record_id="crm_postgres:policies:3#0"))
    assert "car insurance" in pair.assistant and "970.56" in pair.assistant and "25/09/2023" in pair.assistant


def test_misaligned_case_is_refused() -> None:
    rec = case([("customer", CUSTOMER), ("agent", AGENT)])
    rec.metadata["notes"] = rec.metadata["notes"][:1]
    assert build(rec) == ([], {"thread_structure_lost": 1})
