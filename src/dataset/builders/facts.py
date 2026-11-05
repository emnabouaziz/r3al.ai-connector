"""CRM cases (structured facts + notes) -> grounded Q/A pairs.

A CRM case is neither a conversation nor reference knowledge: it is a set of
FACTS about one customer's claim, policy or invoice, plus notes. A support
assistant answers exactly those facts ("what is the status of my claim?",
"has my invoice been paid?"), so the fields are the grounding.

B1 - extracted pairs. A note written by the customer followed by a note
     written TO the customer (author_role "agent") is a real exchange,
     treated exactly like a ticket pair and rejected by the same rules.
     Internal notes (author_role "internal") are never an answer.

B2 - templated factual pairs. Question from (case_type, type of the case),
     answer from (status, dates, amounts) - templates keyed by
     (case_type, status, language). For an open claim, a customer-facing
     next-step sentence is chosen from the latest INTERNAL note's category
     (expert appointed, report received...): the note is classified, never
     copied. An internal note is not a customer-facing answer.

Mandatory rejections, each counted:
  no_template_for_state  no template for (case_type, status, language).
                         There is deliberately no catch-all: a template that
                         always fires is the same failure as cutting text in
                         half. A rejected claim, for one, needs a reasoned
                         human answer, so it has no template.
  fields_insufficient    a field the template needs is missing or unknown.
  placeholder_heavy      same threshold as everywhere, on the answer.
  answer_not_grounded    every date, amount and reference in the answer
                         must be a value of the case's fields.

Pure and deterministic: no I/O, no network, sha256-based variant choice.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Final

from src.contracts import Record
from src.dataset.builders.base import BuildCandidate, BuildSettings, ExampleBuilder, RejectionTally, placeholder_share
from src.dataset.builders.tickets import PAIR_REASONS, pair_rejection, strip_reference_lines
from src.processing.clean import split_messages

# --- Display of field values ----------------------------------------------------------

LABELS: Final[dict[str, dict[str, dict[str, str]]]] = {
    "claim_type": {
        "nl": {"car_accident": "aanrijding", "water_damage": "waterschade", "burglary": "inbraak",
               "liability": "burgerlijke aansprakelijkheid", "fire": "brand"},
        "fr": {"car_accident": "accident de la circulation", "water_damage": "dégât des eaux",
               "burglary": "cambriolage", "liability": "responsabilité civile", "fire": "incendie"},
        "en": {"car_accident": "collision", "water_damage": "water damage", "burglary": "burglary",
               "liability": "third-party liability", "fire": "fire"},
    },
    "product": {
        "nl": {"car": "autoverzekering", "home": "woningverzekering", "family_liability": "familiale verzekering"},
        "fr": {"car": "assurance auto", "home": "assurance habitation", "family_liability": "assurance familiale"},
        "en": {"car": "car insurance", "home": "home insurance", "family_liability": "family liability insurance"},
    },
}
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
_DECIMAL = re.compile(r"^-?\d+\.\d{2}$")


def display_fields(fields: dict[str, Any], language: str) -> dict[str, str]:
    """Field values as they may appear in an answer (labels, dd/mm/yyyy, 845,00)."""
    shown: dict[str, str] = {}
    for key, value in fields.items():
        if value is None or str(value).strip() == "":
            continue
        text = str(value)
        if key in LABELS:
            label = LABELS[key].get(language, {}).get(text)
            if label is None:
                continue  # unknown code: not displayable -> the template will be insufficient
            shown[key] = label
        elif m := _ISO_DATE.match(text):
            shown[key] = f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
        elif _DECIMAL.match(text):
            shown[key] = text if language == "en" else text.replace(".", ",")
        else:
            shown[key] = text
    return shown


# --- Templates: (case_type, status, language) -> [(question, answer), ...] -------------

FACT_TEMPLATES: Final[dict[tuple[str, str, str], tuple[tuple[str, str], ...]]] = {
    ("claim", "open", "nl"): (
        ("Wat is de stand van zaken in mijn dossier {claim_number} voor {claim_type}?",
         "Uw schadedossier {claim_number} voor {claim_type} van {incident_date} is geopend en wordt toegewezen aan "
         "een dossierbeheerder."),
        ("Hoe ver staat de behandeling van mijn schadegeval {claim_number} ({claim_type})?",
         "Wij hebben uw dossier {claim_number} ({claim_type}, schade van {incident_date}) geopend; een "
         "dossierbeheerder neemt het binnenkort op."),
    ),
    ("claim", "in_progress", "nl"): (
        ("Wat is de stand van zaken in mijn dossier {claim_number} voor {claim_type}?",
         "Uw schadedossier {claim_number} voor {claim_type} van {incident_date} is in behandeling."),
        ("Hoe ver staat de behandeling van mijn schadegeval {claim_number} ({claim_type})?",
         "Wij behandelen uw dossier {claim_number} ({claim_type}, schade van {incident_date}) op dit moment."),
    ),
    ("claim", "closed", "nl"): (
        ("Wat is de stand van zaken in mijn dossier {claim_number} voor {claim_type}?",
         "Uw schadedossier {claim_number} voor {claim_type} van {incident_date} is afgesloten; er zijn geen "
         "openstaande stappen meer."),
        ("Hoe ver staat de behandeling van mijn schadegeval {claim_number} ({claim_type})?",
         "De behandeling van uw dossier {claim_number} ({claim_type}) is afgerond en het dossier is afgesloten."),
    ),
    ("claim", "open", "fr"): (
        ("Où en est mon dossier sinistre {claim_number} pour {claim_type} ?",
         "Votre dossier sinistre {claim_number} pour {claim_type} du {incident_date} est ouvert et va être "
         "attribué à un gestionnaire."),
        ("Quel est le statut de mon sinistre {claim_number} ({claim_type}) ?",
         "Nous avons ouvert votre dossier {claim_number} ({claim_type}, sinistre du {incident_date}) ; un "
         "gestionnaire le prendra en charge prochainement."),
    ),
    ("claim", "in_progress", "fr"): (
        ("Où en est mon dossier sinistre {claim_number} pour {claim_type} ?",
         "Votre dossier sinistre {claim_number} pour {claim_type} du {incident_date} est en cours de traitement."),
        ("Quel est le statut de mon sinistre {claim_number} ({claim_type}) ?",
         "Nous traitons actuellement votre dossier {claim_number} ({claim_type}, sinistre du {incident_date})."),
    ),
    ("claim", "closed", "fr"): (
        ("Où en est mon dossier sinistre {claim_number} pour {claim_type} ?",
         "Votre dossier sinistre {claim_number} pour {claim_type} du {incident_date} est clôturé ; aucune étape "
         "n'est encore en attente."),
        ("Quel est le statut de mon sinistre {claim_number} ({claim_type}) ?",
         "Le traitement de votre dossier {claim_number} ({claim_type}) est terminé et le dossier est clôturé."),
    ),
    ("claim", "open", "en"): (
        ("What is the status of my claim {claim_number} for {claim_type}?",
         "Your claim {claim_number} for {claim_type} of {incident_date} has been opened and is being assigned to "
         "a claims handler."),
        ("How far along is my claim {claim_number} ({claim_type})?",
         "We have opened your claim {claim_number} ({claim_type}, incident of {incident_date}); a claims handler "
         "will pick it up shortly."),
    ),
    ("claim", "in_progress", "en"): (
        ("What is the status of my claim {claim_number} for {claim_type}?",
         "Your claim {claim_number} for {claim_type} of {incident_date} is being processed."),
        ("How far along is my claim {claim_number} ({claim_type})?",
         "We are currently handling your claim {claim_number} ({claim_type}, incident of {incident_date})."),
    ),
    ("claim", "closed", "en"): (
        ("What is the status of my claim {claim_number} for {claim_type}?",
         "Your claim {claim_number} for {claim_type} of {incident_date} is closed; there are no outstanding "
         "steps."),
        ("How far along is my claim {claim_number} ({claim_type})?",
         "The handling of your claim {claim_number} ({claim_type}) is complete and the claim is closed."),
    ),
    # Policies: variant 1 is a yes/no question, variant 2 a status question.
    ("policy", "active", "nl"): (
        ("Is mijn {product} ({policy_number}) nog actief?",
         "Ja, uw {product} ({policy_number}) loopt sinds {start_date}; de jaarpremie bedraagt {annual_premium} "
         "euro."),
        ("Wat is de stand van mijn {product} met polisnummer {policy_number}?",
         "Uw {product} met polisnummer {policy_number} is actief sinds {start_date}. De jaarpremie bedraagt "
         "{annual_premium} euro."),
    ),
    ("policy", "cancelled", "nl"): (
        ("Is mijn {product} ({policy_number}) nog actief?",
         "Nee, uw {product} ({policy_number}) werd opgezegd; de dekking loopt niet meer."),
        ("Wat is de stand van mijn {product} met polisnummer {policy_number}?",
         "Uw {product} met polisnummer {policy_number} is opgezegd en niet langer actief."),
    ),
    ("policy", "active", "fr"): (
        ("Mon {product} ({policy_number}) est-elle toujours active ?",
         "Oui, votre {product} ({policy_number}) court depuis le {start_date} ; la prime annuelle est de "
         "{annual_premium} euros."),
        ("Quel est le statut de mon {product} n° {policy_number} ?",
         "Votre {product} n° {policy_number} est active depuis le {start_date}. La prime annuelle s'élève à "
         "{annual_premium} euros."),
    ),
    ("policy", "cancelled", "fr"): (
        ("Mon {product} ({policy_number}) est-elle toujours active ?",
         "Non, votre {product} ({policy_number}) a été résiliée ; la couverture a pris fin."),
        ("Quel est le statut de mon {product} n° {policy_number} ?",
         "Votre {product} n° {policy_number} a été résiliée et n'est plus active."),
    ),
    ("policy", "active", "en"): (
        ("Is my {product} ({policy_number}) still active?",
         "Yes, your {product} ({policy_number}) has been running since {start_date}; the annual premium is "
         "{annual_premium} euros."),
        ("What is the status of my {product} with policy number {policy_number}?",
         "Your {product} with policy number {policy_number} has been active since {start_date}. The annual "
         "premium is {annual_premium} euros."),
    ),
    ("policy", "cancelled", "en"): (
        ("Is my {product} ({policy_number}) still active?",
         "No, your {product} ({policy_number}) was cancelled; the cover has ended."),
        ("What is the status of my {product} with policy number {policy_number}?",
         "Your {product} with policy number {policy_number} has been cancelled and is no longer active."),
    ),
    ("invoice", "paid", "nl"): (
        ("Is mijn factuur {invoice_number} al betaald?",
         "Ja, uw factuur {invoice_number} van {issue_date} ten bedrage van {amount} euro is betaald; u hoeft "
         "niets meer te doen."),
        ("Wat is de stand van mijn factuur {invoice_number}?",
         "Uw factuur {invoice_number} van {issue_date} ({amount} euro) is volledig betaald."),
    ),
    ("invoice", "unpaid", "nl"): (
        ("Is mijn factuur {invoice_number} al betaald?",
         "Nog niet: uw factuur {invoice_number} van {amount} euro staat open en vervalt op {due_date}."),
        ("Wat is de stand van mijn factuur {invoice_number}?",
         "Uw factuur {invoice_number} ({amount} euro) staat nog open; de vervaldag is {due_date}."),
    ),
    ("invoice", "overdue", "nl"): (
        ("Is mijn factuur {invoice_number} al betaald?",
         "Nee, uw factuur {invoice_number} van {amount} euro is vervallen sinds {due_date}. Betaal het bedrag zo "
         "snel mogelijk om een aanmaning te vermijden."),
        ("Wat is de stand van mijn factuur {invoice_number}?",
         "Uw factuur {invoice_number} ({amount} euro) is vervallen sinds {due_date}; betaal ze zo snel mogelijk "
         "om een aanmaning te vermijden."),
    ),
    ("invoice", "paid", "fr"): (
        ("Ma facture {invoice_number} est-elle déjà payée ?",
         "Oui, votre facture {invoice_number} du {issue_date} d'un montant de {amount} euros est payée ; vous "
         "n'avez plus rien à faire."),
        ("Quel est le statut de ma facture {invoice_number} ?",
         "Votre facture {invoice_number} du {issue_date} ({amount} euros) est entièrement payée."),
    ),
    ("invoice", "unpaid", "fr"): (
        ("Ma facture {invoice_number} est-elle déjà payée ?",
         "Pas encore : votre facture {invoice_number} de {amount} euros est en attente de paiement, avec une "
         "échéance au {due_date}."),
        ("Quel est le statut de ma facture {invoice_number} ?",
         "Votre facture {invoice_number} ({amount} euros) est encore ouverte ; l'échéance est fixée au "
         "{due_date}."),
    ),
    ("invoice", "overdue", "fr"): (
        ("Ma facture {invoice_number} est-elle déjà payée ?",
         "Non, votre facture {invoice_number} de {amount} euros est échue depuis le {due_date}. Merci de la "
         "régler rapidement pour éviter une mise en demeure."),
        ("Quel est le statut de ma facture {invoice_number} ?",
         "Votre facture {invoice_number} ({amount} euros) est échue depuis le {due_date} ; réglez-la rapidement "
         "pour éviter une mise en demeure."),
    ),
    ("invoice", "paid", "en"): (
        ("Has my invoice {invoice_number} been paid?",
         "Yes, your invoice {invoice_number} of {issue_date} for {amount} euros has been paid; there is nothing "
         "left to do."),
        ("What is the status of my invoice {invoice_number}?",
         "Your invoice {invoice_number} of {issue_date} ({amount} euros) is fully paid."),
    ),
    ("invoice", "unpaid", "en"): (
        ("Has my invoice {invoice_number} been paid?",
         "Not yet: your invoice {invoice_number} for {amount} euros is still open and due on {due_date}."),
        ("What is the status of my invoice {invoice_number}?",
         "Your invoice {invoice_number} ({amount} euros) is still open; the due date is {due_date}."),
    ),
    ("invoice", "overdue", "en"): (
        ("Has my invoice {invoice_number} been paid?",
         "No, your invoice {invoice_number} for {amount} euros has been overdue since {due_date}. Please pay it as "
         "soon as possible to avoid a formal notice."),
        ("What is the status of my invoice {invoice_number}?",
         "Your invoice {invoice_number} ({amount} euros) has been overdue since {due_date}; please pay it soon to "
         "avoid a formal notice."),
    ),
}

# Customer-facing next step, chosen from the CATEGORY of the latest internal note.
NEXT_STEP_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "expert_visit": ("expert aangesteld", "expert désigné", "loss adjuster appointed"),
    "report_received": ("verslag van de expert ontvangen", "rapport d'expertise reçu", "adjuster report received"),
    "compensation_approved": ("goedgekeurd en doorgestuurd", "approuvée et transmise", "approved and sent"),
    "accident_report": ("aanrijdingsformulier ontvangen", "constat amiable reçu", "accident report received"),
    "photos_requested": ("opgevraagd bij de klant", "demandées au client", "requested from the customer"),
}
NEXT_STEPS: Final[dict[str, dict[str, str]]] = {
    "expert_visit": {
        "nl": "Een expert is aangesteld en neemt contact met u op om de schade ter plaatse vast te stellen.",
        "fr": "Un expert a été désigné et vous contactera pour évaluer les dégâts sur place.",
        "en": "A loss adjuster has been appointed and will contact you to assess the damage on site.",
    },
    "report_received": {
        "nl": "Het verslag van de expert is binnen; wij bepalen nu het bedrag van de vergoeding.",
        "fr": "Le rapport d'expertise est arrivé ; nous fixons maintenant le montant de l'indemnisation.",
        "en": "The adjuster's report has arrived; we are now setting the amount of the compensation.",
    },
    "compensation_approved": {
        "nl": "De vergoeding is goedgekeurd en wordt binnenkort op uw rekening gestort.",
        "fr": "L'indemnisation est approuvée et sera bientôt versée sur votre compte.",
        "en": "The compensation has been approved and will be paid into your account shortly.",
    },
    "accident_report": {
        "nl": "Het aanrijdingsformulier is gecontroleerd; wij contacteren de verzekeraar van de tegenpartij.",
        "fr": "Le constat amiable a été vérifié ; nous contactons l'assureur de la partie adverse.",
        "en": "The accident report has been checked; we are contacting the other party's insurer.",
    },
    "photos_requested": {
        "nl": "Wij hebben nog bijkomende foto's van de schade nodig; u kunt ze opladen via het klantenportaal.",
        "fr": "Nous avons encore besoin de photos supplémentaires des dégâts ; vous pouvez les déposer sur le "
              "portail client.",
        "en": "We still need additional photos of the damage; you can upload them through the customer portal.",
    },
}
NEXT_STEP_STATUSES: Final = ("open", "in_progress")

# Tokens an answer may only assert if they are field values.
_FACT_TOKENS = re.compile(r"\b\d{2}/\d{2}/\d{4}\b|\b\d+[.,]\d{2}\b|\b[A-Z]{2,4}-[\dA-Z-]*\d\b")


def next_step_category(note: str) -> str | None:
    folded = note.casefold()
    for category, markers in NEXT_STEP_MARKERS.items():
        if any(marker in folded for marker in markers):
            return category
    return None


def facts_grounded(answer: str, shown: dict[str, str]) -> bool:
    """Every date, amount and reference in the answer is a displayed field value."""
    values = set(shown.values())
    return all(token in values for token in _FACT_TOKENS.findall(answer))


def _placeholders(template: str) -> list[str]:
    return re.findall(r"\{(\w+)\}", template)


@dataclass(frozen=True)
class FactPair:
    question: str
    answer: str


class FactsBuilder(ExampleBuilder):
    REASONS = (
        "thread_structure_lost", *PAIR_REASONS,
        "no_template_for_state", "fields_insufficient", "answer_not_grounded",
    )

    def __init__(
        self,
        settings: BuildSettings,
        templates: dict[tuple[str, str, str], tuple[tuple[str, str], ...]] | None = None,
    ) -> None:
        super().__init__(settings)
        self.templates = FACT_TEMPLATES if templates is None else templates

    # --- B2 -----------------------------------------------------------------
    def factual_pair(self, rec: Record, internal_notes: list[str]) -> FactPair | str:
        """A templated (question, answer) or the rejection reason."""
        fields: dict[str, Any] = dict(rec.metadata.get("fields") or {})
        case_type = str(rec.metadata.get("case_type", ""))
        status = str(fields.get("status") or "")
        variants = self.templates.get((case_type, status, rec.language))
        if not variants:
            return "no_template_for_state"
        pick = int(hashlib.sha256(rec.id.encode("utf-8")).hexdigest(), 16) % len(variants)
        question_template, answer_template = variants[pick]
        shown = display_fields(fields, rec.language)
        if any(name not in shown for name in _placeholders(question_template + answer_template)):
            return "fields_insufficient"
        question = question_template.format(**shown)
        answer = answer_template.format(**shown)
        if case_type == "claim" and status in NEXT_STEP_STATUSES and internal_notes:
            category = next_step_category(internal_notes[-1])
            if category is not None:
                answer = f"{answer} {NEXT_STEPS[category][rec.language]}"
        if not facts_grounded(answer, shown):
            return "answer_not_grounded"
        if placeholder_share(answer) > self.settings.max_placeholder_share:
            return "placeholder_heavy"
        return FactPair(question, answer)

    # --- B1 + B2 --------------------------------------------------------------
    def build(self, records: list[Record]) -> tuple[list[BuildCandidate], dict[str, int]]:
        candidates: list[BuildCandidate] = []
        tally = RejectionTally()
        for rec in records:
            if rec.domain is None:
                raise ValueError("case reached the builder without a domain")
            parts = [p.strip() for p in split_messages(rec.text)]
            roles = [str(n.get("author_role")) for n in rec.metadata.get("notes", [])]
            if len(parts) != len(roles):
                tally.add("thread_structure_lost")
                continue

            def emit(user: str, assistant: str) -> None:
                assert rec.domain is not None
                candidates.append(BuildCandidate(
                    user=user, assistant=assistant, domain=rec.domain, language=rec.language,
                    source_ids=[rec.source_id], group_id=rec.group_id,
                ))

            # B1: customer note followed by a note written TO the customer.
            pending: str | None = None
            for text, role in zip(parts, roles):
                if not text:
                    continue
                if role == "customer":
                    pending = text
                elif role == "agent" and pending is not None:
                    answer = strip_reference_lines(text)
                    reason = pair_rejection(pending, answer, self.settings)
                    if reason is None:
                        emit(pending, answer)
                    else:
                        tally.add(reason)
                    pending = None
                # "internal" / "document" texts never answer anything.

            # B2: one factual pair per case, grounded in its fields.
            internal_notes = [t for t, r in zip(parts, roles) if r == "internal" and t]
            result = self.factual_pair(rec, internal_notes)
            if isinstance(result, str):
                tally.add(result)
            else:
                emit(result.question, result.answer)
        return candidates, tally.counts
