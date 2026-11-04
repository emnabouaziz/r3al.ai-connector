"""Deterministic synthetic CRM data (customers, policies, claims, notes, invoices).

All people, numbers and addresses are invented. Free texts are written in
Dutch / French / English (the customer's language) and deliberately contain
fake PII (names, phone numbers, e-mails, IBANs) so that the anonymisation
stage can be exercised later.

Each free text = opener + core + detail + closer, picked from per-language
pools. A (core, detail, closer) combination is never reused within a pool,
so two texts of different groups never become near-duplicates (which would
trip the blocking cross-split leakage check).
"""

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Final

Pool = dict[str, tuple[str, ...]]

# --- Text pools ---------------------------------------------------------------

_CUSTOMER_OPENERS: Final[dict[str, tuple[str, ...]]] = {
    "nl": ("Goedemiddag,", "Beste,", "Geachte mevrouw, geachte heer,", "Hallo,"),
    "fr": ("Bonjour,", "Madame, Monsieur,", "Bonsoir,", "Chère équipe,"),
    "en": ("Hello,", "Dear team,", "Good afternoon,", "Hi,"),
}
_CUSTOMER_CLOSERS: Final[dict[str, tuple[str, ...]]] = {
    "nl": (
        "Met vriendelijke groeten, {name}.",
        "Alvast bedankt, {name} ({phone}).",
        "Groeten, {name}. U kunt mij bereiken via {email}.",
        "Vriendelijke groet, {name}.",
    ),
    "fr": (
        "Cordialement, {name}.",
        "Merci d'avance, {name} ({phone}).",
        "Bien à vous, {name}. Vous pouvez me joindre à {email}.",
        "Salutations, {name}.",
    ),
    "en": (
        "Kind regards, {name}.",
        "Thanks in advance, {name} ({phone}).",
        "Best regards, {name}. You can reach me at {email}.",
        "Regards, {name}.",
    ),
}

POOLS: Final[dict[str, dict[str, Pool]]] = {
    "nl": {
        "claim_description": {
            "cores": (
                "Op {date} werd mijn wagen met nummerplaat {plate} aangereden in {city}. Ik wil hiervoor een schadeaangifte indienen.",
                "Na het onweer van {date} hebben we waterschade in de woonkamer. Ik meld dit schadegeval onder polis {policy_no}.",
                "Bij een inbraak op {date} werd onze woning in {city} beschadigd en werden spullen gestolen. Graag open ik een schadedossier.",
                "Mijn zoon heeft op {date} per ongeluk de bril van een klasgenoot gebroken. Ik wil deze schade aangeven op onze familiale polis {policy_no}.",
                "Op {date} is er brand ontstaan in onze keuken. De schade aan de kasten en het plafond is aanzienlijk.",
            ),
            "details": (
                "Het ingevulde aanrijdingsformulier en foto's van de schade voeg ik toe.",
                "Kan er een expert langskomen om de schade vast te stellen?",
                "Ik schat de herstelling op ongeveer {amount} euro.",
                "Graag had ik geweten welke documenten u nog nodig hebt voor de vergoeding.",
                "De politie heeft een proces-verbaal opgemaakt, dat ik meestuur bij deze aangifte.",
            ),
        },
        "claim_note_customer": {
            "cores": (
                "Ik stuur u bijkomende foto's van de schade voor dossier {claim_no}.",
                "Wanneer mag ik de vergoeding voor schadegeval {claim_no} verwachten?",
                "Het bestek van de garage voor de herstelling bedraagt {amount} euro, zie bijlage.",
                "De expert is langsgeweest op {date}, maar ik heb zijn verslag nog niet ontvangen.",
                "Kunt u de vergoeding voor de schade storten op rekening {iban}?",
            ),
            "details": (
                "Laat mij weten of u nog iets nodig hebt om het schadedossier af te ronden.",
                "De herstelling kan pas starten na uw akkoord.",
                "Ik ben bereikbaar op {phone} voor verdere vragen over de schade.",
                "Het gaat nog steeds om de schade van {date}.",
                "Bedankt voor de snelle opvolging van mijn aangifte.",
            ),
        },
        "claim_note_agent": {
            "openers": (
                "Notitie dossierbeheerder:",
                "Telefonisch contact met {name}:",
                "Interne notitie:",
                "Opvolging schadedossier {claim_no}:",
            ),
            "cores": (
                "Expert aangesteld voor de vaststelling van de schade, bezoek gepland op {date}.",
                "Verslag van de expert ontvangen, de schade wordt geraamd op {amount} euro.",
                "Vergoeding van {amount} euro goedgekeurd en doorgestuurd voor uitbetaling.",
                "Aanrijdingsformulier ontvangen en gecontroleerd, aansprakelijkheid bij de tegenpartij.",
                "Bijkomende foto's van de schade opgevraagd bij de klant.",
            ),
            "details": (
                "De klant bevestigt dat er geen andere verzekeraar betrokken is.",
                "Herstelling mag starten na akkoord van de expert.",
                "Klant vraagt een snelle afhandeling van het schadegeval.",
                "Proces-verbaal van de politie toegevoegd aan het dossier.",
                "Tegenpartij werd aangeschreven voor terugvordering van de schade.",
            ),
            "closers": (
                "Volgende opvolging gepland op {date}.",
                "Klant werd per e-mail geïnformeerd.",
                "Status bijgewerkt in het systeem.",
                "Dossier blijft open tot ontvangst van de documenten.",
            ),
        },
        "complaint_note": {
            "cores": (
                "Ik ben zeer ontevreden over de behandeling van mijn dossier {claim_no}. Al weken krijg ik geen antwoord.",
                "Dit is onaanvaardbaar: niemand belt mij terug, ondanks mijn berichten. Ik dien hierbij formeel een klacht in.",
                "Ik ben teleurgesteld in uw dienstverlening en vraag een escalatie naar een verantwoordelijke.",
                "Als dit niet snel opgelost wordt, neem ik contact op met de Ombudsman van de Verzekeringen.",
                "Uw beslissing over mijn dossier is voor mij onbegrijpelijk en ik wil hiertegen bezwaar maken.",
            ),
            "details": (
                "Dit is al de derde keer dat ik u hierover contacteer.",
                "Ik verwacht binnen de week een duidelijk antwoord.",
                "Ik overweeg om naar een andere verzekeraar over te stappen.",
                "Uw klantendienst verwijst mij telkens door zonder oplossing.",
                "Ik ben hierover echt ontevreden en ontgoocheld.",
            ),
        },
        "policy_coverage": {
            "openers": (
                "Samenvatting polis {policy_no} ({product}):",
                "Polisinformatie voor {name}:",
                "Overzicht waarborgen {product}:",
                "Toelichting bij polis {policy_no}:",
            ),
            "cores": (
                "De waarborg brand en waterschade is inbegrepen, met een vrijstelling van {amount} euro per gebeurtenis.",
                "Diefstal is gedekt tot een verzekerd bedrag van {amount} euro, mits melding bij de politie binnen 24 uur.",
                "Uitsluitingen: slijtage, gebrek aan onderhoud en opzettelijke handelingen.",
                "De waarborg rechtsbijstand geldt in heel Europa volgens de bijzondere voorwaarden.",
                "Bestuurders jonger dan 25 jaar hebben een verhoogde vrijstelling van {amount} euro.",
            ),
            "details": (
                "De dekking start op {date}.",
                "Natuurrampen vallen onder een aparte waarborg.",
                "Bijstand na pech is niet gedekt in deze formule.",
                "Het verzekerd bedrag wordt jaarlijks geïndexeerd.",
                "Er geldt geen vrijstelling voor glasbreuk.",
            ),
            "closers": (
                "Laatste wijziging van de voorwaarden: {date}.",
                "Zie de algemene voorwaarden voor de volledige tekst.",
                "Vragen over de dekking kunnen gesteld worden via {email}.",
                "Deze samenvatting vervangt de polisdocumenten niet.",
            ),
        },
        "invoice_note": {
            "cores": (
                "Ik heb een betalingsherinnering ontvangen voor factuur {invoice_no}, terwijl de domiciliering actief is.",
                "Het bedrag op factuur {invoice_no} ({amount} euro) is hoger dan vorig jaar. Waarom is de premie gestegen?",
                "Ik heb factuur {invoice_no} per vergissing twee keer betaald. Graag een terugbetaling op {iban}.",
                "Kan ik de premie voor polis {policy_no} via een maandelijkse domiciliering betalen?",
                "Ik heb factuur {invoice_no} nooit ontvangen, maar nu krijg ik al een aanmaning.",
            ),
            "details": (
                "Ik betaalde op {date} via overschrijving.",
                "Kunt u mij een duplicaat van de factuur bezorgen?",
                "Mijn rekeningnummer is {iban}.",
                "Graag een gedetailleerd overzicht van de premie.",
                "Ik wil vermijden dat er extra kosten worden aangerekend.",
            ),
        },
    },
    "fr": {
        "claim_description": {
            "cores": (
                "Le {date}, mon véhicule immatriculé {plate} a été percuté à {city}. Je souhaite déclarer ce sinistre.",
                "Suite à l'orage du {date}, nous avons un dégât des eaux dans le salon. Je déclare ce sinistre sous la police {policy_no}.",
                "Lors d'un cambriolage le {date}, notre maison à {city} a été endommagée et des objets ont été volés. Je souhaite ouvrir un dossier sinistre.",
                "Le {date}, mon fils a cassé par accident les lunettes d'un camarade de classe. Je souhaite déclarer ce dégât sur notre assurance familiale.",
                "Le {date}, un incendie s'est déclaré dans notre cuisine. Les dégâts aux armoires et au plafond sont importants.",
            ),
            "details": (
                "Je joins le constat amiable rempli et des photos des dégâts.",
                "Un expert peut-il passer pour évaluer les dégâts ?",
                "J'estime la réparation à environ {amount} euros.",
                "Pourriez-vous m'indiquer les documents nécessaires pour l'indemnisation ?",
                "La police a dressé un procès-verbal, que je joins à cette déclaration de sinistre.",
            ),
        },
        "claim_note_customer": {
            "cores": (
                "Je vous envoie des photos supplémentaires des dégâts pour le dossier {claim_no}.",
                "Quand puis-je espérer l'indemnisation pour le sinistre {claim_no} ?",
                "L'estimation du garage pour la réparation des dégâts s'élève à {amount} euros.",
                "L'expert est passé le {date}, mais je n'ai pas encore reçu son rapport.",
                "Pouvez-vous verser l'indemnité sur le compte {iban} ?",
            ),
            "details": (
                "Dites-moi s'il vous manque encore quelque chose pour clôturer le dossier sinistre.",
                "La réparation ne peut commencer qu'après votre accord.",
                "Je reste joignable au {phone} pour toute question sur le sinistre.",
                "Il s'agit toujours du sinistre du {date}.",
                "Merci pour le suivi rapide de ma déclaration.",
            ),
        },
        "claim_note_agent": {
            "openers": (
                "Note du gestionnaire :",
                "Contact téléphonique avec {name} :",
                "Note interne :",
                "Suivi du dossier sinistre {claim_no} :",
            ),
            "cores": (
                "Expert désigné pour l'évaluation des dégâts, visite prévue le {date}.",
                "Rapport d'expertise reçu, les dégâts sont estimés à {amount} euros.",
                "Indemnisation de {amount} euros approuvée et transmise pour versement.",
                "Constat amiable reçu et vérifié, responsabilité de la partie adverse.",
                "Photos supplémentaires des dégâts demandées au client.",
            ),
            "details": (
                "Le client confirme qu'aucun autre assureur n'est impliqué.",
                "La réparation peut commencer après accord de l'expert.",
                "Le client demande un traitement rapide du sinistre.",
                "Procès-verbal de la police ajouté au dossier.",
                "La partie adverse a été contactée pour récupérer le coût des dégâts.",
            ),
            "closers": (
                "Prochain suivi prévu le {date}.",
                "Client informé par e-mail.",
                "Statut mis à jour dans le système.",
                "Dossier ouvert jusqu'à réception des documents.",
            ),
        },
        "complaint_note": {
            "cores": (
                "Je suis très insatisfait du traitement de mon dossier {claim_no}. Depuis des semaines, je n'ai aucune réponse.",
                "C'est inacceptable : personne ne me rappelle malgré mes messages. Je dépose donc une plainte formelle.",
                "Je suis déçu par votre service et je demande une escalade vers un responsable.",
                "Si ce n'est pas résolu rapidement, je contacterai l'Ombudsman des Assurances.",
                "Votre décision concernant mon dossier est incompréhensible et je souhaite introduire une réclamation.",
            ),
            "details": (
                "C'est déjà la troisième fois que je vous contacte à ce sujet.",
                "J'attends une réponse claire dans la semaine.",
                "J'envisage de passer chez un autre assureur.",
                "Votre service clientèle me renvoie à chaque fois sans solution.",
                "Je suis vraiment mécontent de cette situation.",
            ),
        },
        "policy_coverage": {
            "openers": (
                "Résumé de la police {policy_no} ({product}) :",
                "Informations de contrat pour {name} :",
                "Aperçu des garanties {product} :",
                "Explication de la police {policy_no} :",
            ),
            "cores": (
                "La garantie incendie et dégâts des eaux est incluse, avec une franchise de {amount} euros par événement.",
                "Le vol est couvert jusqu'à un capital assuré de {amount} euros, à condition de prévenir la police dans les 24 heures.",
                "Exclusions : usure, défaut d'entretien et actes intentionnels.",
                "La garantie protection juridique s'applique dans toute l'Europe selon les conditions particulières.",
                "Les conducteurs de moins de 25 ans ont une franchise majorée de {amount} euros.",
            ),
            "details": (
                "La couverture prend cours le {date}.",
                "Les catastrophes naturelles relèvent d'une garantie séparée.",
                "L'assistance dépannage n'est pas couverte dans cette formule.",
                "Le capital assuré est indexé chaque année.",
                "Aucune franchise ne s'applique au bris de vitrage.",
            ),
            "closers": (
                "Dernière modification des conditions : {date}.",
                "Voir les conditions générales pour le texte complet.",
                "Les questions sur la couverture peuvent être envoyées à {email}.",
                "Ce résumé ne remplace pas les documents contractuels.",
            ),
        },
        "invoice_note": {
            "cores": (
                "J'ai reçu un rappel de paiement pour la facture {invoice_no}, alors que la domiciliation est active.",
                "Le montant de la facture {invoice_no} ({amount} euros) est plus élevé que l'an dernier. Pourquoi la prime a-t-elle augmenté ?",
                "J'ai payé deux fois la facture {invoice_no} par erreur. Merci de procéder au remboursement sur {iban}.",
                "Puis-je payer la prime de la police {policy_no} par domiciliation mensuelle ?",
                "Je n'ai jamais reçu la facture {invoice_no}, mais je reçois déjà une mise en demeure.",
            ),
            "details": (
                "J'ai effectué un virement le {date}.",
                "Pouvez-vous m'envoyer un duplicata de la facture ?",
                "Mon numéro de compte est {iban}.",
                "Je souhaiterais un détail de la prime.",
                "Je voudrais éviter des frais supplémentaires.",
            ),
        },
    },
    "en": {
        "claim_description": {
            "cores": (
                "On {date} my car with registration {plate} was hit in {city}. I would like to file a claim.",
                "After the storm on {date} we have water damage in the living room. I am reporting this claim under policy {policy_no}.",
                "During a burglary on {date} our house in {city} was damaged and items were stolen. I would like to open a claim.",
                "On {date} my son accidentally broke a classmate's glasses. I would like to report this damage on our family insurance.",
                "On {date} a fire broke out in our kitchen. The damage to the cupboards and ceiling is significant.",
            ),
            "details": (
                "I am attaching the completed accident report and photos of the damage.",
                "Can an expert come by to assess the damage?",
                "I estimate the repair at around {amount} euros.",
                "Could you tell me which documents you still need for the compensation?",
                "The police drew up a report, which I am attaching to this claim.",
            ),
        },
        "claim_note_customer": {
            "cores": (
                "I am sending you additional photos of the damage for claim {claim_no}.",
                "When can I expect the compensation for claim {claim_no}?",
                "The garage estimate for repairing the damage is {amount} euros.",
                "The loss adjuster visited on {date}, but I have not received the report yet.",
                "Could you transfer the compensation to account {iban}?",
            ),
            "details": (
                "Let me know if you need anything else to close the claim.",
                "The repair can only start after your approval.",
                "I can be reached on {phone} for further questions about the damage.",
                "This still concerns the damage of {date}.",
                "Thank you for following up on my claim so quickly.",
            ),
        },
        "claim_note_agent": {
            "openers": (
                "Claims handler note:",
                "Phone call with {name}:",
                "Internal note:",
                "Follow-up on claim {claim_no}:",
            ),
            "cores": (
                "Loss adjuster appointed to assess the damage, visit scheduled for {date}.",
                "Adjuster report received, the damage is estimated at {amount} euros.",
                "Compensation of {amount} euros approved and sent for settlement.",
                "Accident report received and checked, liability lies with the other party.",
                "Additional photos of the damage requested from the customer.",
            ),
            "details": (
                "The customer confirms no other insurer is involved.",
                "Repair may start after the adjuster's approval.",
                "Customer asks for quick handling of the claim.",
                "Police report added to the claim file.",
                "The other party has been contacted to recover the damage costs.",
            ),
            "closers": (
                "Next follow-up scheduled for {date}.",
                "Customer informed by email.",
                "Status updated in the system.",
                "File stays open until the documents are received.",
            ),
        },
        "complaint_note": {
            "cores": (
                "I am very dissatisfied with how my file {claim_no} has been handled. I have had no answer for weeks.",
                "This is unacceptable: nobody calls me back despite my messages. I hereby file a formal complaint.",
                "I am disappointed with your service and I ask for an escalation to a manager.",
                "If this is not resolved soon, I will contact the Insurance Ombudsman.",
                "Your decision on my file makes no sense to me and I want to lodge a complaint.",
            ),
            "details": (
                "This is the third time I have contacted you about this.",
                "I expect a clear answer within the week.",
                "I am considering moving to another insurer.",
                "Your customer service keeps redirecting me without a solution.",
                "I am really unhappy about this situation.",
            ),
        },
        "policy_coverage": {
            "openers": (
                "Summary of policy {policy_no} ({product}):",
                "Contract information for {name}:",
                "Overview of {product} coverage:",
                "Explanation of policy {policy_no}:",
            ),
            "cores": (
                "Fire and water damage cover is included, with a deductible of {amount} euros per event.",
                "Theft is covered up to a sum insured of {amount} euros, provided the police are notified within 24 hours.",
                "Exclusions: wear and tear, lack of maintenance and intentional acts.",
                "Legal assistance coverage applies throughout Europe according to the special conditions.",
                "Drivers under 25 have an increased deductible of {amount} euros.",
            ),
            "details": (
                "Coverage starts on {date}.",
                "Natural disasters fall under a separate cover.",
                "Breakdown assistance is not covered in this plan.",
                "The sum insured is indexed every year.",
                "No deductible applies to glass breakage.",
            ),
            "closers": (
                "Last change to the conditions: {date}.",
                "See the general terms for the full text.",
                "Questions about the coverage can be sent to {email}.",
                "This summary does not replace the contract documents.",
            ),
        },
        "invoice_note": {
            "cores": (
                "I received a payment reminder for invoice {invoice_no}, even though the direct debit is active.",
                "The amount on invoice {invoice_no} ({amount} euros) is higher than last year. Why has the premium increased?",
                "I paid invoice {invoice_no} twice by mistake. Please refund the money to {iban}.",
                "Can I pay the premium for policy {policy_no} by monthly direct debit?",
                "I never received invoice {invoice_no}, but I am already getting an overdue notice.",
            ),
            "details": (
                "I made a bank transfer on {date}.",
                "Could you send me a copy of the invoice?",
                "My account number is {iban}.",
                "I would like a breakdown of the premium.",
                "I would like to avoid any extra charges.",
            ),
        },
    },
}

# Domain each pool is written for (used by tests and FAKES.md, not by the pipeline).
POOL_DOMAINS: Final[dict[str, str]] = {
    "claim_description": "claims",
    "claim_note_customer": "claims",
    "claim_note_agent": "claims",
    "complaint_note": "complaints",
    "policy_coverage": "policies",
    "invoice_note": "billing",
}

_FIRST_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "nl": ("Jan", "An", "Pieter", "Lotte", "Wim", "Els", "Tom", "Sofie", "Bart", "Lien", "Koen", "Nele"),
    "fr": ("Sophie", "Marc", "Julie", "Pierre", "Camille", "Nicolas", "Claire", "Antoine", "Manon", "Luc"),
    "en": ("Emma", "James", "Olivia", "Daniel", "Sarah", "Thomas", "Grace", "Oliver"),
}
_LAST_NAMES: Final[dict[str, tuple[str, ...]]] = {
    "nl": ("Peeters", "Janssens", "Claes", "Maes", "Jacobs", "Willems", "Mertens", "Wouters", "Goossens", "De Smet"),
    "fr": ("Dubois", "Lambert", "Martin", "Leroy", "Dupont", "Lemaire", "Renard", "Simon", "Laurent", "Fontaine"),
    "en": ("Wilson", "Brown", "Smith", "Taylor", "Clarke", "Evans", "Walker", "Hughes"),
}
_CITIES: Final[dict[str, tuple[str, ...]]] = {
    "nl": ("Gent", "Antwerpen", "Leuven", "Brugge", "Hasselt", "Mechelen"),
    "fr": ("Namur", "Liège", "Charleroi", "Mons", "Wavre", "Arlon"),
    "en": ("Brussels", "Brussels", "Leuven", "Waterloo"),
}
_PRODUCT_LABELS: Final[dict[str, dict[str, str]]] = {
    "nl": {"car": "autoverzekering", "home": "woningverzekering", "family_liability": "familiale verzekering"},
    "fr": {"car": "assurance auto", "home": "assurance habitation", "family_liability": "assurance familiale"},
    "en": {"car": "car insurance", "home": "home insurance", "family_liability": "family liability insurance"},
}
_LANGUAGE_WEIGHTS: Final[tuple[tuple[str, int], ...]] = (("nl", 50), ("fr", 35), ("en", 15))


# --- Rows ---------------------------------------------------------------------


@dataclass(frozen=True)
class Customer:
    customer_id: int
    first_name: str
    last_name: str
    email: str
    phone: str
    national_number: str
    iban: str
    language: str
    city: str
    created_at: date


@dataclass(frozen=True)
class Policy:
    policy_id: int
    policy_number: str
    customer_id: int
    product: str
    start_date: date
    annual_premium: Decimal
    status: str
    coverage_summary: str | None


@dataclass(frozen=True)
class Claim:
    claim_id: int
    claim_number: str
    policy_id: int
    incident_date: date
    status: str
    description: str


@dataclass(frozen=True)
class ClaimNote:
    note_id: int
    claim_id: int
    author_type: str
    created_at: datetime
    body: str


@dataclass(frozen=True)
class Invoice:
    invoice_id: int
    invoice_number: str
    policy_id: int
    issue_date: date
    due_date: date
    amount: Decimal
    status: str
    customer_note: str | None


@dataclass
class CrmDataset:
    customers: list[Customer] = field(default_factory=list)
    policies: list[Policy] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    claim_notes: list[ClaimNote] = field(default_factory=list)
    invoices: list[Invoice] = field(default_factory=list)
    # "table:id" -> (pool name, language): ground truth for tests only.
    text_origin: dict[str, tuple[str, str]] = field(default_factory=dict)


# --- Generation ---------------------------------------------------------------


class PoolExhausted(RuntimeError):
    """More texts requested from a pool than it has unique combinations."""


class _Composer:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.used: dict[tuple[str, str], set[tuple[int, int, int]]] = {}

    def compose(self, pool_name: str, lang: str, slots: dict[str, str]) -> str:
        pool = POOLS[lang][pool_name]
        openers = pool.get("openers", _CUSTOMER_OPENERS[lang])
        closers = pool.get("closers", _CUSTOMER_CLOSERS[lang])
        cores, details = pool["cores"], pool["details"]
        used = self.used.setdefault((pool_name, lang), set())
        free = [
            (c, d, k)
            for c in range(len(cores))
            for d in range(len(details))
            for k in range(len(closers))
            if (c, d, k) not in used
        ]
        if not free:
            raise PoolExhausted(f"pool {pool_name}/{lang} has no unused combination left")
        c, d, k = self.rng.choice(free)
        used.add((c, d, k))
        opener = self.rng.choice(openers)
        text = f"{opener}\n\n{cores[c]} {details[d]}\n\n{closers[k]}"
        return text.format_map(slots)


def _weighted_language(rng: random.Random) -> str:
    langs, weights = zip(*_LANGUAGE_WEIGHTS)
    choice: str = rng.choices(langs, weights=weights)[0]
    return choice


def _fmt_date(d: date) -> str:
    return d.strftime("%d/%m/%Y")


def _fmt_amount(rng: random.Random, low: int, high: int) -> str:
    return f"{rng.randint(low, high)},{rng.randint(0, 99):02d}"


def generate_crm(
    seed: int = 42,
    n_customers: int = 200,
    n_claims: int = 150,
    complaint_share: float = 0.3,
    n_policy_summaries: int = 90,
    n_invoice_notes: int = 110,
) -> CrmDataset:
    rng = random.Random(seed)
    composer = _Composer(rng)
    ds = CrmDataset()
    epoch = date(2024, 1, 1)

    # Customers.
    for cid in range(1, n_customers + 1):
        lang = _weighted_language(rng)
        first, last = rng.choice(_FIRST_NAMES[lang]), rng.choice(_LAST_NAMES[lang])
        email_local = f"{first}.{last}".lower().replace(" ", "")
        ds.customers.append(
            Customer(
                customer_id=cid,
                first_name=first,
                last_name=last,
                email=f"{email_local}{cid}@example.be",
                phone=f"+32 4{rng.randint(70, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
                national_number=(
                    f"{rng.randint(50, 99):02d}.{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}"
                    f"-{rng.randint(100, 999)}.{rng.randint(10, 99)}"
                ),
                iban="BE{:02d} {:04d} {:04d} {:04d}".format(
                    rng.randint(10, 99), rng.randint(0, 9999), rng.randint(0, 9999), rng.randint(0, 9999)
                ),
                language=lang,
                city=rng.choice(_CITIES[lang]),
                created_at=epoch - timedelta(days=rng.randint(30, 2000)),
            )
        )
    customers = {c.customer_id: c for c in ds.customers}

    def slots_for(cust: Customer, **extra: str) -> dict[str, str]:
        base = {
            "name": f"{cust.first_name} {cust.last_name}",
            "phone": cust.phone,
            "email": cust.email,
            "iban": cust.iban,
            "city": cust.city,
            "date": _fmt_date(epoch + timedelta(days=rng.randint(0, 600))),
            "amount": _fmt_amount(rng, 150, 9000),
            "plate": f"{rng.randint(1, 2)}-{''.join(rng.choices('ABCDEFGHJKLMNPRSTVWXYZ', k=3))}-{rng.randint(100, 999)}",
            "policy_no": "",
            "claim_no": "",
            "invoice_no": "",
            "product": "",
        }
        base.update(extra)
        return base

    # Policies (1-2 per customer).
    pid = 0
    for cust in ds.customers:
        for _ in range(rng.choice((1, 1, 2))):
            pid += 1
            ds.policies.append(
                Policy(
                    policy_id=pid,
                    policy_number=f"POL-{100000 + pid * 37:06d}",
                    customer_id=cust.customer_id,
                    product=rng.choice(("car", "home", "family_liability")),
                    start_date=epoch - timedelta(days=rng.randint(0, 1500)),
                    annual_premium=Decimal(_fmt_amount(rng, 180, 1600).replace(",", ".")),
                    status=rng.choices(("active", "cancelled"), weights=(9, 1))[0],
                    coverage_summary=None,
                )
            )

    def policy_slots(pol: Policy) -> tuple[Customer, dict[str, str]]:
        cust = customers[pol.customer_id]
        product = _PRODUCT_LABELS[cust.language][pol.product]
        return cust, slots_for(cust, policy_no=pol.policy_number, product=product)

    # Coverage summaries on a subset of policies.
    summary_ids = set(rng.sample([p.policy_id for p in ds.policies], n_policy_summaries))
    for i, pol in enumerate(ds.policies):
        if pol.policy_id in summary_ids:
            cust, slots = policy_slots(pol)
            text = composer.compose("policy_coverage", cust.language, slots)
            ds.policies[i] = Policy(**{**pol.__dict__, "coverage_summary": text})
            ds.text_origin[f"policies:{pol.policy_id}"] = ("policy_coverage", cust.language)

    # Claims and their note threads.
    note_id = 0
    for claim_id in range(1, n_claims + 1):
        pol = rng.choice(ds.policies)
        cust, _ = policy_slots(pol)
        lang = cust.language
        claim_no = f"CLM-{2024}-{claim_id * 13:05d}"
        incident = epoch + timedelta(days=rng.randint(0, 600))
        is_complaint = rng.random() < complaint_share

        def claim_slots() -> dict[str, str]:
            s = slots_for(cust, policy_no=pol.policy_number, claim_no=claim_no)
            s["date"] = _fmt_date(incident)
            return s

        ds.claims.append(
            Claim(
                claim_id=claim_id,
                claim_number=claim_no,
                policy_id=pol.policy_id,
                incident_date=incident,
                status=rng.choice(("open", "in_review", "closed")),
                description=composer.compose("claim_description", lang, claim_slots()),
            )
        )
        ds.text_origin[f"claims:{claim_id}"] = ("claim_description", lang)

        # Complaint thread: 2 complaint notes, so the thread's majority is "complaints".
        # Regular thread: 0-1 customer follow-up + 0-2 agent notes.
        if is_complaint:
            plan = [("customer", "complaint_note"), ("customer", "complaint_note")]
        else:
            plan = [("customer", "claim_note_customer")] * rng.choice((0, 1, 1))
            plan += [("agent", "claim_note_agent")] * rng.choice((0, 1, 2))
        when = datetime.combine(incident, datetime.min.time()) + timedelta(hours=9)
        for author, pool_name in plan:
            note_id += 1
            when += timedelta(days=rng.randint(1, 20), minutes=rng.randint(0, 600))
            body = composer.compose(pool_name, lang, slots_for(cust, policy_no=pol.policy_number, claim_no=claim_no))
            ds.claim_notes.append(ClaimNote(note_id, claim_id, author, when, body))
            ds.text_origin[f"claim_notes:{note_id}"] = (pool_name, lang)

    # Invoices (1-2 per policy), customer notes on a subset.
    iid = 0
    for pol in ds.policies:
        for _ in range(rng.choice((1, 2))):
            iid += 1
            issue = epoch + timedelta(days=rng.randint(0, 600))
            ds.invoices.append(
                Invoice(
                    invoice_id=iid,
                    invoice_number=f"INV-{issue.year}-{iid * 7:06d}",
                    policy_id=pol.policy_id,
                    issue_date=issue,
                    due_date=issue + timedelta(days=30),
                    amount=pol.annual_premium,
                    status=rng.choices(("paid", "unpaid", "overdue"), weights=(7, 2, 1))[0],
                    customer_note=None,
                )
            )
    note_invoice_ids = set(rng.sample([i.invoice_id for i in ds.invoices], n_invoice_notes))
    policies_by_id = {p.policy_id: p for p in ds.policies}
    for i, inv in enumerate(ds.invoices):
        if inv.invoice_id in note_invoice_ids:
            pol = policies_by_id[inv.policy_id]
            cust, slots = policy_slots(pol)
            slots["invoice_no"] = inv.invoice_number
            slots["amount"] = f"{inv.amount:.2f}".replace(".", ",")
            text = composer.compose("invoice_note", cust.language, slots)
            ds.invoices[i] = Invoice(**{**inv.__dict__, "customer_note": text})
            ds.text_origin[f"invoices:{inv.invoice_id}"] = ("invoice_note", cust.language)

    return ds
