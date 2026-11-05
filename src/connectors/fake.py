"""In-memory fake connector used to run the pipeline end to end.

The text content is Dutch / French / English on purpose (it mimics real
support tickets); all field names, ids and metadata keys are English.
"""

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from src.connectors.base import SourceConnector
from src.contracts import ItemRef, RawDocument
from src.processing.clean import MESSAGE_SEPARATOR, split_messages
from src.registry import register

logger = logging.getLogger(__name__)

DOMAINS = ("claims", "policies", "billing", "onboarding", "complaints", "compliance")
LANGUAGES = ("nl", "fr", "en")
CELLS = tuple((d, lang) for d in DOMAINS for lang in LANGUAGES)

N_REGULAR = 52
EXACT_DUPLICATE_OF = (0, 7, 14)  # regular doc indexes copied verbatim (modulo case/space)
NEAR_DUPLICATE_OF = (3, 10)  # regular doc indexes re-sent as a reply in the same thread

OPENERS: dict[str, tuple[str, ...]] = {
    "nl": ("Goedemiddag,", "Beste,", "Geachte mevrouw, geachte heer,"),
    "fr": ("Bonjour,", "Madame, Monsieur,", "Bonsoir,"),
    "en": ("Hello,", "Dear team,", "Good afternoon,"),
}
CLOSERS: dict[str, tuple[str, ...]] = {
    "nl": (
        "Alvast bedankt voor uw hulp. Met vriendelijke groeten, Jan Peeters.",
        "Bedankt voor uw snelle antwoord. Groeten, An Janssens.",
        "Ik hoor graag van u. Vriendelijke groet, Pieter Claes.",
    ),
    "fr": (
        "Merci d'avance pour votre aide. Cordialement, Sophie Dubois.",
        "Dans l'attente de votre réponse, je vous remercie. Bien à vous, Marc Lambert.",
        "Merci pour votre retour rapide. Salutations, Julie Martin.",
    ),
    "en": (
        "Thanks in advance for your help. Kind regards, Tom Wilson.",
        "I look forward to hearing from you. Best regards, Emma Brown.",
        "Thank you for your quick reply. Regards, James Smith.",
    ),
}

# Three distinct core messages per (domain, language) cell.
CORES: dict[tuple[str, str], tuple[str, str, str]] = {
    ("claims", "nl"): (
        "Mijn wagen werd gisteren aangereden op de parking van de supermarkt. Ik wil graag "
        "een schadeaangifte doen en stuur u het ingevulde aanrijdingsformulier.",
        "Na het onweer van vorige week is er waterschade in onze keuken. Wanneer kan de "
        "expert langskomen om de schade vast te stellen?",
        "Ik heb drie weken geleden een schadegeval gemeld voor mijn fiets, maar ik heb nog "
        "geen vergoeding ontvangen. Kunt u nagaan hoe ver de behandeling staat?",
    ),
    ("claims", "fr"): (
        "Ma voiture a été percutée hier sur un parking. Je souhaite déclarer le sinistre et "
        "je vous joins le constat amiable signé par les deux conducteurs.",
        "Suite à la tempête, la toiture de notre maison est endommagée. Quand l'expert "
        "pourra-t-il passer pour évaluer les dégâts ?",
        "J'ai signalé un sinistre il y a un mois et je n'ai toujours reçu aucune "
        "indemnisation. Pouvez-vous me dire où en est le traitement ?",
    ),
    ("claims", "en"): (
        "My car was hit in a car park yesterday. I would like to file a claim and I am "
        "attaching the accident report signed by both drivers.",
        "After last week's storm there is water damage in our kitchen. When can the loss "
        "adjuster visit to assess the damage?",
        "I reported a claim for my stolen bicycle three weeks ago and have not received any "
        "compensation yet. Could you tell me what the status is?",
    ),
    ("policies", "nl"): (
        "Is een overstroming gedekt door mijn woningverzekering? Ik vind die uitsluiting "
        "niet terug in de algemene voorwaarden.",
        "Hoeveel bedraagt de vrijstelling op mijn autoverzekering en geldt de waarborg "
        "rechtsbijstand ook in het buitenland?",
        "Kunt u mij uitleggen welke dekking ik heb voor diefstal uit mijn wagen volgens de "
        "bijzondere voorwaarden van mijn contract?",
    ),
    ("policies", "fr"): (
        "Une inondation est-elle couverte par mon assurance habitation ? Je ne retrouve pas "
        "cette exclusion dans les conditions générales.",
        "Quelle est la franchise sur mon assurance auto et la garantie protection juridique "
        "s'applique-t-elle à l'étranger ?",
        "Pouvez-vous m'expliquer quelle couverture j'ai pour le vol dans mon véhicule selon "
        "les conditions particulières de mon contrat ?",
    ),
    ("policies", "en"): (
        "Is flooding covered by my home insurance? I cannot find that exclusion anywhere in "
        "the terms and conditions.",
        "How high is the deductible on my car insurance, and does the legal assistance "
        "coverage also apply abroad?",
        "Could you explain what cover I have for theft from my vehicle according to the "
        "policy wording of my contract?",
    ),
    ("billing", "nl"): (
        "Ik heb een betalingsherinnering ontvangen voor mijn premie van maart, terwijl de "
        "domiciliering op mijn rekening actief is. Kunt u dit nakijken?",
        "Op mijn laatste factuur staat een hoger bedrag dan vorig jaar. Kunt u mij uitleggen "
        "waarom de premie gestegen is?",
        "Ik heb per vergissing twee keer betaald via overschrijving. Wanneer mag ik de "
        "terugbetaling op mijn rekening verwachten?",
    ),
    ("billing", "fr"): (
        "J'ai reçu un rappel de paiement pour ma prime de mars alors que la domiciliation "
        "est active sur mon compte. Pouvez-vous vérifier ?",
        "Ma dernière facture indique un montant plus élevé que l'an dernier. Pouvez-vous "
        "m'expliquer pourquoi la prime a augmenté ?",
        "J'ai payé deux fois par erreur par virement bancaire. Quand puis-je espérer le "
        "remboursement sur mon compte ?",
    ),
    ("billing", "en"): (
        "I received a payment reminder for my March premium even though the direct debit on "
        "my account is active. Could you look into this?",
        "My latest invoice shows a higher amount than last year. Could you explain why the "
        "premium has increased?",
        "I accidentally paid twice by bank transfer. When can I expect the refund on my "
        "account?",
    ),
    ("onboarding", "nl"): (
        "Wij verhuizen binnenkort naar Gent. Kunt u de adreswijziging doorvoeren en mij een "
        "offerte bezorgen voor de nieuwe woning?",
        "Ik ben een nieuwe klant en wil graag een familiale verzekering onderschrijven. "
        "Welke documenten heeft u nodig?",
        "Ik wil mijn omniumverzekering opzeggen omdat ik mijn wagen verkocht heb. Hoe "
        "verloopt de opzegging precies?",
    ),
    ("onboarding", "fr"): (
        "Nous déménageons bientôt à Namur. Pouvez-vous enregistrer le changement d'adresse "
        "et m'envoyer un devis pour le nouveau logement ?",
        "Je suis un nouveau client et je souhaite souscrire une assurance familiale. Quels "
        "documents dois-je vous fournir ?",
        "Je souhaite résilier mon assurance omnium car j'ai vendu ma voiture. Comment se "
        "passe la résiliation exactement ?",
    ),
    ("onboarding", "en"): (
        "We are moving to Leuven soon. Could you register the change of address and send me "
        "a quote for the new home?",
        "I am a new customer and would like to sign up for a family liability insurance. "
        "Which documents do you need from me?",
        "I want to cancel my comprehensive car insurance because I sold the car. How does "
        "the cancellation work exactly?",
    ),
    ("complaints", "nl"): (
        "Ik ben zeer ontevreden over de manier waarop mijn vraag behandeld werd. Ik wens "
        "formeel een klacht in te dienen.",
        "Dit is al de derde keer dat niemand mij terugbelt. Als dit niet snel opgelost "
        "wordt, stap ik naar de ombudsman.",
        "Ik ben teleurgesteld in uw dienstverlening en vraag een escalatie naar een "
        "verantwoordelijke. Deze situatie is onaanvaardbaar.",
    ),
    ("complaints", "fr"): (
        "Je suis très insatisfait de la manière dont ma demande a été traitée. Je souhaite "
        "déposer une plainte formelle.",
        "C'est la troisième fois que personne ne me rappelle. Si ce n'est pas résolu "
        "rapidement, je contacterai le médiateur.",
        "Je suis déçu par votre service et je demande que ma réclamation soit transmise à "
        "un responsable. Cette situation est inacceptable.",
    ),
    ("complaints", "en"): (
        "I am very dissatisfied with the way my request was handled. I would like to file a "
        "formal complaint.",
        "This is the third time nobody has called me back. If this is not resolved soon, I "
        "will contact the ombudsman.",
        "I am disappointed with your service and I ask for an escalation to a manager. This "
        "situation is unacceptable.",
    ),
    ("compliance", "nl"): (
        "Op basis van de AVG wil ik mijn inzagerecht uitoefenen. Welke persoonsgegevens "
        "bewaart u over mij?",
        "Ik trek mijn toestemming in voor het gebruik van mijn gegevens voor marketing. Kunt "
        "u bevestigen dat dit verwerkt is?",
        "Waar kan ik het IPID-document van mijn verzekering raadplegen? Ik wil ook weten hoe "
        "u mijn privacy beschermt.",
    ),
    ("compliance", "fr"): (
        "Conformément au RGPD, je souhaite exercer mon droit d'accès. Quelles données "
        "personnelles conservez-vous à mon sujet ?",
        "Je retire mon consentement à l'utilisation de mes données à des fins de marketing. "
        "Pouvez-vous confirmer que c'est pris en compte ?",
        "Où puis-je consulter le document IPID de mon assurance ? Je voudrais aussi savoir "
        "comment vous protégez ma vie privée.",
    ),
    ("compliance", "en"): (
        "Under the GDPR I would like to exercise my right of access. What personal data do "
        "you hold about me?",
        "I withdraw my consent to the use of my data for marketing purposes. Could you "
        "confirm that this has been processed?",
        "Where can I find the IPID document for my insurance? I would also like to know how "
        "you protect my privacy.",
    ),
}

# One agent reply per cell: every regular item is a 2-message thread
# (customer message, agent reply), so the example builder has a real pair.
AGENT_GREETING: dict[str, str] = {"nl": "Beste klant,", "fr": "Bonjour,", "en": "Hello,"}
REPLIES: dict[tuple[str, str], str] = {
    ("claims", "nl"): "Wij hebben uw schadeaangifte goed ontvangen en een expert aangesteld. Hij neemt binnen twee "
                      "werkdagen contact met u op om de schade vast te stellen; daarna volgt de vergoeding.",
    ("claims", "fr"): "Nous avons bien reçu votre déclaration de sinistre et désigné un expert. Il vous contactera sous "
                      "deux jours ouvrables pour évaluer les dégâts, puis l'indemnisation suivra.",
    ("claims", "en"): "We have registered your claim and appointed a loss adjuster. They will contact you within two "
                      "working days to assess the damage, after which the compensation follows.",
    ("policies", "nl"): "Volgens de algemene voorwaarden van uw polis is dit gedekt, met een vrijstelling van 250 euro "
                        "per gebeurtenis. De uitsluitingen staan in artikel 5 van de voorwaarden.",
    ("policies", "fr"): "Selon les conditions générales de votre police, ce risque est couvert, avec une franchise de "
                        "250 euros par événement. Les exclusions figurent à l'article 5.",
    ("policies", "en"): "Under the terms and conditions of your policy this is covered, with a deductible of 250 euros "
                        "per event. The exclusions are listed in section 5 of the policy wording.",
    ("billing", "nl"): "Wij hebben uw betaling teruggevonden en de herinnering geannuleerd. De volgende premie wordt via "
                       "domiciliering geïnd op de vervaldag.",
    ("billing", "fr"): "Nous avons retrouvé votre paiement et annulé le rappel. La prochaine prime sera prélevée par "
                       "domiciliation à l'échéance.",
    ("billing", "en"): "We found your payment and cancelled the reminder. The next premium will be collected by direct "
                       "debit on the due date.",
    ("onboarding", "nl"): "Uw aanvraag is verwerkt: de nieuwe polis gaat in op de eerste dag van volgende maand en u "
                          "ontvangt het bijvoegsel binnen vijf werkdagen per e-mail.",
    ("onboarding", "fr"): "Votre demande est traitée : le nouveau contrat prend cours le premier jour du mois prochain et "
                          "l'avenant vous sera envoyé par e-mail sous cinq jours ouvrables.",
    ("onboarding", "en"): "Your request has been processed: the new policy starts on the first day of next month and the "
                          "endorsement will be e-mailed to you within five working days.",
    ("complaints", "nl"): "Wij betreuren dat u ontevreden bent. Uw klacht is doorgegeven aan onze klachtendienst, die u "
                          "binnen vijf werkdagen contacteert. U kunt zich ook tot de Ombudsman wenden.",
    ("complaints", "fr"): "Nous regrettons votre insatisfaction. Votre plainte a été transmise à notre service des "
                          "réclamations, qui vous contactera sous cinq jours ouvrables.",
    ("complaints", "en"): "We are sorry you are dissatisfied. Your complaint has been passed to our complaints team, who "
                          "will contact you within five working days.",
    ("compliance", "nl"): "Uw verzoek op basis van de AVG is geregistreerd. Wij beantwoorden het binnen één maand en "
                          "verwerken uw persoonsgegevens enkel voor het beheer van uw contracten.",
    ("compliance", "fr"): "Votre demande au titre du RGPD est enregistrée. Nous y répondrons dans un délai d'un mois ; vos "
                          "données personnelles servent uniquement à la gestion de vos contrats.",
    ("compliance", "en"): "Your GDPR request has been registered. We will answer within one month; your personal data is "
                          "only used to manage your contracts.",
}

OFF_TOPIC = (
    "Hi all, the team meeting is moved to Tuesday afternoon in room 3. Please bring your "
    "laptops and the slides for the quarterly review."
)
TEN_CHARS = "Merci bien"


@dataclass(frozen=True)
class _FakeItem:
    item_id: str
    thread_id: str
    kind: str  # regular | exact_duplicate | near_duplicate | empty | too_short | off_topic
    text: str


def _generate(seed: int) -> list[_FakeItem]:
    rng = random.Random(seed)
    regular: list[_FakeItem] = []
    for i in range(N_REGULAR):
        domain, lang = CELLS[i % len(CELLS)]
        core = CORES[(domain, lang)][i // len(CELLS)]
        customer = f"{rng.choice(OPENERS[lang])}\n\n{core}\n\n{rng.choice(CLOSERS[lang])}"
        reply = f"{AGENT_GREETING[lang]}\n\n{REPLIES[(domain, lang)]}"
        text = f"{customer}{MESSAGE_SEPARATOR}{reply}"
        regular.append(_FakeItem(f"doc-{i:03d}", f"thread-{i:03d}", "regular", text))

    items = list(regular)
    n = len(items)
    for k, src_index in enumerate(EXACT_DUPLICATE_OF):
        original = regular[src_index].text
        # Same normalised text: variants in case / whitespace only.
        text = (original, original.upper(), "  " + original.replace(" ", "  ") + "\n")[k]
        items.append(_FakeItem(f"doc-{n:03d}", f"thread-{n:03d}", "exact_duplicate", text))
        n += 1
    for near_index in NEAR_DUPLICATE_OF:
        source_item = regular[near_index]
        # Re-sent as a reply in the same thread -> same group, never split.
        items.append(
            _FakeItem(
                f"doc-{n:03d}", source_item.thread_id, "near_duplicate", "RE: " + source_item.text
            )
        )
        n += 1
    for kind, text in (("empty", ""), ("too_short", TEN_CHARS), ("off_topic", OFF_TOPIC)):
        items.append(_FakeItem(f"doc-{n:03d}", f"thread-{n:03d}", kind, text))
        n += 1
    return items


@register
class FakeConnector(SourceConnector):
    """Yield 60 in-memory RawDocuments covering the 6 domains and 3
    languages, deliberately including: 3 exact duplicates, 2 near
    duplicates, 1 empty document, 1 ten-character document.
    Deterministic (seed from config). The text content is Dutch /
    French / English; all field names and ids are English.

    It also includes 1 off-topic document, to exercise the
    ``domain_unresolved`` drop path.
    """

    source_type = "fake"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.seed = int(config.get("seed", 0))
        self._items = {item.item_id: item for item in _generate(self.seed)}

    def test_connection(self) -> bool:
        return True

    def list_items(self) -> Iterator[ItemRef]:
        for item in self._items.values():
            yield ItemRef(
                source_type=self.source_type,
                item_id=item.item_id,
                uri=f"memory://{self.name}/{item.item_id}",
                hints={"thread_id": item.thread_id},
            )

    def fetch(self, ref: ItemRef) -> RawDocument:
        item = self._items[ref.item_id]
        return RawDocument(
            id=f"{self.name}:{item.item_id}",
            source_type=self.source_type,
            source_uri=ref.uri,
            content=item.text,
            mime_type="text/plain",
            metadata={
                "thread_id": item.thread_id,
                "fake_kind": item.kind,
                "kind": "thread",
                "messages": [
                    {"direction": "inbound" if i == 0 else "outbound"}
                    for i in range(len(split_messages(item.text)))
                ],
            },
        )
