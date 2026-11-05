"""Deterministic synthetic support-ticket export: a CSV AND re-exported .eml.

data/generated/tickets/tickets_export.csv
    ~60 threads / ~180 messages, columns CSV_COLUMNS, direction inbound
    (customer) / outbound (agent). Content in Dutch / French / English.
data/generated/tickets/emails/*.eml
    15 of those threads RE-EXPORTED from the mailbox: other ids, quoted
    history (">" lines), other signature blocks, multipart with an HTML part
    (one message HTML-only), some parts in ISO-8859-1. The ticket reference
    in agent replies differs too (ticket id vs mailbox id), so the two
    exports are near-duplicates, not exact copies.

Noise: NL/FR/EN signature blocks, a 6-line legal disclaimer on every
outbound message, 2 auto-acknowledgement threads, 2 threads with no agent
reply, 1 empty body, mojibake in 3 CSV messages.
Coverage: onboarding and complaints are pushed (thin domains), 3 GDPR
access requests, 3 threads about claims that ALSO exist in the CRM
claim_notes (cross-source semantic duplicates, deliberately unresolved).
Every PII value written is registered in the shared manifest (origin
"tickets"). Byte-reproducible: seeded RNG, sha256-derived ids and MIME
boundaries, fixed timestamps.
"""

import csv
import hashlib
import io
import random
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.utils import format_datetime
from html import escape
from pathlib import Path
from typing import Final

import ftfy

from src.connectors.email import CSV_COLUMNS, csv_thread_uri
from src.synthetic.crm import generate_crm, valid_be_iban, valid_rrn
from src.synthetic.manifest import update_manifest

AGENT_DOMAIN: Final = "assurance.example.be"
TZ: Final = timezone(timedelta(hours=2))
BASE_TIME: Final = datetime(2025, 10, 1, 9, 0, tzinfo=TZ)
N_EML_THREADS: Final = 15

# --- Per-language boilerplate -----------------------------------------------------------

OPENERS = {"nl": ("Goedemiddag,", "Beste,", "Hallo,"), "fr": ("Bonjour,", "Madame, Monsieur,", "Bonsoir,"),
           "en": ("Hello,", "Dear team,", "Good morning,")}
CUSTOMER_SIG_CSV = {"nl": "Met vriendelijke groeten,\n{name}\n{phone}", "fr": "Cordialement,\n{name}\n{phone}",
                    "en": "Kind regards,\n{name}\n{phone}"}
CUSTOMER_SIG_EML = {"nl": "Groeten,\n{name}", "fr": "Bien à vous,\n{name}", "en": "Best regards,\n{name}"}
GREETING = {"nl": "Beste {name},", "fr": "Bonjour {name},", "en": "Dear {name},"}
AGENT_SIG_CSV = {"nl": "Met vriendelijke groeten,\n{agent}\nKlantendienst\nVerzekeringen NV",
                 "fr": "Cordialement,\n{agent}\nService clientèle\nAssurances SA",
                 "en": "Kind regards,\n{agent}\nCustomer service\nInsurance NV"}
AGENT_SIG_EML = {"nl": "--\n{agent} | Klantendienst\nT 02 555 01 23",
                 "fr": "--\n{agent} | Service clientèle\nT 02 555 01 23",
                 "en": "--\n{agent} | Customer service\nT 02 555 01 23"}
REFERENCE = {"nl": "Referentie: {ref}", "fr": "Référence : {ref}", "en": "Reference: {ref}"}
REPLY_PREFIX = {"nl": "Antw: ", "fr": "RE: ", "en": "Re: "}
ATTRIBUTION = {"nl": "Op {when} schreef {who} <{addr}>:", "fr": "Le {when}, {who} <{addr}> a écrit :",
               "en": "On {when}, {who} <{addr}> wrote:"}
SERVICE_ADDRESS = {"nl": f"klantendienst@{AGENT_DOMAIN}", "fr": f"service.clientele@{AGENT_DOMAIN}",
                   "en": f"customer.service@{AGENT_DOMAIN}"}
DISCLAIMER = {
    "nl": ("Dit bericht is vertrouwelijk en uitsluitend bestemd voor de geadresseerde.\n"
           "Indien u dit bericht per vergissing ontving, verwittig dan de afzender\n"
           "en verwijder het bericht onmiddellijk uit uw systeem.\n"
           "Elke verspreiding of kopie zonder voorafgaande toelating is verboden.\n"
           "Verzekeringen NV, Koningsstraat 1, 1000 Brussel, RPR Brussel.\n"
           "Verzekeringsonderneming toegelaten onder codenummer 0000."),
    "fr": ("Ce message est confidentiel et destiné exclusivement à son destinataire.\n"
           "Si vous l'avez reçu par erreur, merci d'en avertir l'expéditeur\n"
           "et de le supprimer immédiatement de votre système.\n"
           "Toute diffusion ou copie non autorisée est interdite.\n"
           "Assurances SA, rue Royale 1, 1000 Bruxelles, RPM Bruxelles.\n"
           "Entreprise d'assurances agréée sous le code 0000."),
    "en": ("This message is confidential and intended solely for the addressee.\n"
           "If you received it in error, please notify the sender\n"
           "and delete it from your system immediately.\n"
           "Any distribution or copy without prior authorisation is prohibited.\n"
           "Insurance NV, Rue Royale 1, 1000 Brussels, RPR Brussels.\n"
           "Insurance company authorised under code 0000."),
}
AUTO_ACK = {
    "nl": ("Dit is een automatisch antwoord. Wij hebben uw bericht goed ontvangen en behandelen het binnen "
           "drie werkdagen. Gelieve niet te antwoorden op dit bericht."),
    "fr": ("Ceci est une réponse automatique. Nous avons bien reçu votre message et le traiterons dans les "
           "trois jours ouvrables. Merci de ne pas répondre à ce message."),
}
AGENTS = {"nl": (("Inge", "Vermeulen"), ("Dirk", "Wouters")), "fr": (("Nathalie", "Lefèvre"), ("Olivier", "Martin")),
          "en": (("Helen", "Carter"), ("Mark", "Dawson"))}
CUSTOMERS = {
    "nl": (("Jan", "Peeters"), ("Lotte", "Maes"), ("Wim", "Jacobs"), ("Sofie", "Wouters"), ("Bart", "Goossens"),
           ("Nele", "De Smet"), ("Koen", "Claes"), ("An", "Janssens"), ("Tom", "Mertens"), ("Els", "Willems")),
    "fr": (("Sophie", "Dubois"), ("Marc", "Lambert"), ("Julie", "Martin"), ("Pierre", "Leroy"),
           ("Camille", "Dupont"), ("Nicolas", "Lemaire"), ("Claire", "Renard"), ("Antoine", "Simon")),
    "en": (("Emma", "Wilson"), ("James", "Brown"), ("Olivia", "Taylor"), ("Daniel", "Clarke")),
}
STREETS = {"nl": ("Kerkstraat", "Stationsstraat", "Molenstraat"), "fr": ("Rue de la Gare", "Rue du Moulin",
           "Avenue des Tilleuls"), "en": ("Rue Royale", "Avenue Louise", "Boulevard Anspach")}
CITIES = {"nl": (("Gent", 9000), ("Leuven", 3000), ("Brugge", 8000)), "fr": (("Namur", 5000), ("Liège", 4000),
          ("Mons", 7000)), "en": (("Brussels", 1000), ("Waterloo", 1410), ("Leuven", 3000))}

# --- Topic pools: each request carries its own reply and follow-up -------------------------


@dataclass(frozen=True)
class Topic:
    """One customer request with ITS agent reply and ITS follow-up exchange.

    Replies are tied to the request: a reply drawn independently of the
    question produced pairs where the agent answered another topic.
    """

    subject: str
    request: str
    reply: str
    followup_in: str
    followup_out: str


@dataclass(frozen=True)
class Pool:
    topics: tuple[Topic, ...]
    details: tuple[str, ...]


POOLS: dict[tuple[str, str], Pool] = {
    ("onboarding", "nl"): Pool(
        topics=(
            Topic("Offerte autoverzekering",
                  "Ik koop binnenkort een nieuwe wagen en had graag een offerte voor een omniumverzekering. Kunt u "
                  "mij een prijsofferte bezorgen?",
                  "Bedankt voor uw aanvraag. In bijlage vindt u de offerte; de nieuwe polis kan ingaan zodra u ze "
                  "ondertekend terugstuurt.",
                  "Ik heb de offerte bekeken. Kan de nieuwe polis al vanaf volgende maand ingaan?",
                  "Dat is mogelijk. Wij passen de ingangsdatum van de nieuwe polis aan zodra wij uw akkoord "
                  "ontvangen."),
            Topic("Adreswijziging",
                  "Wij verhuizen op {date} naar {address}. Graag een adreswijziging voor al mijn polissen.",
                  "Wij hebben de adreswijziging verwerkt. U ontvangt binnen vijf werkdagen een bijvoegsel bij uw "
                  "polis met het nieuwe adres.",
                  "Bedankt voor de snelle opvolging. Moet ik het bijvoegsel ondertekend terugsturen?",
                  "Ja, een ondertekend exemplaar van het bijvoegsel volstaat; u mag het per e-mail terugsturen."),
            Topic("Opzegging woningverzekering",
                  "Ik wil mijn woningverzekering opzeggen omdat ik mijn huis verkocht heb. De opzegging mag ingaan "
                  "op {date}.",
                  "Uw aanvraag tot opzegging is goed ontvangen. U krijgt een schriftelijke bevestiging met de "
                  "einddatum van de overeenkomst.",
                  "Krijg ik na de opzegging het deel van de premie terug voor de resterende maanden?",
                  "Ja, het deel van de jaarpremie na de einddatum van de overeenkomst wordt binnen dertig dagen "
                  "teruggestort."),
            Topic("Nieuwe polis familiale",
                  "Als nieuwe klant wil ik graag een familiale verzekering onderschrijven. Hoe verloopt de "
                  "onderschrijving precies?",
                  "Welkom! De onderschrijving gebeurt online: u vult het aanvraagformulier in, wij controleren de "
                  "gegevens en de familiale verzekering gaat in op de dag van uw elektronische handtekening.",
                  "Moet ik voor de onderschrijving al een rekeningnummer doorgeven?",
                  "Ja, voor de domiciliering van de premie hebben wij uw rekeningnummer nodig; u kunt het in het "
                  "aanvraagformulier invullen."),
        ),
        details=("Mijn huidige polisnummer is {policy_no}; u kunt mij bellen op {phone}.",
                 "Laat mij weten welke documenten u nodig hebt om het dossier af te ronden.",
                 "Kunt u bevestigen tegen wanneer ik een antwoord mag verwachten over mijn aanvraag?",
                 "Ik ben al jaren klant bij u en hoop op een snelle afhandeling.",
                 "Graag ontvang ik de bevestiging per e-mail.",
                 "Mijn partner staat mee op de polis vermeld."),
    ),
    ("onboarding", "fr"): Pool(
        topics=(
            Topic("Demande de devis habitation",
                  "Nous achetons un appartement à {city} et je souhaiterais recevoir un devis pour une assurance "
                  "habitation.",
                  "Merci pour votre demande. Vous trouverez le devis en annexe ; la souscription peut se faire par "
                  "signature électronique.",
                  "J'ai lu le devis. Le nouveau contrat peut-il commencer dès le mois prochain ?",
                  "C'est possible. Nous adapterons la date de souscription dès réception de votre accord."),
            Topic("Changement d'adresse",
                  "Je déménage le {date} à l'adresse suivante : {address}. Merci d'enregistrer ce changement "
                  "d'adresse sur mes contrats.",
                  "Le changement d'adresse a bien été enregistré. Vous recevrez un avenant à votre contrat dans les "
                  "cinq jours ouvrables.",
                  "Merci pour la réponse rapide. Dois-je renvoyer l'avenant signé ?",
                  "Oui, un exemplaire signé de l'avenant suffit ; vous pouvez le renvoyer par e-mail."),
            Topic("Résiliation contrat auto",
                  "J'ai vendu mon véhicule et je souhaite résilier mon contrat auto. La résiliation peut prendre "
                  "effet le {date}.",
                  "Votre demande de résiliation est bien reçue. Une confirmation écrite avec la date de fin du "
                  "contrat vous sera envoyée.",
                  "La part de prime déjà payée pour la période après la résiliation me sera-t-elle remboursée ?",
                  "Oui, la partie de la prime annuelle couvrant la période après la résiliation vous sera "
                  "remboursée sous trente jours."),
            Topic("Souscription assurance familiale",
                  "En tant que nouveau client, je voudrais souscrire une assurance familiale. Comment se passe la "
                  "souscription ?",
                  "Bienvenue ! La souscription se fait en ligne : vous remplissez la demande, nous vérifions les "
                  "informations et l'assurance familiale prend cours à la date de votre signature électronique.",
                  "Dois-je déjà communiquer un numéro de compte pour la souscription ?",
                  "Oui, votre numéro de compte est nécessaire pour la domiciliation de la prime ; vous pouvez "
                  "l'indiquer dans la demande."),
        ),
        details=("Mon numéro de contrat actuel est {policy_no} ; vous pouvez me joindre au {phone}.",
                 "Pouvez-vous m'indiquer les documents nécessaires pour compléter le dossier ?",
                 "Pourriez-vous me confirmer le délai de traitement de ma demande ?",
                 "Je suis client chez vous depuis des années et j'espère un traitement rapide.",
                 "Merci de m'envoyer la confirmation par e-mail.",
                 "Mon conjoint figure également sur le contrat."),
    ),
    ("onboarding", "en"): Pool(
        topics=(
            Topic("Quote for car insurance",
                  "I am buying a new car next month and would like a quote for comprehensive car insurance.",
                  "Thank you for your request. Please find the quotation attached; the new policy can start as soon "
                  "as you return it signed.",
                  "I have read the quotation. Can the new policy start next month?",
                  "That is possible. We will adjust the start date of the new policy once we receive your "
                  "agreement."),
            Topic("Change of address",
                  "We are moving on {date} to {address}. Please register the change of address on all my policies.",
                  "We have processed the change of address. You will receive an endorsement to your policy within "
                  "five working days.",
                  "Thanks for the quick reply. Do I need to send the endorsement back signed?",
                  "Yes, a signed copy of the endorsement is enough; you can return it by e-mail."),
            Topic("Cancellation of home insurance",
                  "I have sold my house and would like to cancel my home insurance. The cancellation can take "
                  "effect on {date}.",
                  "We have received your cancellation request. You will receive a written confirmation with the end "
                  "date of the contract.",
                  "Will I get back the premium for the months after the cancellation?",
                  "Yes, the part of the annual premium after the end date of the contract is refunded within thirty "
                  "days."),
            Topic("New policy for my family",
                  "As a new customer I would like to sign up for a family liability policy. How do I subscribe?",
                  "Welcome! You can subscribe online: you fill in the application, we check the details and the "
                  "family liability policy starts on the day of your electronic signature.",
                  "Do I already need to give a bank account number to subscribe?",
                  "Yes, we need your account number for the direct debit of the premium; you can enter it in the "
                  "application."),
        ),
        details=("My current policy number is {policy_no}; you can reach me on {phone}.",
                 "Please let me know which documents you need to complete the file.",
                 "Could you confirm when I can expect an answer to my request?",
                 "I have been a customer for years and hope for a quick answer.",
                 "Please send me the confirmation by e-mail.",
                 "My partner is also named on the policy."),
    ),
    ("complaints", "nl"): Pool(
        topics=(
            Topic("Klacht over behandeling",
                  "Ik ben zeer ontevreden over de behandeling van mijn dossier {policy_no}. Al drie weken krijg ik "
                  "geen antwoord op mijn vragen.",
                  "Wij betreuren dat u ontevreden bent. Uw klacht werd geregistreerd en doorgegeven aan onze "
                  "klachtendienst, die u binnen vijf werkdagen contacteert.",
                  "Ik heb nog steeds niets gehoord over mijn klacht. Dit is werkelijk onaanvaardbaar.",
                  "Excuses voor de vertraging. Uw klacht heeft nu voorrang bij onze klachtendienst."),
            Topic("Formele klacht",
                  "Ik dien hierbij een formele klacht in: de afspraak van {date} werd zonder verwittiging "
                  "geannuleerd. Dit is onaanvaardbaar.",
                  "Wij bieden u onze excuses aan voor de geannuleerde afspraak. Uw klacht is geregistreerd en een "
                  "medewerker plant binnen twee werkdagen een nieuwe afspraak met u in.",
                  "Bedankt, maar ik wil ook weten waarom de afspraak zonder verwittiging werd geannuleerd.",
                  "De afspraak werd geannuleerd door een onverwachte afwezigheid van de expert; wij hadden u moeten "
                  "verwittigen en hebben onze procedure aangepast."),
            Topic("Escalatie gevraagd",
                  "Ik vraag een escalatie naar een verantwoordelijke. Uw medewerkers verwijzen mij telkens door "
                  "zonder oplossing.",
                  "Uw escalatie werd aanvaard: een teamverantwoordelijke neemt uw klacht persoonlijk op en belt u "
                  "morgen terug.",
                  "Bedankt voor het telefoontje, maar ik blijf ontevreden over de voorgestelde oplossing.",
                  "Wij noteren dat u ontevreden blijft en sturen uw klacht door naar de interne ombudsdienst."),
            Topic("Ombudsman",
                  "Als ik deze week geen duidelijk antwoord krijg, stap ik naar de Ombudsman van de Verzekeringen "
                  "met mijn klacht.",
                  "Wij begrijpen uw ongenoegen. Mocht ons antwoord op uw klacht u niet tevredenstellen, dan kunt u "
                  "zich wenden tot de Ombudsman van de Verzekeringen.",
                  "Hoe neem ik contact op met de Ombudsman van de Verzekeringen?",
                  "U kunt uw klacht online indienen via de website van de Ombudsman van de Verzekeringen of per "
                  "brief; vermeld daarbij het nummer van uw dossier."),
        ),
        details=("Ik ben bereikbaar op {phone}, maar niemand belt mij terug.",
                 "Ik ben teleurgesteld, want ik ben al vijftien jaar klant.",
                 "Ik verwacht een schriftelijke reactie op mijn bezwaar."),
    ),
    ("complaints", "fr"): Pool(
        topics=(
            Topic("Plainte traitement dossier",
                  "Je suis très insatisfait du traitement de mon dossier {policy_no}. Depuis trois semaines, "
                  "personne ne répond à mes messages.",
                  "Nous regrettons votre insatisfaction. Votre plainte est enregistrée et transmise à notre service "
                  "des réclamations, qui vous contactera sous cinq jours ouvrables.",
                  "Je n'ai toujours aucune nouvelle de ma plainte. C'est vraiment inacceptable.",
                  "Toutes nos excuses pour ce retard. Votre plainte est désormais traitée en priorité."),
            Topic("Réclamation formelle",
                  "Je dépose une réclamation formelle : le rendez-vous du {date} a été annulé sans prévenir. C'est "
                  "inacceptable.",
                  "Nous vous présentons nos excuses pour le rendez-vous annulé. Votre réclamation est enregistrée et "
                  "un collaborateur vous proposera une nouvelle date sous deux jours ouvrables.",
                  "Merci, mais je voudrais aussi savoir pourquoi le rendez-vous a été annulé sans prévenir.",
                  "Le rendez-vous a été annulé suite à l'absence imprévue de l'expert ; nous aurions dû vous "
                  "prévenir et avons adapté notre procédure."),
            Topic("Demande d'escalade",
                  "Je demande une escalade vers un responsable. Vos collaborateurs me renvoient de service en "
                  "service sans solution.",
                  "Votre demande d'escalade est acceptée : un responsable d'équipe reprend personnellement votre "
                  "plainte et vous rappelle demain.",
                  "Merci pour l'appel, mais je reste mécontent de la solution proposée.",
                  "Nous notons votre insatisfaction et transmettons votre réclamation au médiateur interne."),
            Topic("Médiateur",
                  "Sans réponse claire cette semaine, je saisirai le médiateur de l'Ombudsman des Assurances avec "
                  "ma plainte.",
                  "Nous comprenons votre mécontentement. Si notre réponse ne vous satisfait pas, vous pouvez vous "
                  "adresser au médiateur de l'Ombudsman des Assurances.",
                  "Comment puis-je contacter l'Ombudsman des Assurances ?",
                  "Vous pouvez introduire votre plainte en ligne sur le site de l'Ombudsman des Assurances ou par "
                  "courrier, en mentionnant le numéro de votre dossier."),
        ),
        details=("Je reste joignable au {phone}, mais personne ne me rappelle.",
                 "Je suis déçu, je suis client depuis quinze ans.",
                 "J'attends une réponse écrite à ma plainte."),
    ),
    ("complaints", "en"): Pool(
        topics=(
            Topic("Complaint about my file",
                  "I am very dissatisfied with how my file {policy_no} is being handled. Nobody has answered my "
                  "questions for three weeks.",
                  "We are sorry you are unhappy. Your complaint has been registered and passed to our complaints "
                  "team, who will contact you within five working days.",
                  "I still have not heard anything about my complaint. This is really unacceptable.",
                  "Apologies for the delay. Your complaint is now being handled as a priority."),
            Topic("Formal complaint",
                  "This is a formal complaint: the appointment of {date} was cancelled without notice. This is "
                  "unacceptable.",
                  "We apologise for the cancelled appointment. Your complaint has been registered and a colleague "
                  "will offer you a new date within two working days.",
                  "Thank you, but I would also like to know why the appointment was cancelled without notice.",
                  "The appointment was cancelled because the adjuster was unexpectedly absent; we should have "
                  "informed you and have changed our procedure."),
            Topic("Escalation request",
                  "I ask for an escalation to a manager. Your staff keep redirecting me without a solution.",
                  "Your escalation has been accepted: a team manager is taking over your complaint and will call "
                  "you tomorrow.",
                  "Thank you for the call, but I am still dissatisfied with the solution.",
                  "We note that you remain dissatisfied and are escalating your complaint to our internal "
                  "ombudsman."),
            Topic("Ombudsman",
                  "If I do not get a clear answer this week, I will take my complaint to the Insurance Ombudsman.",
                  "We understand that you are dissatisfied. If our answer does not satisfy you, you can contact the "
                  "Insurance Ombudsman.",
                  "How do I contact the Insurance Ombudsman?",
                  "You can file your complaint online on the Insurance Ombudsman's website or by letter, quoting "
                  "your file number."),
        ),
        details=("I can be reached on {phone}, but nobody calls me back.",
                 "I am disappointed, I have been a customer for fifteen years.",
                 "I expect a written answer to my complaint."),
    ),
    ("billing", "nl"): Pool(
        topics=(
            Topic("Betalingsherinnering",
                  "Ik ontving een betalingsherinnering voor de premie van polis {policy_no}, terwijl de "
                  "domiciliering actief is.",
                  "Wij hebben uw betaling via domiciliering teruggevonden; de herinnering mag u als onbestaande "
                  "beschouwen.",
                  "Bedankt. Wordt de domiciliering vanaf nu automatisch gebruikt voor de premie?",
                  "Ja, de volgende premies worden automatisch via domiciliering geïnd."),
            Topic("Dubbele betaling",
                  "Ik heb de factuur van {date} per vergissing twee keer betaald via overschrijving. Graag een "
                  "terugbetaling op {iban}.",
                  "De terugbetaling van de dubbele betaling werd vandaag uitgevoerd en staat binnen drie werkdagen op "
                  "uw rekening.",
                  "Ik heb de terugbetaling nog niet ontvangen, kunt u dit nakijken?",
                  "De overschrijving is vertrokken; de terugbetaling zou morgen zichtbaar moeten zijn op uw "
                  "rekening."),
            Topic("Vraag over factuur",
                  "Op mijn laatste factuur staat een bedrag van {amount} euro. Waarom is de premie gestegen?",
                  "De premie werd geïndexeerd; in bijlage vindt u een gedetailleerd overzicht van de factuur.",
                  "Met hoeveel procent werd de premie precies geïndexeerd?",
                  "De premie werd met 3,2 procent geïndexeerd, volgens het indexcijfer van de consumptieprijzen."),
            Topic("Domiciliering wijzigen",
                  "Ik wil mijn domiciliering wijzigen naar rekening {iban} vanaf de volgende vervaldag van mijn "
                  "premie.",
                  "Wij hebben uw nieuwe rekeningnummer geregistreerd. De domiciliering wordt aangepast vanaf de "
                  "volgende vervaldag van uw premie.",
                  "Moet ik de oude domiciliering zelf stopzetten bij mijn bank?",
                  "Nee, wij annuleren de oude domiciliering zelf; u hoeft niets te doen bij uw bank."),
        ),
        details=("Kunt u mij een overzicht van de betalingen van dit jaar bezorgen?",
                 "Ik wil vermijden dat er een aanmaning volgt.",
                 "U kunt mij bellen op {phone} als er iets onduidelijk is aan de factuur."),
    ),
    ("billing", "fr"): Pool(
        topics=(
            Topic("Rappel de paiement",
                  "J'ai reçu un rappel de paiement pour la prime du contrat {policy_no} alors que la domiciliation "
                  "est active.",
                  "Nous avons retrouvé votre paiement par domiciliation ; vous pouvez ignorer ce rappel.",
                  "Merci. La domiciliation sera-t-elle utilisée automatiquement pour la prime désormais ?",
                  "Oui, les prochaines primes seront prélevées automatiquement par domiciliation."),
            Topic("Double paiement",
                  "J'ai payé deux fois la facture du {date} par virement. Merci de procéder au remboursement sur "
                  "{iban}.",
                  "Le remboursement du double paiement a été effectué aujourd'hui et sera visible sur votre compte "
                  "sous trois jours ouvrables.",
                  "Je n'ai pas encore reçu le remboursement, pouvez-vous vérifier ?",
                  "Le virement est parti ; le remboursement devrait apparaître demain sur votre compte."),
            Topic("Question sur ma facture",
                  "Ma dernière facture indique un montant de {amount} euros. Pourquoi la prime a-t-elle augmenté ?",
                  "La prime a été indexée ; vous trouverez en annexe le détail de la facture.",
                  "De combien la prime a-t-elle été indexée exactement ?",
                  "La prime a été indexée de 3,2 %, selon l'indice des prix à la consommation."),
            Topic("Modification domiciliation",
                  "Je souhaite modifier ma domiciliation vers le compte {iban} à partir de la prochaine échéance de "
                  "ma prime.",
                  "Votre nouveau numéro de compte est enregistré. La domiciliation sera adaptée à partir de la "
                  "prochaine échéance de votre prime.",
                  "Dois-je annuler moi-même l'ancienne domiciliation auprès de ma banque ?",
                  "Non, nous annulons nous-mêmes l'ancienne domiciliation ; vous n'avez rien à faire auprès de "
                  "votre banque."),
        ),
        details=("Pouvez-vous m'envoyer un relevé des paiements de cette année ?",
                 "Je voudrais éviter une mise en demeure.",
                 "Vous pouvez me joindre au {phone} si la facture pose question."),
    ),
    ("billing", "en"): Pool(
        topics=(
            Topic("Payment reminder",
                  "I received a payment reminder for the premium of policy {policy_no}, although my direct debit "
                  "is active.",
                  "We found your payment by direct debit; you can ignore the payment reminder.",
                  "Thank you. Will the direct debit be used automatically for the premium from now on?",
                  "Yes, the next premiums will be collected automatically by direct debit."),
            Topic("Double payment",
                  "I paid the invoice of {date} twice by bank transfer by mistake. Please refund it to {iban}.",
                  "The refund of the double payment was made today and will be in your account within three "
                  "working days.",
                  "I have not received the refund yet, could you check?",
                  "The bank transfer has left our account; the refund should be visible in your account "
                  "tomorrow."),
            Topic("Question about my invoice",
                  "My latest invoice shows an amount of {amount} euros. Why has the premium gone up?",
                  "The premium was indexed; please find attached a detailed breakdown of the invoice.",
                  "By what percentage exactly was the premium indexed?",
                  "The premium was indexed by 3.2 percent, in line with the consumer price index."),
            Topic("Changing my direct debit",
                  "I would like to move my direct debit to account {iban} from the next due date of my premium.",
                  "We have registered your new account number. The direct debit will be changed from the next due "
                  "date of your premium.",
                  "Do I need to cancel the old direct debit with my bank myself?",
                  "No, we cancel the old direct debit ourselves; you do not need to do anything with your bank."),
        ),
        details=("Could you send me an overview of this year's payments?",
                 "I would like to avoid an overdue notice.",
                 "You can call me on {phone} if anything about the invoice is unclear."),
    ),
    ("compliance", "nl"): Pool(
        topics=(
            Topic("Inzage in mijn persoonsgegevens",
                  "Op basis van de AVG wil ik weten welke persoonsgegevens u over mij bewaart en waarvoor u ze "
                  "gebruikt.",
                  "Wij bezorgen u binnen één maand een overzicht van uw persoonsgegevens, de doeleinden van de "
                  "verwerking en de bewaartermijnen, zoals de AVG voorschrijft.",
                  "Krijg ik dat overzicht van mijn persoonsgegevens ook in een leesbaar formaat?",
                  "Ja, u ontvangt het overzicht als pdf via het beveiligde klantenportaal; het inzagerecht is "
                  "kosteloos."),
            Topic("Verbetering van mijn gegevens",
                  "Mijn adres en telefoonnummer in uw bestand zijn verouderd. Ik wil die persoonsgegevens laten "
                  "verbeteren volgens de AVG.",
                  "Wij hebben uw persoonsgegevens verbeterd. Het recht op verbetering is kosteloos en geldt voor al "
                  "uw contracten bij ons.",
                  "Worden mijn oude persoonsgegevens ook uit uw systemen gewist?",
                  "Ja, de verouderde gegevens worden vervangen en niet langer gebruikt; enkel de wettelijk verplichte "
                  "historiek blijft bewaard."),
            Topic("Toestemming marketing intrekken",
                  "Ik wil mijn toestemming voor marketing per e-mail intrekken. Hoe doe ik dat volgens de AVG?",
                  "Uw toestemming is ingetrokken; u ontvangt geen marketingberichten meer. Het intrekken van uw "
                  "toestemming heeft geen invloed op uw lopende contracten.",
                  "Kan ik mijn toestemming voor marketing later opnieuw geven?",
                  "Ja, u kunt uw toestemming op elk moment opnieuw geven via het klantenportaal of de "
                  "klantendienst."),
            Topic("Functionaris voor gegevensbescherming",
                  "Wie is bij u verantwoordelijk voor de bescherming van mijn persoonsgegevens en hoe neem ik "
                  "contact op?",
                  "Onze functionaris voor gegevensbescherming behandelt alle vragen over uw persoonsgegevens. U "
                  "bereikt hem via het contactformulier privacy op onze website.",
                  "Kan ik over mijn persoonsgegevens ook terecht bij de toezichthouder?",
                  "Ja, u kunt zich altijd tot de Gegevensbeschermingsautoriteit wenden, ook als u eerst ons hebt "
                  "gecontacteerd over uw persoonsgegevens."),
        ),
        details=("Mijn klantnummer kan ik op verzoek bezorgen.",
                 "Ik verwacht een antwoord binnen de wettelijke termijn.",
                 "U kunt mij bereiken op {phone}.",
                 "Graag een schriftelijke bevestiging van mijn verzoek."),
    ),
    ("compliance", "fr"): Pool(
        topics=(
            Topic("Accès à mes données personnelles",
                  "Conformément au RGPD, je souhaite savoir quelles données personnelles vous conservez à mon sujet "
                  "et à quelles fins.",
                  "Nous vous transmettrons sous un mois un aperçu de vos données personnelles, des finalités du "
                  "traitement et des durées de conservation, comme le prévoit le RGPD.",
                  "Recevrai-je cet aperçu de mes données personnelles dans un format lisible ?",
                  "Oui, vous recevrez l'aperçu en pdf sur le portail client sécurisé ; le droit d'accès est "
                  "gratuit."),
            Topic("Rectification de mes données",
                  "Mon adresse et mon numéro de téléphone dans vos fichiers sont obsolètes. Je souhaite faire "
                  "rectifier ces données personnelles selon le RGPD.",
                  "Vos données personnelles ont été rectifiées. Le droit de rectification est gratuit et s'applique "
                  "à tous vos contrats chez nous.",
                  "Mes anciennes données personnelles sont-elles aussi effacées de vos systèmes ?",
                  "Oui, les données obsolètes sont remplacées et ne sont plus utilisées ; seul l'historique imposé "
                  "par la loi est conservé."),
            Topic("Retrait du consentement marketing",
                  "Je souhaite retirer mon consentement aux e-mails marketing. Comment faire selon le RGPD ?",
                  "Votre consentement est retiré ; vous ne recevrez plus de messages marketing. Ce retrait n'a "
                  "aucun effet sur vos contrats en cours.",
                  "Puis-je redonner mon consentement au marketing plus tard ?",
                  "Oui, vous pouvez redonner votre consentement à tout moment via le portail client ou le service "
                  "clientèle."),
            Topic("Délégué à la protection des données",
                  "Qui est responsable de la protection de mes données personnelles chez vous et comment le "
                  "contacter ?",
                  "Notre délégué à la protection des données traite toutes les questions relatives à vos données "
                  "personnelles. Vous pouvez le joindre via le formulaire vie privée de notre site.",
                  "Puis-je aussi m'adresser à l'autorité de contrôle au sujet de mes données personnelles ?",
                  "Oui, vous pouvez toujours saisir l'Autorité de protection des données, même après nous avoir "
                  "contactés au sujet de vos données personnelles."),
        ),
        details=("Je peux vous communiquer mon numéro de client sur demande.",
                 "J'attends une réponse dans le délai légal.",
                 "Vous pouvez me joindre au {phone}.",
                 "Merci de me confirmer ma demande par écrit."),
    ),
    ("compliance", "en"): Pool(
        topics=(
            Topic("Access to my personal data",
                  "Under the GDPR I would like to know which personal data you hold about me and for what purposes.",
                  "Within one month we will send you an overview of your personal data, the purposes of processing "
                  "and the retention periods, as the GDPR requires.",
                  "Will I receive that overview of my personal data in a readable format?",
                  "Yes, you will receive the overview as a pdf in the secure customer portal; the right of access "
                  "is free of charge."),
            Topic("Correction of my data",
                  "My address and phone number in your records are out of date. I would like this personal data "
                  "corrected under the GDPR.",
                  "Your personal data has been corrected. The right to rectification is free of charge and applies "
                  "to all your contracts with us.",
                  "Is my old personal data also deleted from your systems?",
                  "Yes, the outdated personal data is replaced and no longer used; only the history required by law "
                  "is kept."),
            Topic("Withdrawing marketing consent",
                  "I want to withdraw my consent to marketing e-mails. How do I do that under the GDPR?",
                  "Your consent has been withdrawn; you will no longer receive marketing messages. Withdrawing "
                  "consent has no effect on your current contracts.",
                  "Can I give my consent to marketing again later?",
                  "Yes, you can give your consent again at any time through the customer portal or customer "
                  "service."),
            Topic("Data protection officer",
                  "Who is responsible for protecting my personal data at your company and how do I contact them?",
                  "Our data protection officer handles all questions about your personal data. You can reach them "
                  "through the privacy contact form on our website.",
                  "Can I also turn to the supervisory authority about my personal data?",
                  "Yes, you can always contact the Data Protection Authority about your personal data, even after "
                  "contacting us first."),
        ),
        details=("I can provide my customer number on request.",
                 "I expect an answer within the legal deadline.",
                 "You can reach me on {phone}.",
                 "Please confirm my request in writing."),
    ),
}

THREADS_PER_POOL: dict[tuple[str, str], int] = {
    # ~115 threads in total: volume goes where the dataset is thin (onboarding,
    # complaints, compliance - English included), not where it is already thick.
    ("onboarding", "nl"): 16, ("onboarding", "fr"): 14, ("onboarding", "en"): 12,
    ("complaints", "nl"): 9, ("complaints", "fr"): 9, ("complaints", "en"): 10,
    ("billing", "nl"): 6, ("billing", "fr"): 6, ("billing", "en"): 6,
    ("compliance", "nl"): 7, ("compliance", "fr"): 7, ("compliance", "en"): 8,
}
NO_REPLY_POOLS: Final = (("complaints", "nl"), ("onboarding", "fr"))  # first thread of each: no agent reply

GDPR: dict[str, dict[str, str]] = {
    "nl": {
        "subject": "Inzageverzoek AVG",
        "in1": "Op basis van artikel 15 van de AVG wil ik mijn inzagerecht uitoefenen. Ik wil weten welke "
               "persoonsgegevens u over mij bewaart, voor welke doeleinden u ze verwerkt en aan wie u ze doorgeeft. "
               "Graag ook de bewaartermijn van mijn persoonsgegevens.",
        "out1": "Wij hebben uw verzoek tot inzage in uw persoonsgegevens goed ontvangen. Om uw identiteit te "
                "controleren vragen wij u ons uw rijksregisternummer en een kopie van uw identiteitskaart te bezorgen. "
                "Daarna beantwoorden wij uw inzageverzoek binnen de wettelijke termijn van één maand, conform de AVG.",
        "in2": "Mijn rijksregisternummer is {rrn}. In bijlage vindt u een kopie van mijn identiteitskaart. Ik wil "
               "ook mijn toestemming voor marketing intrekken.",
        "out2": "In bijlage vindt u het overzicht van uw persoonsgegevens: identificatiegegevens, contactgegevens en "
                "gegevens over uw polissen. Wij verwerken ze voor het beheer van uw contracten en op basis van "
                "wettelijke verplichtingen. Uw toestemming voor marketing werd ingetrokken. Voor vragen over "
                "gegevensbescherming kunt u terecht bij onze functionaris voor gegevensbescherming.",
    },
    "fr": {
        "subject": "Demande d'accès RGPD",
        "in1": "Conformément à l'article 15 du RGPD, je souhaite exercer mon droit d'accès. Je voudrais savoir "
               "quelles données personnelles vous conservez à mon sujet, dans quel but vous les traitez et à qui "
               "vous les communiquez, ainsi que leur durée de conservation.",
        "out1": "Nous avons bien reçu votre demande de droit d'accès à vos données personnelles. Afin de vérifier "
                "votre identité, merci de nous communiquer votre numéro de registre national et une copie de votre "
                "carte d'identité. Nous répondrons ensuite dans le délai légal d'un mois prévu par le RGPD.",
        "in2": "Mon numéro de registre national est le {rrn}. Vous trouverez ma carte d'identité en annexe. Je "
               "retire également mon consentement à l'utilisation de mes données pour le marketing.",
        "out2": "Vous trouverez en annexe l'aperçu de vos données personnelles : données d'identification, "
                "coordonnées et données relatives à vos contrats. Elles sont traitées pour la gestion de vos "
                "contrats et le respect de nos obligations légales. Votre consentement au marketing a été retiré. "
                "Notre délégué à la protection des données reste à votre disposition.",
    },
    "en": {
        "subject": "GDPR subject access request",
        "in1": "Under Article 15 of the GDPR I would like to exercise my right of access. Please tell me which "
               "personal data you hold about me, for what purposes you process it, who you share it with and how "
               "long you keep it.",
        "out1": "We have received your request for access to your personal data. To verify your identity, please "
                "send us your national register number and a copy of your identity card. We will then answer "
                "within the one-month period set by the GDPR.",
        "in2": "My national register number is {rrn}. Please find a copy of my identity card attached. I also "
               "withdraw my consent to the use of my data for marketing.",
        "out2": "Please find attached an overview of your personal data: identification data, contact details and "
                "data about your policies. We process it to manage your contracts and to meet legal obligations. "
                "Your consent to marketing has been withdrawn. Our data protection officer remains available for "
                "any privacy question.",
    },
}
# Two more GDPR threads (erasure, consent withdrawal) so compliance clears
# min_train_per_domain with margin. Their replies inform, not attach.
GDPR_EXTRA: tuple[tuple[str, dict[str, str]], ...] = (
    ("nl", {
        "subject": "Verwijdering persoonsgegevens (AVG)",
        "in1": "Ik ben sinds vorig jaar geen klant meer en wil dat u mijn persoonsgegevens verwijdert, zoals de "
               "AVG mij toelaat. Kan dat volledig?",
        "out1": "Uw verzoek tot wissing van uw persoonsgegevens is geregistreerd. Volgens de AVG wissen wij alle "
                "gegevens die wij niet wettelijk moeten bewaren. Gegevens over uw vroegere polissen bewaren wij nog "
                "tien jaar voor onze wettelijke verplichtingen; daarna worden ook die gewist.",
        "in2": "Bedankt voor de uitleg. Gebruikt u mijn persoonsgegevens in tussentijd nog voor marketing?",
        "out2": "Nee. Uw toestemming voor marketing is ingetrokken en de bewaarde persoonsgegevens worden "
                "uitsluitend gebruikt voor onze wettelijke verplichtingen. U kunt altijd uw inzagerecht uitoefenen "
                "om dat te controleren.",
    }),
    ("fr", {
        "subject": "Retrait du consentement RGPD",
        "in1": "Je souhaite retirer mon consentement au traitement de mes données personnelles à des fins "
               "commerciales. Comment le RGPD s'applique-t-il dans mon cas ?",
        "out1": "Le retrait de votre consentement a été enregistré aujourd'hui. Conformément au RGPD, vos données "
                "personnelles ne seront plus utilisées pour des offres commerciales. Ce retrait n'affecte pas la "
                "gestion de vos contrats en cours.",
        "in2": "Merci pour ces précisions. Combien de temps conservez-vous encore mes données personnelles ?",
        "out2": "Vos données personnelles sont conservées pendant la durée de vos contrats, puis dix ans pour "
                "respecter nos obligations légales. Les données liées au marketing sont effacées immédiatement, "
                "comme le prévoit le RGPD.",
    }),
)

PLANTED_CLAIM = {
    "nl": ("Schadedossier {claim_no}",
           "Ik contacteer u over mijn schadedossier {claim_no}, het schadegeval van {incident}. Kunt u mij laten "
           "weten of de expert zijn verslag al heeft ingediend en wanneer ik de vergoeding mag verwachten?",
           "Het verslag van de expert voor schadedossier {claim_no} is ontvangen. De vergoeding wordt binnen tien "
           "werkdagen uitbetaald op uw rekening."),
    "fr": ("Dossier sinistre {claim_no}",
           "Je vous contacte au sujet de mon dossier sinistre {claim_no}, le sinistre du {incident}. Pouvez-vous me "
           "dire si l'expert a remis son rapport et quand je peux espérer l'indemnisation ?",
           "Le rapport d'expertise pour le dossier sinistre {claim_no} a été reçu. L'indemnisation sera versée sur "
           "votre compte dans les dix jours ouvrables."),
    "en": ("Claim {claim_no}",
           "I am writing about my claim {claim_no}, the incident of {incident}. Could you tell me whether the loss "
           "adjuster has filed the report and when I can expect the compensation?",
           "The loss adjuster's report for claim {claim_no} has been received. The compensation will be paid into "
           "your account within ten working days."),
}

# --- Model -----------------------------------------------------------------------------


@dataclass
class Message:
    direction: str
    created_at: datetime
    channel: str
    body_csv: str
    body_eml: str
    html_only: bool = False


@dataclass
class Thread:
    thread_id: str
    lang: str
    subject: str
    customer_name: str
    customer_email: str
    agent_name: str
    messages: list[Message] = field(default_factory=list)
    pii: set[tuple[str, str]] = field(default_factory=set)


def _ascii(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().replace(" ", "")


def _sha(*parts: object) -> str:
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _mojibake(text: str) -> str | None:
    """UTF-8 read as Windows-1252, only if ftfy restores it exactly."""
    try:
        broken = text.encode("utf-8").decode("cp1252")
    except UnicodeDecodeError:
        return None
    return broken if broken != text and ftfy.fix_text(broken) == text else None


class _Builder:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.threads: list[Thread] = []

    def new_thread(self, lang: str, subject: str, customer: tuple[str, str] | None = None) -> Thread:
        index = len(self.threads)
        first, last = customer or self.rng.choice(CUSTOMERS[lang])
        agent = self.rng.choice(AGENTS[lang])
        thread = Thread(
            thread_id=f"THR-{index + 1:04d}",
            lang=lang,
            subject=subject,
            customer_name=f"{first} {last}",
            customer_email=f"{_ascii(first)}.{_ascii(last)}{index + 1}@example.be",
            agent_name=f"{agent[0]} {agent[1]}",
        )
        self.threads.append(thread)
        return thread

    def slots(self, thread: Thread) -> dict[str, str]:
        city, postcode = self.rng.choice(CITIES[thread.lang])
        return {
            "date": (BASE_TIME + timedelta(days=self.rng.randint(5, 60))).strftime("%d/%m/%Y"),
            "address": f"{self.rng.choice(STREETS[thread.lang])} {self.rng.randint(2, 180)}, {postcode} {city}",
            "city": city,
            "policy_no": f"POL-{self.rng.randint(100000, 999999)}",
            "phone": f"+32 4{self.rng.randint(70, 99)} {self.rng.randint(10, 99)} {self.rng.randint(10, 99)} "
                     f"{self.rng.randint(10, 99)}",
            "iban": valid_be_iban(self.rng),
            "rrn": valid_rrn(self.rng),
            "amount": f"{self.rng.randint(180, 1600)},{self.rng.randint(0, 99):02d}",
        }

    def _when(self, thread: Thread) -> datetime:
        index = int(thread.thread_id.split("-")[1]) - 1
        start = BASE_TIME + timedelta(days=index // 2, hours=index % 8)
        if not thread.messages:
            return start
        return thread.messages[-1].created_at + timedelta(hours=self.rng.randint(2, 30), minutes=self.rng.randint(0, 59))

    def inbound(self, thread: Thread, text: str, phone: str, channel: str = "email") -> None:
        lang = thread.lang
        opener = self.rng.choice(OPENERS[lang])
        csv_sig = CUSTOMER_SIG_CSV[lang].format(name=thread.customer_name, phone=phone)
        eml_sig = CUSTOMER_SIG_EML[lang].format(name=thread.customer_name)
        thread.pii |= {("PERSON", thread.customer_name), ("PHONE", phone)}
        thread.messages.append(Message(
            "inbound", self._when(thread), channel,
            f"{opener}\n\n{text}\n\n{csv_sig}", f"{opener}\n\n{text}\n\n{eml_sig}",
        ))

    def outbound(self, thread: Thread, text: str) -> None:
        lang = thread.lang
        greeting = GREETING[lang].format(name=thread.customer_name)
        index = len(thread.messages)
        csv_ref = REFERENCE[lang].format(ref=f"TCK-{_sha(self.seed, thread.thread_id, index)[:6].upper()}")
        eml_ref = REFERENCE[lang].format(ref=f"MSG-{_sha('mailbox', self.seed, thread.thread_id, index)[:8].upper()}")
        body = f"{greeting}\n\n{text}"
        thread.pii |= {("PERSON", thread.customer_name), ("PERSON", thread.agent_name)}
        thread.messages.append(Message(
            "outbound", self._when(thread), "email",
            f"{body}\n\n{csv_ref}\n\n{AGENT_SIG_CSV[lang].format(agent=thread.agent_name)}\n\n{DISCLAIMER[lang]}",
            f"{body}\n\n{eml_ref}\n\n{AGENT_SIG_EML[lang].format(agent=thread.agent_name)}\n\n{DISCLAIMER[lang]}",
        ))

    def auto_ack(self, thread: Thread) -> None:
        ref = REFERENCE[thread.lang].format(ref=f"TCK-{_sha(self.seed, thread.thread_id)[:6].upper()}")
        text = f"{AUTO_ACK[thread.lang]}\n\n{ref}\n\n{DISCLAIMER[thread.lang]}"
        thread.messages.append(Message("outbound", self._when(thread), "email", text, text))


def build_threads(seed: int) -> list[Thread]:
    b = _Builder(seed)

    # Topic pools: a unique (core, detail) opening per thread, so two different customers
    # never send the identical first message (no accidental near-duplicates or group collisions).
    for (domain, lang), count in THREADS_PER_POOL.items():
        pool = POOLS[(domain, lang)]
        combos = [(t, d) for t in range(len(pool.topics)) for d in range(len(pool.details))]
        for n, (t, d) in enumerate(b.rng.sample(combos, count)):
            topic = pool.topics[t]
            thread = b.new_thread(lang, topic.subject)
            slots = b.slots(thread)
            template = f"{topic.request} {pool.details[d]}"
            text = template.format(**slots)
            for key, pii_type in (("address", "ADDRESS"), ("iban", "IBAN"), ("phone", "PHONE")):
                if "{" + key + "}" in template:
                    thread.pii.add((pii_type, slots[key]))
            b.inbound(thread, text, slots["phone"], channel=b.rng.choice(("email", "email", "webform")))
            if n == 0 and (domain, lang) in NO_REPLY_POOLS:
                continue  # no agent reply at all
            b.outbound(thread, topic.reply)  # the reply that answers THIS request
            if b.rng.random() < 0.65:
                b.inbound(thread, topic.followup_in, slots["phone"])
                b.outbound(thread, topic.followup_out)

    # GDPR access requests (compliance), two exchanges each.
    for lang, texts in (*GDPR.items(), *GDPR_EXTRA):
        thread = b.new_thread(lang, texts["subject"])
        slots = b.slots(thread)
        thread.pii.add(("NATIONAL_ID", slots["rrn"]))
        b.inbound(thread, texts["in1"], slots["phone"])
        b.outbound(thread, texts["out1"])
        b.inbound(thread, texts["in2"].format(**slots), slots["phone"])
        b.outbound(thread, texts["out2"])

    # Claims that ALSO exist in the CRM claim_notes (cross-source semantic duplicates).
    crm = generate_crm(seed=seed)
    customers = {c.customer_id: c for c in crm.customers}
    policies = {p.policy_id: p for p in crm.policies}
    picked: dict[str, tuple[str, str, str, str]] = {}
    for claim in crm.claims:
        cust = customers[policies[claim.policy_id].customer_id]
        picked.setdefault(cust.language, (claim.claim_number, claim.incident_date.strftime("%d/%m/%Y"),
                                          cust.first_name, cust.last_name))
    for lang in ("nl", "fr", "en"):
        claim_no, incident, first, last = picked[lang]
        subject, ask, answer = PLANTED_CLAIM[lang]
        thread = b.new_thread(lang, subject.format(claim_no=claim_no), customer=(first, last))
        slots = b.slots(thread)
        b.inbound(thread, ask.format(claim_no=claim_no, incident=incident), slots["phone"])
        b.outbound(thread, answer.format(claim_no=claim_no))

    # Noise: two auto-acknowledgement "threads", one empty body.
    for lang in ("nl", "fr"):
        thread = b.new_thread(lang, "Automatisch antwoord" if lang == "nl" else "Réponse automatique")
        b.auto_ack(thread)
    empty = b.new_thread("nl", "(geen onderwerp)")
    empty.messages.append(Message("inbound", b._when(empty), "webform", "", ""))

    # Mojibake in three inbound CSV messages (the .eml exports stay clean).
    candidates = [m for t in b.threads if t.lang in ("nl", "fr") for m in t.messages if m.direction == "inbound"]
    damaged = 0
    for message in candidates:
        broken = _mojibake(message.body_csv)
        if broken is not None:
            message.body_csv = broken
            damaged += 1
            if damaged == 3:
                break
    return b.threads


# --- Writers -----------------------------------------------------------------------------


def write_csv(threads: list[Thread], path: Path) -> int:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    rows = 0
    for thread in threads:
        service = SERVICE_ADDRESS[thread.lang]
        for i, m in enumerate(thread.messages):
            sender, recipient = (thread.customer_email, service) if m.direction == "inbound" else (service,
                                                                                                    thread.customer_email)
            if m.direction == "outbound" and not thread.customer_email:
                recipient = ""
            subject = thread.subject if i == 0 else f"{REPLY_PREFIX[thread.lang]}{thread.subject}"
            ticket_id = f"TCK-{int(thread.thread_id[4:]) * 10 + i:06d}"
            writer.writerow([ticket_id, thread.thread_id, m.created_at.isoformat(), m.channel, m.direction,
                             sender, recipient, subject, m.body_csv])
            rows += 1
    path.write_bytes(buffer.getvalue().encode("utf-8"))
    return rows


def _html(text: str) -> str:
    paragraphs = "".join(f"<p>{escape(p).replace(chr(10), '<br>')}</p>" for p in text.split("\n\n"))
    return f"<html><head><style>p {{ margin: 0 0 1em; }}</style></head><body>{paragraphs}</body></html>"


def _charset(text: str, lang: str) -> str:
    if lang == "fr":
        try:
            text.encode("iso-8859-1")
            return "iso-8859-1"
        except UnicodeEncodeError:
            pass
    return "utf-8"


def write_eml(threads: list[Thread], root: Path, seed: int) -> list[Path]:
    rng = random.Random(seed + 1)
    eligible = [t for t in threads if any(m.direction == "outbound" for m in t.messages)
                and any(m.direction == "inbound" for m in t.messages)]
    chosen = sorted(rng.sample(eligible, N_EML_THREADS), key=lambda t: t.thread_id)
    chosen[0].messages[1].html_only = True  # one agent reply exported as HTML only
    written: list[Path] = []
    for thread in chosen:
        lang = thread.lang
        agent_addr = f"{_ascii(thread.agent_name.split()[0])}.{_ascii(thread.agent_name.split()[-1])}@{AGENT_DOMAIN}"
        ids: list[str] = []
        previous: tuple[Message, str, str] | None = None
        for i, m in enumerate(thread.messages):
            message_id = f"<{_sha(seed, thread.thread_id, i)[:20]}@mail.{AGENT_DOMAIN}>"
            who, addr = ((thread.customer_name, thread.customer_email) if m.direction == "inbound"
                         else (thread.agent_name, agent_addr))
            body = m.body_eml
            if previous is not None:
                prev_message, prev_who, prev_addr = previous
                attribution = ATTRIBUTION[lang].format(
                    when=prev_message.created_at.strftime("%d/%m/%Y %H:%M"), who=prev_who, addr=prev_addr)
                quoted = "\n".join(f"> {line}" if line else ">" for line in prev_message.body_eml.splitlines())
                body = f"{body}\n\n{attribution}\n{quoted}"
                thread.pii.add(("EMAIL", prev_addr) if prev_who == thread.customer_name else ("PERSON", prev_who))
            msg = EmailMessage()
            msg["From"] = f"{who} <{addr}>"
            msg["To"] = f"{thread.customer_email}" if m.direction == "outbound" else SERVICE_ADDRESS[lang]
            msg["Subject"] = thread.subject if i == 0 else f"{REPLY_PREFIX[lang]}{thread.subject}"
            msg["Date"] = format_datetime(m.created_at)
            msg["Message-ID"] = message_id
            if ids:
                msg["In-Reply-To"] = ids[-1]
                msg["References"] = " ".join(ids)
            if m.html_only:
                msg.set_content(_html(body), subtype="html", charset="utf-8")
            else:
                msg.set_content(body, charset=_charset(body, lang))
                msg.add_alternative(_html(body), subtype="html", charset="utf-8")
                msg.set_boundary(f"=_part_{_sha(seed, message_id)[:24]}")
            path = root / f"{thread.thread_id.lower()}_{i:02d}_{_sha(message_id)[:8]}.eml"
            path.write_bytes(msg.as_bytes(policy=policy.default))
            written.append(path)
            ids.append(message_id)
            previous = (m, who, addr)
    return written


def generate_tickets(root: str | Path, seed: int, manifest_path: str | Path | None = None) -> tuple[Path, list[Path]]:
    """Write the CSV export and the .eml re-exports; extend the PII manifest."""
    target = Path(root)
    eml_dir = target / "emails"
    eml_dir.mkdir(parents=True, exist_ok=True)
    for stale in eml_dir.glob("*.eml"):
        stale.unlink()
    threads = build_threads(seed)
    csv_path = target / "tickets_export.csv"
    write_csv(threads, csv_path)
    emls = write_eml(threads, eml_dir, seed)
    if manifest_path is not None:
        values: dict[tuple[str, str], set[str]] = {}
        for thread in threads:
            for item in thread.pii:
                values.setdefault(item, set()).add(csv_thread_uri(thread.thread_id))
        update_manifest(manifest_path, "tickets", seed, values)
    return csv_path, emls
