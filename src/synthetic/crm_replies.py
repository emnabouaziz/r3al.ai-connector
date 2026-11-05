"""Customer-facing agent replies for the synthetic CRM.

One reply per core of each customer pool, in the same order as the pool's
cores (src/synthetic/crm.py POOLS), so a reply always answers the request it
follows - a reply drawn independently of the request produced off-topic
pairs before. These are what the CRM calls author_role="agent": written TO
the customer. Internal notes ("Interne notitie: expert aangesteld...") are
author_role="internal" and never become answers.
"""

from typing import Final

REPLIES: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "claim_description": {
        "nl": (
            "Wij hebben uw schadeaangifte voor de aanrijding goed ontvangen. Een expert neemt binnen twee "
            "werkdagen contact met u op om de schade aan uw wagen vast te stellen.",
            "Wij hebben uw melding van waterschade goed ontvangen. Bewaar de beschadigde voorwerpen tot de expert "
            "langskomt; hij neemt binnen twee werkdagen contact met u op.",
            "Wij hebben uw aangifte van inbraak goed ontvangen. Zodra het proces-verbaal van de politie in ons "
            "bezit is, stellen wij de schade vast en bepalen wij de vergoeding.",
            "Wij hebben uw aangifte goed ontvangen. De schade aan de bril valt onder de waarborg burgerlijke "
            "aansprakelijkheid van uw familiale polis; wij vergoeden de herstelling na ontvangst van de factuur.",
            "Wij hebben uw aangifte van brandschade goed ontvangen. Een expert komt deze week langs om de schade "
            "aan uw keuken vast te stellen.",
        ),
        "fr": (
            "Nous avons bien reçu votre déclaration de sinistre pour l'accident. Un expert vous contactera sous "
            "deux jours ouvrables pour évaluer les dégâts de votre véhicule.",
            "Nous avons bien reçu votre déclaration de dégât des eaux. Conservez les objets endommagés jusqu'au "
            "passage de l'expert, qui vous contactera sous deux jours ouvrables.",
            "Nous avons bien reçu votre déclaration de cambriolage. Dès réception du procès-verbal de la police, "
            "nous évaluerons les dégâts et fixerons l'indemnisation.",
            "Nous avons bien reçu votre déclaration. Les dégâts aux lunettes relèvent de la garantie "
            "responsabilité civile de votre assurance familiale ; nous rembourserons la réparation sur facture.",
            "Nous avons bien reçu votre déclaration de sinistre incendie. Un expert passera cette semaine pour "
            "évaluer les dégâts dans votre cuisine.",
        ),
        "en": (
            "We have received your claim for the collision. A loss adjuster will contact you within two working "
            "days to assess the damage to your car.",
            "We have received your water damage claim. Please keep the damaged items until the loss adjuster "
            "visits; they will contact you within two working days.",
            "We have received your burglary claim. As soon as we have the police report, we will assess the "
            "damage and determine the compensation.",
            "We have received your claim. The damage to the glasses falls under the liability cover of your "
            "family policy; we will pay for the repair once we receive the invoice.",
            "We have received your fire damage claim. A loss adjuster will visit this week to assess the damage "
            "to your kitchen.",
        ),
    },
    "claim_note_customer": {
        "nl": (
            "Bedankt voor de bijkomende foto's. Ze zijn toegevoegd aan uw schadedossier en doorgegeven aan de "
            "expert.",
            "De vergoeding wordt uitbetaald zodra het verslag van de expert is goedgekeurd, meestal binnen tien "
            "werkdagen.",
            "Het bestek van de garage is ontvangen. De expert keurt het goed of stelt een aanpassing voor; daarna "
            "mag de herstelling starten.",
            "Het verslag van de expert over de schade wordt binnen vijf werkdagen verwacht. Wij sturen het u door "
            "zodra het binnen is.",
            "Wij hebben uw rekeningnummer genoteerd; de schadevergoeding voor uw schadedossier wordt op die "
            "rekening gestort zodra ze is goedgekeurd.",
        ),
        "fr": (
            "Merci pour les photos supplémentaires. Elles ont été ajoutées à votre dossier sinistre et "
            "transmises à l'expert.",
            "L'indemnisation est versée dès que le rapport d'expertise est validé, généralement sous dix jours "
            "ouvrables.",
            "L'estimation du garage est bien reçue. L'expert la valide ou propose une adaptation ; la réparation "
            "peut ensuite commencer.",
            "Le rapport de l'expert est attendu dans les cinq jours ouvrables. Nous vous le transmettrons dès "
            "réception.",
            "Nous avons noté votre numéro de compte ; l'indemnité y sera versée dès qu'elle sera approuvée.",
        ),
        "en": (
            "Thank you for the additional photos. They have been added to your claim and passed on to the loss "
            "adjuster.",
            "The compensation is paid as soon as the adjuster's report is approved, usually within ten working "
            "days.",
            "We have received the garage estimate. The adjuster will approve it or propose an adjustment; the "
            "repair can start after that.",
            "The adjuster's report is expected within five working days. We will forward it to you as soon as it "
            "arrives.",
            "We have recorded your account number; the compensation will be paid into that account once it is "
            "approved.",
        ),
    },
    "complaint_note": {
        "nl": (
            "Wij betreuren dat u ontevreden bent en zo lang moest wachten. Uw klacht krijgt voorrang en een "
            "dossierbeheerder neemt morgen contact met u op.",
            "Uw formele klacht is geregistreerd bij onze klachtendienst. U ontvangt binnen vijf werkdagen een "
            "schriftelijk antwoord.",
            "Uw vraag tot escalatie is aanvaard: een verantwoordelijke behandelt uw klacht persoonlijk en neemt "
            "deze week contact met u op.",
            "Wij begrijpen uw ongenoegen. Als ons antwoord op uw klacht u niet tevredenstelt, kunt u zich "
            "kosteloos wenden tot de Ombudsman van de Verzekeringen.",
            "Uw bezwaar tegen de beslissing is geregistreerd als klacht. Een andere dossierbeheerder herbekijkt "
            "het dossier en motiveert de beslissing schriftelijk.",
        ),
        "fr": (
            "Nous regrettons votre insatisfaction et ce long délai. Votre plainte est traitée en priorité et un "
            "gestionnaire vous contactera demain.",
            "Votre plainte formelle est enregistrée auprès de notre service des réclamations. Vous recevrez une "
            "réponse écrite sous cinq jours ouvrables.",
            "Votre demande d'escalade est acceptée : un responsable traite personnellement votre plainte et vous "
            "contactera cette semaine.",
            "Nous comprenons votre mécontentement. Si notre réponse à votre plainte ne vous satisfait pas, vous "
            "pouvez saisir gratuitement le médiateur de l'Ombudsman des Assurances.",
            "Votre réclamation contre la décision est enregistrée. Un autre gestionnaire réexaminera le dossier et "
            "motivera la décision par écrit.",
        ),
        "en": (
            "We are sorry you are dissatisfied and had to wait so long. Your complaint is now a priority and a "
            "claims handler will contact you tomorrow.",
            "Your formal complaint has been registered with our complaints team. You will receive a written "
            "answer within five working days.",
            "Your escalation request has been accepted: a manager is handling your complaint personally and will "
            "contact you this week.",
            "We understand that you are unhappy. If our answer to your complaint does not satisfy you, you can "
            "contact the Insurance Ombudsman free of charge.",
            "Your complaint about the decision has been registered. Another claims handler will review the file "
            "and explain the decision in writing.",
        ),
    },
    "invoice_note": {
        "nl": (
            "Wij hebben uw betaling via domiciliering teruggevonden. De betalingsherinnering was een vergissing; "
            "u hoeft niets te doen.",
            "De premie werd geïndexeerd volgens het indexcijfer van de consumptieprijzen; daardoor ligt het "
            "bedrag op de factuur hoger dan vorig jaar.",
            "De dubbele betaling is bevestigd. De terugbetaling wordt binnen vijf werkdagen op uw rekening "
            "gestort.",
            "Dat kan: wij passen de domiciliering aan zodat de premie vanaf de volgende vervaldag maandelijks "
            "wordt geïnd, zonder meerkost.",
            "Excuses voor dit ongemak. Wij sturen u een duplicaat van de factuur en annuleren de aanmaning; u "
            "krijgt dertig dagen om te betalen.",
        ),
        "fr": (
            "Nous avons retrouvé votre paiement par domiciliation. Le rappel était une erreur ; vous n'avez rien "
            "à faire.",
            "La prime a été indexée selon l'indice des prix à la consommation ; c'est pourquoi le montant de la "
            "facture est plus élevé que l'an dernier.",
            "Le double paiement est confirmé. Le remboursement sera versé sur votre compte sous cinq jours "
            "ouvrables.",
            "C'est possible : nous adaptons la domiciliation pour prélever la prime mensuellement à partir de la "
            "prochaine échéance, sans frais.",
            "Toutes nos excuses pour ce désagrément. Nous vous envoyons un duplicata de la facture et annulons la "
            "mise en demeure ; vous disposez de trente jours pour payer.",
        ),
        "en": (
            "We found your payment by direct debit. The payment reminder was a mistake; you do not need to do "
            "anything.",
            "The premium was indexed in line with the consumer price index, which is why the amount on the "
            "invoice is higher than last year.",
            "The double payment is confirmed. The refund will be paid into your account within five working "
            "days.",
            "Yes: we will change the direct debit so that the premium is collected monthly from the next due "
            "date, at no extra cost.",
            "Apologies for the inconvenience. We are sending you a duplicate of the invoice and cancelling the "
            "overdue notice; you have thirty days to pay.",
        ),
    },
}

GREETINGS: Final[dict[str, str]] = {"nl": "Beste {name},", "fr": "Bonjour {name},", "en": "Dear {name},"}
# The department signing a reply depends on what the reply is about: a complaint or
# an invoice question is answered by customer service, not the claims department.
DEPARTMENTS: Final[dict[str, dict[str, str]]] = {
    "claims": {"nl": "Dienst Schadegevallen", "fr": "Service sinistres", "en": "Claims department"},
    "service": {"nl": "Klantendienst", "fr": "Service clientèle", "en": "Customer service"},
}
REPLY_DEPARTMENT: Final[dict[str, str]] = {
    "claim_description": "claims", "claim_note_customer": "claims",
    "complaint_note": "service", "invoice_note": "service",
}
SIGN_OFFS: Final[dict[str, str]] = {
    "nl": "Met vriendelijke groeten,\n{agent}\n{department}",
    "fr": "Cordialement,\n{agent}\n{department}",
    "en": "Kind regards,\n{agent}\n{department}",
}
AGENTS: Final[dict[str, tuple[str, ...]]] = {
    "nl": ("Inge Vermeulen", "Dirk Wouters", "Sara Peeters"),
    "fr": ("Nathalie Lefèvre", "Olivier Martin", "Chloé Dubois"),
    "en": ("Helen Carter", "Mark Dawson", "Ruth Evans"),
}
CLAIM_TYPES: Final[tuple[str, ...]] = ("car_accident", "water_damage", "burglary", "liability", "fire")


def reply_text(kind: str, lang: str, core: int, customer: str, agent: str) -> str:
    """Greeting + the reply to core ``core`` + sign-off (removed by cleaning)."""
    return (
        f"{GREETINGS[lang].format(name=customer)}\n\n{REPLIES[kind][lang][core]}\n\n"
        f"{SIGN_OFFS[lang].format(agent=agent, department=DEPARTMENTS[REPLY_DEPARTMENT[kind]][lang])}"
    )
