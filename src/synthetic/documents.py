"""Deterministic, deliberately messy synthetic document folder (PDF + DOCX).

English filenames, Dutch / French / English content. The set contains the
traps a real knowledge base has:
  - long policy conditions with numbered sections (chunking);
  - a v2 (FR) / v3 (NL) pair of the SAME procedure with CONTRADICTORY
    deadlines (5 vs 8 days) - the supersede case, cross-language on purpose;
  - a DOCX table (tariffs), an image-only "scanned" PDF, a DOCX with
    mojibake and mid-sentence line breaks, an almost empty template;
  - sample claims with names, IBANs, phones, addresses and a national
    register number, recorded in the shared PII manifest (origin
    "documents") so the anonymiser's recall is measured on this source too.

Byte-reproducible: fixed creation dates inside the files, fixed ZIP entry
timestamps in DOCX, fixed file modification times (v3 newer than v2).
"""

import os
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from docx import Document
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image, ImageDraw, ImageFont

from src.synthetic.crm import valid_be_iban, valid_rrn
from src.synthetic.manifest import update_manifest

FIXED_CREATED = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc)
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

BlockKind = Literal["heading", "subheading", "para", "table"]


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str = ""
    rows: tuple[tuple[str, ...], ...] = ()


@dataclass
class DocSpec:
    filename: str
    title: str
    blocks: list[Block]
    modified: datetime
    mojibake: bool = False
    scanned_lines: tuple[str, ...] = ()  # image-only PDF when set


@dataclass
class Person:
    name: str
    iban: str
    phone: str
    rrn: str
    address: str


@dataclass
class _Recorder:
    """Collects the PII actually written into each document."""

    values: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    current: str = ""

    def __call__(self, pii_type: str, value: str) -> str:
        self.values.setdefault((pii_type, value), set()).add(f"file:{self.current}")
        return value


def _person(rng: random.Random, first: str, last: str, street: str, city: str, postcode: int) -> Person:
    return Person(
        name=f"{first} {last}",
        iban=valid_be_iban(rng),
        phone=f"+32 4{rng.randint(70, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)} {rng.randint(10, 99)}",
        rrn=valid_rrn(rng),
        address=f"{street} {rng.randint(2, 150)}, {postcode} {city}",
    )


def H(text: str) -> Block:  # noqa: N802 - short DSL for readability
    return Block("heading", text)


def S(text: str) -> Block:  # noqa: N802
    return Block("subheading", text)


def P(text: str) -> Block:  # noqa: N802
    return Block("para", text)


def build_specs(seed: int) -> tuple[list[DocSpec], dict[tuple[str, str], set[str]]]:
    rng = random.Random(seed)
    pii = _Recorder()
    camille = _person(rng, "Camille", "Lemaire", "Rue des Pêcheurs", "Liège", 4000)
    koen = _person(rng, "Koen", "Wouters", "Lindenlaan", "Mechelen", 2800)
    els = _person(rng, "Els", "Jacobs", "Veldstraat", "Hasselt", 3500)
    luc = _person(rng, "Luc", "Fontaine", "Rue Haute", "Namur", 5000)
    marc = _person(rng, "Marc", "Renard", "Avenue des Lilas", "Wavre", 1300)

    def doc(filename: str, build: Callable[[], DocSpec]) -> DocSpec:
        pii.current = filename
        return build()

    specs = [
        doc("policy_conditions_auto_fr.pdf", lambda: DocSpec(
            "policy_conditions_auto_fr.pdf", "Conditions générales - Assurance Auto", [
                H("1. Objet du contrat"),
                P("Les présentes conditions générales décrivent les garanties de votre police d'assurance auto. "
                  "Elles forment, avec les conditions particulières, le contrat qui vous lie à la compagnie. En cas "
                  "de contradiction, les conditions particulières priment sur les conditions générales. La couverture "
                  "s'applique au véhicule désigné ainsi qu'à sa remorque d'un poids maximal de 750 kg."),
                H("2. Garantie responsabilité civile"),
                P("La garantie responsabilité civile couvre les dommages causés à des tiers par le véhicule assuré, "
                  "conformément à la loi du 21 novembre 1989. Cette garantie est obligatoire et ne peut être limitée "
                  "dans le temps. Les passagers sont couverts, à l'exception du conducteur responsable. Le plafond pour "
                  "les dommages matériels est fixé à 100 millions d'euros par événement."),
                H("3. Garantie omnium"),
                P("La couverture omnium complète s'ajoute à la responsabilité civile et protège le véhicule assuré "
                  "lui-même : collision, vol, incendie, bris de vitrage et forces de la nature. La garantie omnium "
                  "partielle couvre uniquement le vol, l'incendie et le bris de vitrage. La valeur couverte est la "
                  "valeur catalogue, options comprises, diminuée de 1 % par mois à partir du septième mois."),
                H("4. Exclusions"),
                P("Sont exclus de la couverture : les dommages survenus lorsque le conducteur est en état d'ivresse, "
                  "les dommages causés intentionnellement, la participation à des courses de vitesse et l'usure "
                  "normale du véhicule. Ces exclusions s'appliquent à toutes les garanties, sauf disposition contraire "
                  "dans les conditions particulières. Une exclusion n'est opposable à la victime que dans les limites "
                  "prévues par la loi."),
                H("5. Franchise"),
                P("Une franchise de 250 EUR est appliquée par événement en omnium complète. Cette franchise est portée "
                  "à 500 EUR lorsque le conducteur a moins de 25 ans ou détient son permis depuis moins de deux ans. "
                  "Aucune franchise n'est appliquée au bris de vitrage réparé par un réparateur agréé. Le niveau de la "
                  "franchise figure dans les conditions particulières."),
                H("6. Obligations en cas de sinistre"),
                P("En cas de sinistre, l'assuré en fait la déclaration dans les huit jours. Il joint le constat "
                  "amiable, les coordonnées des témoins et, le cas échéant, le procès-verbal de la police. L'assuré ne "
                  "peut reconnaître aucune responsabilité sans notre accord. Un expert peut être désigné pour évaluer "
                  "les dégâts avant toute réparation."),
                H("7. Durée et résiliation"),
                P("Le contrat est conclu pour un an et se renouvelle tacitement. Le preneur peut procéder à la "
                  "résiliation au moins trois mois avant la date anniversaire, par lettre recommandée ou par voie "
                  "électronique. En cas de changement d'adresse ou de véhicule, un avenant au contrat est établi "
                  "sans frais."),
                H("8. Prime et paiement"),
                P("La prime annuelle est payable d'avance à l'échéance. Le paiement peut être fractionné par "
                  "domiciliation mensuelle sans supplément. En cas de non-paiement, un rappel est envoyé, suivi d'une "
                  "mise en demeure ; la couverture est suspendue quinze jours après la mise en demeure."),
                H("9. Plaintes"),
                P("Toute plainte peut être adressée à notre service de gestion des plaintes. Si la réponse ne vous "
                  "satisfait pas, vous pouvez vous adresser au médiateur, l'Ombudsman des Assurances. L'introduction "
                  "d'une plainte ne fait pas obstacle à une action en justice."),
            ], datetime(2025, 1, 15, tzinfo=timezone.utc))),
        doc("policy_conditions_home_nl.pdf", lambda: DocSpec(
            "policy_conditions_home_nl.pdf", "Algemene voorwaarden - Woningverzekering", [
                H("1. Voorwerp van de overeenkomst"),
                P("Deze algemene voorwaarden beschrijven de waarborgen van uw woningverzekering. Samen met de "
                  "bijzondere voorwaarden vormen zij de verzekeringsovereenkomst. Bij tegenstrijdigheid hebben de "
                  "bijzondere voorwaarden voorrang. De dekking geldt voor het gebouw en de inhoud op het adres dat in "
                  "de bijzondere voorwaarden is vermeld."),
                H("2. Waarborg brand"),
                P("De waarborg brand dekt brand, ontploffing, blikseminslag en rook. Ook de kosten voor het blussen en "
                  "het opruimen van het puin zijn gedekt tot het verzekerd bedrag. Voor de inhoud geldt de "
                  "nieuwwaarde, voor het gebouw de heropbouwwaarde zonder aftrek van slijtage tot 30 procent."),
                H("3. Waarborg waterschade"),
                P("De waarborg waterschade dekt het wegvloeien van water uit leidingen, verwarmingsinstallaties en "
                  "daken. Niet gedekt zijn de kosten om het defecte onderdeel zelf te herstellen en het opstijgend "
                  "grondwater, zoals vermeld bij de uitsluitingen. De opsporingskosten van het lek zijn gedekt tot "
                  "2.500 EUR per gebeurtenis."),
                H("4. Waarborg diefstal"),
                P("Diefstal en vandalisme door derden zijn gedekt na inbraak of geweldpleging. Voor juwelen geldt een "
                  "maximale dekking van 5.000 EUR. De waarborg diefstal is enkel van toepassing wanneer de woning "
                  "beschikt over de beveiliging die in de bijzondere voorwaarden is vermeld."),
                H("5. Uitsluitingen"),
                P("Uitgesloten van elke waarborg zijn: opzettelijke daden van de verzekerde, oorlog, nucleaire "
                  "risico's en geleidelijke aantasting door slijtage of gebrek aan onderhoud. Een uitsluiting geldt "
                  "enkel indien zij uitdrukkelijk in deze algemene voorwaarden of in de bijzondere voorwaarden is "
                  "opgenomen."),
                H("6. Vrijstelling"),
                P("Per gebeurtenis geldt een wettelijke vrijstelling van 290,03 EUR, geïndexeerd volgens het "
                  "indexcijfer van de consumptieprijzen. Voor de waarborg natuurrampen bedraagt de vrijstelling "
                  "610 EUR. De vrijstelling wordt afgetrokken van het bedrag dat wij betalen."),
                H("7. Verplichtingen bij schade"),
                P("Bij schade verwittigt de verzekerde ons zo snel mogelijk en dient hij uiterlijk binnen acht dagen "
                  "een schadeaangifte in. Hij neemt alle redelijke maatregelen om de schade te beperken en bewaart de "
                  "beschadigde voorwerpen tot de expert ze heeft gezien. Bij diefstal gebeurt binnen 24 uur aangifte "
                  "bij de politie."),
                H("8. Premie"),
                P("De jaarlijkse premie is vooraf betaalbaar op de vervaldag. U kunt de premie via domiciliering in "
                  "maandelijkse schijven betalen. Bij niet-betaling volgt een herinnering en vervolgens een aanmaning; "
                  "de dekking wordt geschorst vijftien dagen na de aanmaning."),
                H("9. Duur en opzegging"),
                P("De overeenkomst loopt één jaar en wordt stilzwijgend verlengd. Opzeggen kan ten minste drie "
                  "maanden voor de jaarlijkse vervaldag. Bij een verhuis wordt de polis via een bijvoegsel aangepast "
                  "aan het nieuwe adres, zonder bijkomende kosten."),
            ], datetime(2025, 1, 20, tzinfo=timezone.utc))),
        doc("policy_conditions_liability_en.pdf", lambda: DocSpec(
            "policy_conditions_liability_en.pdf", "General terms - Family liability insurance", [
                H("1. Scope of the policy"),
                P("These general terms describe the cover provided by your family liability insurance policy. "
                  "Together with the special conditions, they form your contract; where the two differ, the special "
                  "conditions prevail. The policy covers the policyholder and every member of the household living "
                  "under the same roof, including students living away from home."),
                H("2. What is covered"),
                P("The policy covers your civil liability towards third parties in private life, worldwide. Coverage "
                  "includes harm caused by your children, your pets, domestic staff and guests during their stay. The "
                  "sum insured is 25 million euros for bodily injury and 6.2 million euros for property, per event."),
                H("3. Exclusions"),
                P("The policy does not cover intentional acts, liability arising from the use of motor vehicles, "
                  "property in your custody, or professional activities. These exclusions are listed exhaustively: "
                  "anything not excluded is covered within the limits of the sum insured. Each exclusion is applied "
                  "as written in the policy wording."),
                H("4. Deductible"),
                P("A deductible of 250 euros applies per event for property. No deductible applies to bodily injury. "
                  "The deductible is indexed every year and is stated in your special conditions; a higher deductible "
                  "may be chosen in exchange for a lower premium."),
                H("5. Legal assistance coverage"),
                P("Legal assistance coverage pays the costs of defending your interests when your liability is "
                  "invoked, up to 25,000 euros per dispute. You remain free to choose your lawyer. This coverage does "
                  "not apply to disputes about family law or about contracts you signed."),
                H("6. Policy wording and changes"),
                P("Any change to the policy wording is notified in writing at least three months before the renewal "
                  "date. If the terms and conditions change to your disadvantage, you may end the policy before that "
                  "date. The general terms in force on the renewal date apply to the following year."),
                H("7. Duration and cancellation"),
                P("The policy runs for one year and renews automatically. You may cancel it at least three months "
                  "before the renewal date by registered letter or e-mail. A cancellation by the insurer must be "
                  "justified and notified with the same notice period."),
                H("8. Reporting a claim"),
                P("Report any claim to us within eight days. Send us the accident report, the names of witnesses and "
                  "any letter you receive from the injured party. Do not acknowledge liability yourself: our loss "
                  "adjuster handles the settlement with the third party."),
                H("9. Complaints"),
                P("If you are dissatisfied with our service, send your complaint to our complaints department. If our "
                  "answer does not satisfy you, you can contact the Insurance Ombudsman. Filing a complaint does not "
                  "affect your right to take legal action."),
            ], datetime(2025, 2, 3, tzinfo=timezone.utc))),
        doc("procedure_water_damage_v2.docx", lambda: DocSpec(
            "procedure_water_damage_v2.docx", "Procédure dégât des eaux - version 2", [
                H("1. Limiter les dégâts"),
                P("Coupez l'arrivée d'eau, protégez vos biens et prenez des photos des dégâts avant tout nettoyage. "
                  "Conservez les objets endommagés jusqu'au passage de l'expert."),
                H("2. Déclarer le sinistre dans les 5 jours"),
                P("La déclaration de sinistre doit nous parvenir dans les 5 jours calendrier suivant la découverte du "
                  "dégât. Passé ce délai, l'indemnisation peut être réduite si le retard nous a causé un préjudice."),
                H("3. Expertise"),
                P("Pour tout dégât estimé à plus de 2.500 EUR, un expert est désigné. Il vous contacte dans les 48 "
                  "heures et évalue les dégâts sur place."),
                H("4. Indemnisation"),
                P("L'indemnisation est versée dans les 30 jours suivant l'accord sur l'évaluation des dégâts, déduction "
                  "faite de la franchise prévue par les conditions particulières."),
                H("Exemple de déclaration"),
                P(f"Mme {pii('PERSON', camille.name)} signale le 12/03/2025 une fuite dans la salle de bains de son "
                  f"appartement, {pii('ADDRESS', camille.address)}. Elle est joignable au "
                  f"{pii('PHONE', camille.phone)} et souhaite recevoir l'indemnisation sur le compte "
                  f"{pii('IBAN', camille.iban)}."),
            ], datetime(2025, 3, 1, tzinfo=timezone.utc))),
        doc("procedure_waterschade_v3.docx", lambda: DocSpec(
            "procedure_waterschade_v3.docx", "Procedure waterschade - versie 3", [
                H("1. Schade beperken"),
                P("Sluit de watertoevoer af, bescherm uw bezittingen en neem foto's van de schade voor u begint op te "
                  "ruimen. Bewaar beschadigde voorwerpen tot de expert langs is geweest."),
                H("2. Schade aangeven binnen 8 dagen"),
                P("De schadeaangifte moet ons bereiken binnen 8 kalenderdagen na de ontdekking van de schade. Na die "
                  "termijn kan de vergoeding verminderd worden als de laattijdige aangifte ons nadeel berokkent."),
                H("3. Expertise"),
                P("Voor schade boven 2.500 EUR stellen wij een expert aan. Die neemt binnen 48 uur contact met u op en "
                  "stelt de schade ter plaatse vast."),
                H("4. Vergoeding"),
                P("De vergoeding wordt uitbetaald binnen 30 dagen na akkoord over de raming van de schade, na aftrek "
                  "van de vrijstelling uit de bijzondere voorwaarden."),
                H("Voorbeeld van een aangifte"),
                P(f"Dhr. {pii('PERSON', koen.name)} meldt op 04/09/2025 een lek in de badkamer van zijn woning, "
                  f"{pii('ADDRESS', koen.address)}. Hij is bereikbaar op {pii('PHONE', koen.phone)} en vraagt de "
                  f"vergoeding te storten op rekening {pii('IBAN', koen.iban)}."),
            ], datetime(2025, 9, 15, tzinfo=timezone.utc))),
        doc("faq_customers_nl.docx", lambda: DocSpec(
            "faq_customers_nl.docx", "Veelgestelde vragen van klanten", [
                S("Hoe wijzig ik mijn domiciliering?"),
                P("U wijzigt uw domiciliering via het klantenportaal of door ons een nieuw rekeningnummer te bezorgen. "
                  "De wijziging geldt vanaf de volgende vervaldag van uw premie. Een herinnering voor een openstaande "
                  "factuur blijft geldig tot de betaling is ontvangen."),
                S("Hoe vraag ik een offerte aan?"),
                P("Een offerte voor een nieuwe polis vraagt u online aan of via uw makelaar. Wij bezorgen u binnen twee "
                  "werkdagen een prijsofferte. Wilt u onderschrijven, dan volstaat een elektronische handtekening."),
                S("Hoe geef ik een schadegeval aan?"),
                P("Een schadeaangifte doet u binnen acht dagen via het klantenportaal. Voeg foto's van de schade en "
                  "eventueel het aanrijdingsformulier toe; een expert neemt indien nodig contact met u op. Voorbeeld "
                  f"van een correcte aangifte: 'Mijn naam is {pii('PERSON', els.name)}, mijn rekeningnummer is "
                  f"{pii('IBAN', els.iban)} en u kunt mij bereiken op {pii('PHONE', els.phone)}.'"),
                S("Ik ben niet tevreden over de behandeling van mijn dossier. Wat nu?"),
                P("Dien een klacht in bij onze klachtendienst; wij antwoorden binnen vijftien werkdagen. Bent u na ons "
                  "antwoord nog steeds ontevreden, dan kunt u zich wenden tot de Ombudsman van de Verzekeringen."),
                S("Wat doet u met mijn persoonsgegevens?"),
                P("Wij verwerken uw persoonsgegevens volgens de AVG. U hebt inzagerecht en kunt uw toestemming voor "
                  "marketing op elk moment intrekken. Meer informatie vindt u in onze privacyverklaring."),
                S("Ik verhuis. Wat moet ik doen?"),
                P("Meld uw adreswijziging minstens twee weken vooraf. Uw woningverzekering volgt u naar het nieuwe "
                  "adres via een bijvoegsel bij uw polis, zonder nieuwe onderschrijving."),
            ], datetime(2025, 4, 2, tzinfo=timezone.utc))),
        doc("faq_customers_fr.docx", lambda: DocSpec(
            "faq_customers_fr.docx", "Questions fréquentes des clients", [
                S("Puis-je payer ma prime en plusieurs fois ?"),
                P("Oui. La prime annuelle peut être payée par domiciliation mensuelle, sans supplément. Le premier "
                  "prélèvement a lieu à l'échéance indiquée sur votre facture."),
                S("Comment obtenir un devis ?"),
                P("Demandez un devis en ligne ou auprès de votre courtier. La souscription peut ensuite se faire par "
                  "signature électronique, et votre nouveau client reçoit ses documents dans les 48 heures."),
                S("Que faire en cas de sinistre ?"),
                P("Faites votre déclaration de sinistre dans les huit jours, avec le constat amiable et des photos des "
                  "dégâts. Exemple de message complet : 'Je m'appelle "
                  f"{pii('PERSON', luc.name)}, mon numéro de compte est {pii('IBAN', luc.iban)} et je suis joignable "
                  f"au {pii('PHONE', luc.phone)}.'"),
                S("Comment introduire une plainte ?"),
                P("Adressez votre plainte à notre service des réclamations, qui répond dans les quinze jours ouvrables. "
                  "Si vous restez insatisfait, le médiateur de l'Ombudsman des Assurances peut être saisi gratuitement."),
                S("Comment exercer mon droit d'accès ?"),
                P("Conformément au RGPD, vous pouvez exercer votre droit d'accès à vos données personnelles en nous "
                  "écrivant. Vous pouvez aussi retirer votre consentement au marketing à tout moment."),
                S("Comment résilier mon contrat ?"),
                P("La résiliation se fait par lettre recommandée ou par e-mail, au moins trois mois avant l'échéance "
                  "annuelle. Pour un changement d'adresse, un simple avenant suffit : il n'est pas nécessaire de "
                  "résilier."),
            ], datetime(2025, 4, 2, tzinfo=timezone.utc))),
        doc("gdpr_data_subject_rights_fr.pdf", lambda: DocSpec(
            "gdpr_data_subject_rights_fr.pdf", "Vos droits en matière de protection des données (RGPD)", [
                H("1. Responsable du traitement"),
                P("La compagnie est responsable du traitement de vos données personnelles au sens du RGPD. Notre "
                  "délégué à la protection des données veille au respect du règlement et répond à vos questions sur "
                  "la vie privée. Les données personnelles sont traitées uniquement pour la gestion de vos contrats "
                  "et le respect de nos obligations légales."),
                H("2. Droit d'accès"),
                P("Le droit d'accès vous permet d'obtenir une copie des données personnelles que nous détenons à votre "
                  "sujet, ainsi que les finalités du traitement, les destinataires et la durée de conservation. Nous "
                  "répondons à toute demande de droit d'accès dans un délai d'un mois, prolongeable de deux mois."),
                H("3. Rectification et droit à l'oubli"),
                P("Vous pouvez demander la rectification de données inexactes. Le droit à l'oubli vous permet "
                  "d'obtenir l'effacement de vos données personnelles lorsqu'elles ne sont plus nécessaires, sauf si "
                  "la loi nous impose de les conserver, par exemple pour la lutte contre le blanchiment."),
                H("4. Consentement"),
                P("Lorsque le traitement repose sur votre consentement, par exemple pour l'envoi d'offres "
                  "commerciales, vous pouvez retirer ce consentement à tout moment. Le retrait du consentement "
                  "n'affecte pas la licéité du traitement effectué auparavant conformément au RGPD."),
                H("5. Comment exercer vos droits"),
                P("Envoyez votre demande signée avec une copie de votre carte d'identité. Exemple de demande "
                  f"correcte : 'Je soussigné {pii('PERSON', marc.name)}, numéro de registre national "
                  f"{pii('NATIONAL_ID', marc.rrn)}, demande à exercer mon droit d'accès à mes données "
                  "personnelles conformément au RGPD.'"),
                H("6. Autorité de contrôle"),
                P("Si vous estimez que vos droits ne sont pas respectés au sens du RGPD, vous pouvez saisir l'Autorité "
                  "de protection des données. Elle contrôle l'application du RGPD en Belgique et traite les "
                  "demandes relatives aux données personnelles."),
            ], datetime(2025, 5, 12, tzinfo=timezone.utc))),
        doc("gdpr_privacy_notice_nl.pdf", lambda: DocSpec(
            "gdpr_privacy_notice_nl.pdf", "Privacyverklaring (AVG)", [
                H("1. Wie verwerkt uw persoonsgegevens?"),
                P("De maatschappij is verwerkingsverantwoordelijke voor uw persoonsgegevens in de zin van de AVG. Onze "
                  "functionaris voor gegevensbescherming ziet toe op de naleving van de AVG en beantwoordt uw vragen "
                  "over privacy."),
                H("2. Welke persoonsgegevens verzamelen wij?"),
                P("Wij verzamelen identificatiegegevens, contactgegevens, gegevens over uw polissen en, met uw "
                  "uitdrukkelijke toestemming, gezondheidsgegevens. Wij verzamelen enkel de persoonsgegevens die "
                  "noodzakelijk zijn voor het beheer van uw contracten."),
                H("3. Waarom verwerken wij uw persoonsgegevens?"),
                P("De verwerking steunt op de uitvoering van de overeenkomst, op een wettelijke verplichting, op ons "
                  "gerechtvaardigd belang of op uw toestemming. Uw toestemming kunt u op elk ogenblik intrekken; dat "
                  "doet geen afbreuk aan de verwerking die al plaatsvond volgens de AVG."),
                H("4. Hoe lang bewaren wij uw persoonsgegevens?"),
                P("Wij bewaren uw persoonsgegevens zolang de overeenkomst loopt en daarna gedurende de wettelijke "
                  "verjaringstermijn van tien jaar. Gegevens voor marketing op basis van toestemming worden na drie "
                  "jaar zonder contact gewist, conform de AVG."),
                H("5. Uw rechten"),
                P("U hebt recht op inzage, verbetering en wissing van uw persoonsgegevens, en het recht om bezwaar te "
                  "maken tegen direct marketing. Het inzagerecht oefent u uit met een ondertekende aanvraag en een "
                  "kopie van uw identiteitskaart."),
                H("6. Toezicht op gegevensbescherming"),
                P("Vragen over gegevensbescherming richt u aan onze functionaris voor gegevensbescherming. De "
                  "Gegevensbeschermingsautoriteit houdt toezicht op de naleving van de AVG en de bescherming van "
                  "persoonsgegevens in België."),
            ], datetime(2025, 5, 12, tzinfo=timezone.utc))),
        doc("gdpr_retention_policy_en.docx", lambda: DocSpec(
            "gdpr_retention_policy_en.docx", "Data retention policy (GDPR)", [
                H("1. Purpose"),
                P("This policy sets the retention periods for personal data processed by the company, in line with the "
                  "GDPR principle of storage limitation. Personal data is kept no longer than necessary for the purpose "
                  "for which it was collected, and data protection is reviewed at every system change."),
                H("2. Retention periods"),
                P("Personal data of customers is kept for 10 years after the end of the contract. Personal data used "
                  "for marketing on the basis of consent is kept for 3 years after the last contact. Recorded phone "
                  "calls are kept for 6 months, unless needed as evidence under the GDPR."),
                H("3. Data subject requests"),
                P("A request under the right of access or the right to be forgotten does not override a legal "
                  "retention obligation. In that case the personal data is restricted and no longer used for any "
                  "other purpose, and the data subject is informed in line with the GDPR."),
                H("4. Deletion and anonymisation"),
                P("At the end of the retention period, personal data is deleted or irreversibly anonymised. Backups "
                  "are overwritten within 90 days. The data protection officer audits the deletion logs every year."),
                H("5. Responsibilities"),
                P("Each department head applies this policy to the systems under their control. Questions about data "
                  "protection or privacy can be sent to the data protection officer. This GDPR policy is reviewed "
                  "every two years."),
            ], datetime(2025, 6, 30, tzinfo=timezone.utc))),
        doc("tariffs_2026.docx", lambda: DocSpec(
            "tariffs_2026.docx", "Tarifs 2026 - primes annuelles", [
                P("Les primes ci-dessous s'appliquent aux contrats souscrits ou renouvelés à partir du 1er janvier 2026. "
                  "Elles sont payables annuellement ou par domiciliation mensuelle sans supplément."),
                Block("table", rows=(
                    ("Produit", "Formule", "Prime annuelle", "Franchise"),
                    ("Auto", "Responsabilité civile", "412 EUR", "-"),
                    ("Auto", "Omnium complète", "845 EUR", "250 EUR"),
                    ("Habitation", "Incendie et dégâts des eaux", "298 EUR", "290 EUR"),
                    ("Familiale", "Responsabilité civile vie privée", "96 EUR", "250 EUR"),
                )),
                P("Une réduction de 10 % sur la prime est accordée à partir de trois contrats. Les primes sont "
                  "indexées chaque année au 1er janvier."),
            ], datetime(2025, 11, 20, tzinfo=timezone.utc))),
        doc("internal_claims_note_scanned.pdf", lambda: DocSpec(
            "internal_claims_note_scanned.pdf", "", [], datetime(2025, 7, 8, tzinfo=timezone.utc),
            scanned_lines=(
                "NOTE INTERNE - DOSSIER SINISTRE 2025-0417",
                "Rapport d'expertise recu le 07/07/2025.",
                "Accord sur l'evaluation des degats.",
                "Indemnisation a verser, dossier a cloturer.",
            ))),
        doc("broken_encoding_note_nl.docx", lambda: DocSpec(
            "broken_encoding_note_nl.docx", "Interne notitie schadedossier", [
                P("De klant werd op 12 maart geïnformeerd dat de"),
                P("schadevergoeding pas na de expertise wordt"),
                P("uitbetaald. Het verslag van de expert bevat twee"),
                P("ideeën om de herstelling te versnellen: één"),
                P("aannemer aanstellen en de werken vóór de winter"),
                P("laten beëindigen."),
                P(""),
                P("De schadeaangifte is volledig en het dossier blijft open tot de vergoeding"),
                P("is uitbetaald aan de verzekerde."),
            ], datetime(2025, 8, 1, tzinfo=timezone.utc), mojibake=True)),
        doc("empty_template.docx", lambda: DocSpec(
            "empty_template.docx", "Template", [P(""), P("")], datetime(2025, 1, 2, tzinfo=timezone.utc))),
    ]
    return specs, pii.values


# --- Writers --------------------------------------------------------------------


def _mojibake(text: str) -> str:
    """UTF-8 bytes wrongly decoded as Windows-1252: 'geïnformeerd' -> 'geÃ¯nformeerd'."""
    return text.encode("utf-8").decode("cp1252")


def _write_pdf(spec: DocSpec, path: Path) -> None:
    pdf = FPDF()
    pdf.set_creation_date(FIXED_CREATED)
    pdf.set_creator("synthetic-documents")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    if spec.scanned_lines:
        # Image only: rendered text, no text layer at all.
        image = Image.new("L", (1240, 600), 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=32)
        for i, line in enumerate(spec.scanned_lines):
            draw.text((60, 60 + i * 70), line, fill=0, font=font)
        pdf.image(image, x=10, y=10, w=190)
    else:
        pdf.set_font("Helvetica", "B", 15)
        pdf.multi_cell(0, 8, spec.title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
        for block in spec.blocks:
            if block.kind in ("heading", "subheading"):
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 6, block.text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.set_font("Helvetica", "", 10)
                # Deliberately narrow column: body lines wrap mid-sentence.
                pdf.multi_cell(150, 5, block.text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)
    pdf.output(str(path))


def _write_docx(spec: DocSpec, path: Path) -> None:
    document = Document()
    props = document.core_properties
    props.author = "synthetic-documents"
    props.title = spec.title
    props.created = FIXED_CREATED.replace(tzinfo=None)
    props.modified = FIXED_CREATED.replace(tzinfo=None)
    props.last_modified_by = "synthetic-documents"
    props.revision = 1
    fix = _mojibake if spec.mojibake else (lambda s: s)
    document.add_heading(spec.title, level=0)
    for block in spec.blocks:
        if block.kind == "heading":
            document.add_heading(fix(block.text), level=1)
        elif block.kind == "subheading":
            document.add_heading(fix(block.text), level=2)
        elif block.kind == "table":
            table = document.add_table(rows=len(block.rows), cols=len(block.rows[0]))
            for r, row in enumerate(block.rows):
                for c, value in enumerate(row):
                    table.cell(r, c).text = value
        else:
            document.add_paragraph(fix(block.text))
    document.save(str(path))
    _normalise_zip(path)


def _normalise_zip(path: Path) -> None:
    """Rewrite the DOCX container with fixed entry timestamps (byte-reproducible)."""
    with zipfile.ZipFile(path) as source:
        entries = [(info.filename, source.read(info.filename)) for info in source.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for name, data in entries:
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            target.writestr(info, data)


def generate_documents(root: str | Path, seed: int, manifest_path: str | Path | None = None) -> list[Path]:
    """Write every document under ``root``; extend the PII manifest if given."""
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    specs, pii_values = build_specs(seed)
    written: list[Path] = []
    for spec in specs:
        path = target / spec.filename
        if path.suffix == ".pdf":
            _write_pdf(spec, path)
        else:
            _write_docx(spec, path)
        ts = spec.modified.timestamp()
        os.utime(path, (ts, ts))
        written.append(path)
    if manifest_path is not None:
        update_manifest(manifest_path, "documents", seed, pii_values)
    return written
