"""Ticket threads -> Q/A pairs: pairing rules and every rejection reason."""

from typing import Any

from src.contracts import Domain, Language, Record
from src.dataset.builders import BuildSettings, TicketThreadBuilder
from src.dataset.builders.tickets import is_only_a_question, strip_reference_lines
from src.dataset.splitter import SplitConfig, split
from src.dataset.builder import candidate_to_record
from src.processing.clean import MESSAGE_SEPARATOR

QUESTION = "Goedemiddag,\n\nIk verhuis volgende maand. Hoe pas ik het adres op mijn woningverzekering aan?"
ANSWER = ("Beste [PERSON],\n\nWij hebben de adreswijziging verwerkt. U ontvangt binnen vijf werkdagen een "
          "bijvoegsel bij uw polis met het nieuwe adres.\n\nReferentie: TCK-1A2B3C")
FOLLOW_Q = "Bedankt voor de snelle opvolging. Moet ik het bijvoegsel ondertekend terugsturen?"
FOLLOW_A = "Beste [PERSON],\n\nJa, een ondertekend exemplaar van het bijvoegsel volstaat; u mag het per e-mail terugsturen."


def thread(messages: list[tuple[str, str]], record_id: str = "support_mailbox:csv:THR-0001#0",
           domain: Domain = "onboarding", language: Language = "nl") -> Record:
    return Record(
        id=record_id,
        text=MESSAGE_SEPARATOR.join(text for _, text in messages),
        group_id="mail:abc-def",
        source_id=record_id.split("#")[0],
        source_type="email",
        language=language,
        domain=domain,
        metadata={"kind": "thread", "messages": [{"direction": d} for d, _ in messages]},
    )


def build(rec: Record, **settings: Any) -> tuple[list[Any], dict[str, int]]:
    return TicketThreadBuilder(BuildSettings(**settings)).build([rec])


def test_two_exchanges_yield_two_pairs_sharing_one_group() -> None:
    candidates, rejections = build(thread([("inbound", QUESTION), ("outbound", ANSWER),
                                           ("inbound", FOLLOW_Q), ("outbound", FOLLOW_A)]))
    assert rejections == {}
    assert [c.user for c in candidates] == [QUESTION, FOLLOW_Q]
    assert {c.group_id for c in candidates} == {"mail:abc-def"}
    assert all(c.source_ids == ["support_mailbox:csv:THR-0001"] for c in candidates)


def test_pairs_are_whole_messages_and_reference_ids_are_removed() -> None:
    (candidate,), _ = build(thread([("inbound", QUESTION), ("outbound", ANSWER)]))
    assert candidate.user == QUESTION  # never a text cut in half
    assert candidate.assistant == strip_reference_lines(ANSWER)
    assert "TCK-" not in candidate.assistant  # a ticket id the model cannot know


def test_last_inbound_before_the_reply_is_the_question() -> None:
    (candidate,), _ = build(thread([("inbound", "Eerste bericht zonder veel inhoud, maar lang genoeg."),
                                    ("inbound", QUESTION), ("outbound", ANSWER)]))
    assert candidate.user == QUESTION


def test_thread_without_agent_reply_yields_nothing_and_is_counted() -> None:
    candidates, rejections = build(thread([("inbound", QUESTION), ("inbound", FOLLOW_Q)]))
    assert candidates == [] and rejections == {"no_agent_reply": 1}


def test_reply_pointing_to_an_attachment_is_rejected() -> None:
    for reply in ("Beste klant,\n\nZie bijlage voor het overzicht van uw polis en de nieuwe voorwaarden.",
                  "Bonjour,\n\nVeuillez trouver ci-joint le devis demandé pour votre assurance habitation.",
                  "Hello,\n\nPlease see attached the quotation for your new car insurance policy."):
        candidates, rejections = build(thread([("inbound", QUESTION), ("outbound", reply)]))
        assert candidates == [] and rejections == {"refers_to_attachment": 1}, reply


def test_reply_that_only_asks_a_question_is_rejected() -> None:
    reply = "Beste klant,\n\nKunt u ons uw polisnummer bezorgen? Wanneer bent u verhuisd?"
    candidates, rejections = build(thread([("inbound", QUESTION), ("outbound", reply)]))
    assert candidates == [] and rejections == {"answer_is_a_question": 1}
    # A reply that informs AND asks is a real answer.
    assert not is_only_a_question("Wij hebben uw aanvraag ontvangen. Kunt u uw polisnummer bezorgen?")


def test_auto_acknowledgement_reply_is_rejected() -> None:
    reply = "Dit is een automatisch antwoord. Wij hebben uw bericht goed ontvangen en antwoorden snel."
    assert build(thread([("inbound", QUESTION), ("outbound", reply)])) == ([], {"auto_acknowledgement": 1})


def test_too_short_answer_and_question_are_rejected() -> None:
    assert build(thread([("inbound", QUESTION), ("outbound", "Beste [PERSON],\n\nOk, gedaan.")]))[1] == {
        "answer_too_short": 1}
    assert build(thread([("inbound", "Adres?"), ("outbound", ANSWER)]))[1] == {"question_too_short": 1}


def test_placeholder_heavy_example_is_rejected() -> None:
    # 60% placeholders: nothing left to learn from.
    hollow_q = "[PERSON] [ADDRESS] [PHONE] [IBAN] [EMAIL] [PERSON] verhuist naar dit adres"
    hollow_a = "Beste [PERSON], [PERSON] [ADDRESS] [PHONE] [IBAN] [EMAIL] zijn genoteerd voor de polis vandaag"
    assert build(thread([("inbound", hollow_q), ("outbound", hollow_a)]))[1] == {"placeholder_heavy": 1}


def test_reply_without_a_prior_customer_message_is_counted() -> None:
    assert build(thread([("outbound", ANSWER)]))[1] == {"no_customer_message": 1}


def test_misaligned_thread_is_refused_not_guessed() -> None:
    rec = thread([("inbound", QUESTION), ("outbound", ANSWER)])
    rec.metadata["messages"] = [{"direction": "inbound"}]
    assert build(rec) == ([], {"thread_structure_lost": 1})


def test_all_pairs_of_a_thread_stay_on_one_side_of_the_split() -> None:
    examples = []
    for t in range(40):
        rec = thread([("inbound", f"{QUESTION} ({t})"), ("outbound", ANSWER), ("inbound", f"{FOLLOW_Q} {t}"),
                      ("outbound", FOLLOW_A), ("inbound", f"Nog een vraag over het dossier nummer {t}, graag."),
                      ("outbound", f"{FOLLOW_A} Dossier {t} is afgerond.")], record_id=f"m:csv:T{t}#0")
        rec.group_id = f"g{t}"
        candidates, _ = TicketThreadBuilder(BuildSettings()).build([rec])
        assert len(candidates) == 3
        examples += [candidate_to_record(c, f"{rec.id}/qa{k}", "thread") for k, c in enumerate(candidates)]
    result = split(examples, SplitConfig(min_train_per_domain=1, eval_floor_per_domain=3, eval_floor_per_cell=0,
                                         max_share_per_domain=1.0, domains=("onboarding",), languages=("nl",)))
    sides: dict[str, set[str]] = {}
    for side, recs in (("train", result.train), ("eval", result.eval)):
        for r in recs:
            sides.setdefault(r.group_id, set()).add(side)
    assert result.eval and all(len(s) == 1 for s in sides.values())
    assert all(sum(r.group_id == g for r in result.train + result.eval) == 3 for g in sides)
