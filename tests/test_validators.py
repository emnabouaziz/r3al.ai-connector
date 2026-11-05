import pytest

from src.processing.anonymize.validators import (
    is_valid_be_iban,
    is_valid_be_phone,
    is_valid_be_postcode,
    is_valid_email,
    is_valid_iban,
    is_valid_rrn,
)

VALID_BE_IBAN = "BE68 5390 0754 7034"  # textbook valid Belgian IBAN


# --- IBAN -------------------------------------------------------------------------


@pytest.mark.parametrize("value", [VALID_BE_IBAN, "BE68539007547034", "be68 5390 0754 7034"])
def test_valid_be_iban_accepted(value: str) -> None:
    assert is_valid_be_iban(value)


@pytest.mark.parametrize(
    "value",
    [
        "BE68 5390 0754 7035",  # one digit changed
        "BE69 5390 0754 7034",  # check digits changed
        "BE68 5390 0754 703",  # too short
        "NL91 ABNA 0417 1643 00",  # valid IBAN, but not Belgian
    ],
)
def test_invalid_be_iban_rejected(value: str) -> None:
    assert not is_valid_be_iban(value)


def test_generic_iban_accepts_neighbouring_countries() -> None:
    assert is_valid_iban("NL91 ABNA 0417 1643 00")
    assert not is_valid_iban("NL91 ABNA 0417 1643 01")


# --- National register number ------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "85.07.30-033.28",  # born 1985: 97 - (850730033 mod 97) = 28
        "85073003328",
        "05.01.01-123.85",  # born 2005: 97 - (2050101123 mod 97) = 85
    ],
)
def test_valid_rrn_both_centuries(value: str) -> None:
    assert is_valid_rrn(value)


@pytest.mark.parametrize(
    "value",
    [
        "85.07.30-033.29",  # wrong checksum
        "05.01.01-123.56",  # wrong for both branches? 56 is the pre-2000 value -> valid
    ][:1]
    + [
        "85.13.30-033.70",  # month 13, checksum otherwise correct
        "85.02.30-033.90",  # 30 February, checksum otherwise correct
        "85.07.30-033",  # too short
    ],
)
def test_invalid_rrn_rejected(value: str) -> None:
    assert not is_valid_rrn(value)


def test_rrn_branch_choice_is_either_not_both() -> None:
    # 56 is the pre-2000 checksum of 050101123: valid as a 1905 birth, by design.
    assert is_valid_rrn("05.01.01-123.56")
    assert not is_valid_rrn("05.01.01-123.57")


# --- Phone ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["+32 478 12 34 56", "+32478123456", "0478 12 34 56", "0032 478 12 34 56",
     "02/345.67.89", "02 345 67 89", "09 123 45 67", "+32 (0)2 345 67 89"],
)
def test_valid_phone_accepted(value: str) -> None:
    assert is_valid_be_phone(value)


@pytest.mark.parametrize(
    "value",
    [
        "2024 123456",  # policy number
        "policy 2024 123456",
        "01/03/2025",  # date
        "1.234,56",  # amount
        "0478 12 34",  # too short
        "+32 478 12 34 56 7",  # too long
        "+32 312 34 56 78",  # 9 digits not starting with 4: not a mobile
    ],
)
def test_invalid_phone_rejected(value: str) -> None:
    assert not is_valid_be_phone(value)


# --- Email, postcode -------------------------------------------------------------------


def test_email_and_postcode() -> None:
    assert is_valid_email("jan.peeters12@example.be")
    assert not is_valid_email("jan.peeters@localhost")
    assert is_valid_be_postcode("9000") and is_valid_be_postcode("1000")
    assert not is_valid_be_postcode("0999") and not is_valid_be_postcode("12345")
