"""Checksum / structure validators confirming regex candidates.

A regex candidate is replaced only when its validator returns True; this
is what keeps the deliberately wide patterns precise. Pure functions.
"""

import re
from datetime import date

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]")
_NON_DIGIT = re.compile(r"\D")
_DATE_LIKE = re.compile(r"^\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\s*$")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")

# IBAN lengths for the countries a Belgian insurer realistically sees.
_IBAN_LENGTHS = {"BE": 16, "NL": 18, "LU": 20, "FR": 27, "DE": 22}


def _iban_mod97(compact: str) -> bool:
    """ISO 13616: move the first 4 chars to the end, A=10..Z=35, mod 97 == 1."""
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def is_valid_be_iban(value: str) -> bool:
    """Belgian IBAN: BE + 14 digits. Validate by rearranging (move the
    first 4 chars to the end), converting letters to digits
    (A=10 ... Z=35) and checking mod 97 == 1."""
    compact = _NON_ALNUM.sub("", value).upper()
    return bool(re.fullmatch(r"BE\d{14}", compact)) and _iban_mod97(compact)


def is_valid_iban(value: str) -> bool:
    """Belgian IBAN, or a neighbouring-country IBAN of the right length (mod 97)."""
    compact = _NON_ALNUM.sub("", value).upper()
    if compact.startswith("BE"):
        return is_valid_be_iban(compact)
    expected = _IBAN_LENGTHS.get(compact[:2])
    return (
        expected is not None
        and len(compact) == expected
        and bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact))
        and _iban_mod97(compact)
    )


def _plausible_birth_date(yy: int, mm: int, dd: int) -> bool:
    # BIS numbers add 20 or 40 to the month; month/day 00 (unknown) is rejected.
    for offset in (0, 20, 40):
        month = mm - offset
        if 1 <= month <= 12:
            try:
                # Year is unknown at this point: 2000 is a leap year, so 29/02 passes.
                date(2000, month, dd)
            except ValueError:
                return False
            return True
    return False


def is_valid_rrn(value: str) -> bool:
    """Belgian national register number, 11 digits YYMMDDXXXCC.
    CC = 97 - (first 9 digits mod 97) for births before 2000.
    For births from 2000 on, prefix the 9 digits with '2' before the
    modulo. Accept BOTH branches: a number is valid if either matches.
    Also reject impossible month/day values."""
    digits = _NON_DIGIT.sub("", value)
    if len(digits) != 11:
        return False
    first9, check = digits[:9], int(digits[9:])
    if not _plausible_birth_date(int(digits[0:2]), int(digits[2:4]), int(digits[4:6])):
        return False
    before_2000 = 97 - int(first9) % 97
    from_2000 = 97 - int("2" + first9) % 97
    return check in (before_2000, from_2000)


def is_valid_be_phone(value: str) -> bool:
    """Accept +32 4xx xx xx xx (mobile) and 0x/xxx.xx.xx (landline),
    with or without separators. Reject sequences that are plausibly
    amounts, dates or policy numbers."""
    raw = value.strip()
    if "," in raw or _DATE_LIKE.match(raw):
        return False  # amount (1.234,56) or date (01/03/2025)
    compact = re.sub(r"[\s./()\-]", "", raw)
    if compact.startswith("+32"):
        national = compact[3:]
    elif compact.startswith("0032"):
        national = compact[4:]
    elif compact.startswith("0") and not compact.startswith("00"):
        national = compact[1:]
    else:
        return False  # no international or trunk prefix: policy / reference number
    national = national.removeprefix("0")  # "+32 (0)2 ..." style
    if not national.isdigit():
        return False
    if len(national) == 9:
        return national[0] == "4" and national[1] in "56789"  # mobile 045x-049x
    if len(national) == 8:
        return national[0] != "0"  # landline: 1-digit or 2-digit area code
    return False


def is_valid_email(value: str) -> bool:
    return bool(_EMAIL.match(value))


def is_valid_be_postcode(value: str) -> bool:
    """ADDRESS has no checksum: the postcode must be in the Belgian range."""
    digits = _NON_DIGIT.sub("", value)
    return len(digits) == 4 and 1000 <= int(digits) <= 9999
