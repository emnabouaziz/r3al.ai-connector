"""SQL connector: config validation and row mapping (no DB), plus an optional live test.

The live test runs only when R3AL_TEST_POSTGRES_DSN points to a database
seeded with scripts/seed_crm.py.
"""

import os
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from src.connectors.sql import SqlConfigError, SqlConnector, parse_table_specs, row_to_document

TABLES: dict[str, Any] = {
    "claims": {"id": "claim_id", "text": ["description"], "group_key": "claim_id",
               "group_prefix": "claim", "meta": ["status", "incident_date"]},
    "claim_notes": {"id": "note_id", "text": ["body"], "group_key": "claim_id",
                    "group_prefix": "claim", "meta": ["author_role"]},
}


def test_specs_are_parsed() -> None:
    specs = parse_table_specs(TABLES)
    assert [s.table for s in specs] == ["claims", "claim_notes"]
    assert specs[1].columns == ("note_id", "claim_id", "body", "author_role")


@pytest.mark.parametrize(
    "tables",
    [
        {},
        {"claims; DROP TABLE x": {"id": "claim_id", "text": ["description"]}},
        {"claims": {"id": "claim_id", "text": []}},
        {"claims": {"id": "claim id", "text": ["description"]}},
        {"claims": {"id": "claim_id", "text": ["description\"--"]}},
    ],
)
def test_invalid_specs_are_rejected(tables: dict[str, Any]) -> None:
    with pytest.raises(SqlConfigError):
        parse_table_specs(tables)


def test_dsn_is_required() -> None:
    with pytest.raises(SqlConfigError):
        SqlConnector({"tables": TABLES})


def test_claim_and_its_notes_share_a_group() -> None:
    specs = {s.table: s for s in parse_table_specs(TABLES)}
    claim = row_to_document(
        "crm", "sql", specs["claims"],
        {"claim_id": 7, "description": "  Schade aan de wagen.  ", "status": "open",
         "incident_date": date(2025, 3, 1)},
    )
    note = row_to_document(
        "crm", "sql", specs["claim_notes"],
        {"note_id": 99, "claim_id": 7, "body": "Expert aangesteld.", "author_role": "agent"},
    )
    assert claim.id == "crm:claims:7" and claim.source_uri == "claims:7"
    assert claim.content == "Schade aan de wagen."
    assert claim.metadata["thread_id"] == note.metadata["thread_id"] == "claim:7"
    assert claim.metadata["incident_date"] == "2025-03-01"


def test_multiple_text_columns_are_joined_and_values_serialised() -> None:
    spec = parse_table_specs({"t": {"id": "id", "text": ["a", "b"], "meta": ["amount"]}})[0]
    doc = row_to_document("s", "sql", spec, {"id": 1, "a": "first", "b": None, "amount": Decimal("12.50")})
    assert doc.content == "first"
    assert doc.metadata["amount"] == "12.50"
    assert doc.metadata["thread_id"] == "t:1"


LIVE_DSN = os.environ.get("R3AL_TEST_POSTGRES_DSN")


@pytest.mark.skipif(not LIVE_DSN, reason="set R3AL_TEST_POSTGRES_DSN to run against a seeded DB")
def test_live_database() -> None:
    connector = SqlConnector({"dsn": LIVE_DSN, "tables": TABLES})
    connector.name = "crm_postgres"
    assert connector.test_connection()
    refs = list(connector.list_items())
    claims, notes = len([r for r in refs if r.item_id.startswith("claims:")]), len([r for r in refs if r.item_id.startswith("claim_notes:")])
    assert claims == 85 and notes > claims  # default seed: 85 claims, several notes each
    doc = connector.fetch(refs[0])
    assert doc.id == "crm_postgres:claims:1"
    assert doc.content and doc.metadata["thread_id"] == "claim:1"
    # Fallback path: fetch a row that is no longer cached.
    again = connector.fetch(refs[0])
    assert again.content == doc.content


def test_unreachable_database_fails_connection_test() -> None:
    connector = SqlConnector({"dsn": "postgresql://nobody:x@127.0.0.1:1/none", "tables": TABLES})
    assert connector.test_connection() is False


CASES: dict[str, Any] = {
    "claim": {"table": "claims", "id": "claim_id", "group_prefix": "claim",
              "messages": [{"column": "description", "role": "customer"}],
              "fields": ["claim_number", "claim_type", "status", "incident_date"],
              "notes": {"table": "claim_notes", "key": "claim_id", "id": "note_id", "text": "body",
                        "role": "author_role", "order_by": "created_at"}},
}


def test_case_bundle_keeps_texts_aligned_with_roles_and_typed_fields() -> None:
    from src.connectors.sql import case_to_document, parse_case_specs
    from src.processing.clean import split_messages

    (spec,) = parse_case_specs(CASES)
    doc = case_to_document(
        "crm", "sql", spec,
        {"claim_id": 7, "description": "Waterschade in de keuken.", "claim_number": "CLM-1", "claim_type":
         "water_damage", "status": "in_progress", "incident_date": date(2025, 3, 12)},
        [{"claim_id": 7, "note_id": 1, "body": "Expert aangesteld.", "author_role": "internal",
          "created_at": date(2025, 3, 13)},
         {"claim_id": 7, "note_id": 2, "body": "Wij sturen een expert.", "author_role": "agent",
          "created_at": date(2025, 3, 14)}],
    )
    assert doc.id == "crm:claims:7" and doc.source_uri == "claims:7"
    assert doc.metadata["kind"] == "case" and doc.metadata["thread_id"] == "claim:7"
    assert [n["author_role"] for n in doc.metadata["notes"]] == ["customer", "internal", "agent"]
    assert split_messages(str(doc.content))[2] == "Wij sturen een expert."
    assert doc.metadata["fields"]["incident_date"] == "2025-03-12"


@pytest.mark.skipif(not LIVE_DSN, reason="set R3AL_TEST_POSTGRES_DSN to run against a seeded DB")
def test_live_case_bundles() -> None:
    connector = SqlConnector({"dsn": LIVE_DSN, "cases": CASES})
    connector.name = "crm_postgres"
    assert connector.test_connection()
    docs = [connector.fetch(r) for r in connector.list_items()]
    assert len(docs) == 85
    roles = {n["author_role"] for d in docs for n in d.metadata["notes"]}
    assert roles == {"customer", "agent", "internal"}
