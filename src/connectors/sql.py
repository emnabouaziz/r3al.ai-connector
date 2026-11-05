"""Generic read-only PostgreSQL connector.

One row with non-empty text = one RawDocument. Which columns hold text,
which column groups rows into a thread, and which columns are kept as
metadata all come from config, e.g.:

    tables:
      claim_notes:
        id: note_id                # primary key
        text: [body]               # concatenated with blank lines
        group_key: claim_id        # rows sharing it form one group
        group_prefix: claim        # -> thread_id "claim:42"
        meta: [author_type, created_at]

Rows of different tables with the same group_prefix + group_key value
(e.g. claims.claim_id and claim_notes.claim_id) end up in the same group,
so a claim and its notes are never split between train and eval.
"""

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from src.connectors.base import SourceConnector
from src.contracts import ItemRef, RawDocument
from src.processing.clean import MESSAGE_SEPARATOR
from src.registry import register

logger = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


class SqlConfigError(ValueError):
    """Invalid sql connector configuration."""


@dataclass(frozen=True)
class TableSpec:
    table: str
    id_column: str
    text_columns: tuple[str, ...]
    group_key: str
    group_prefix: str
    meta_columns: tuple[str, ...] = ()

    @property
    def columns(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for col in (self.id_column, self.group_key, *self.text_columns, *self.meta_columns):
            seen.setdefault(col, None)
        return tuple(seen)


def _ident(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise SqlConfigError(f"invalid SQL identifier for {where}")
    return value


def parse_table_specs(raw: Any) -> list[TableSpec]:
    if not isinstance(raw, dict) or not raw:
        raise SqlConfigError("'tables' must be a non-empty mapping of table -> spec")
    specs: list[TableSpec] = []
    for table, spec in raw.items():
        t = _ident(table, "table name")
        if not isinstance(spec, dict):
            raise SqlConfigError(f"spec for table '{t}' must be a mapping")
        text = spec.get("text")
        if not isinstance(text, list) or not text:
            raise SqlConfigError(f"table '{t}': 'text' must be a non-empty list of columns")
        id_col = _ident(spec.get("id"), f"{t}.id")
        specs.append(
            TableSpec(
                table=t,
                id_column=id_col,
                text_columns=tuple(_ident(c, f"{t}.text") for c in text),
                group_key=_ident(spec.get("group_key", id_col), f"{t}.group_key"),
                group_prefix=_ident(spec.get("group_prefix", t), f"{t}.group_prefix"),
                meta_columns=tuple(_ident(c, f"{t}.meta") for c in spec.get("meta", [])),
            )
        )
    return specs


def build_select(spec: TableSpec, single_row: bool = False) -> sql.Composed:
    """SELECT the spec's columns for rows having at least one non-empty text column."""
    non_empty = sql.SQL(" OR ").join(
        sql.SQL("coalesce(btrim({}::text), '') <> ''").format(sql.Identifier(c))
        for c in spec.text_columns
    )
    query = sql.SQL("SELECT {cols} FROM {table} WHERE ({non_empty})").format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in spec.columns),
        table=sql.Identifier(spec.table),
        non_empty=non_empty,
    )
    if single_row:
        return query + sql.SQL(" AND {} = %s").format(sql.Identifier(spec.id_column))
    return query + sql.SQL(" ORDER BY {}").format(sql.Identifier(spec.id_column))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def row_to_document(source_name: str, source_type: str, spec: TableSpec, row: dict[str, Any]) -> RawDocument:
    pk = row[spec.id_column]
    texts = [str(row[c]).strip() for c in spec.text_columns if row.get(c) is not None]
    item_id = f"{spec.table}:{pk}"
    return RawDocument(
        id=f"{source_name}:{item_id}",
        source_type=source_type,
        source_uri=item_id,  # "table:pk"
        content="\n\n".join(t for t in texts if t),
        mime_type="text/plain",
        metadata={
            "table": spec.table,
            "pk": _jsonable(pk),
            "thread_id": f"{spec.group_prefix}:{row[spec.group_key]}",
            **{c: _jsonable(row.get(c)) for c in spec.meta_columns},
        },
    )


# --- Case bundles ---------------------------------------------------------------
#
# A CASE is one business object with its structured facts and its texts:
#   cases:
#     claim:
#       table: claims
#       id: claim_id
#       group_prefix: claim
#       messages: [{column: description, role: customer}]
#       fields: [claim_number, claim_type, status, incident_date]
#       notes: {table: claim_notes, key: claim_id, id: note_id, text: body,
#               role: author_role, order_by: created_at}
# One RawDocument per case. Texts (main columns, then notes in time order) go
# in the CONTENT, joined by MESSAGE_SEPARATOR, so they are cleaned and
# anonymised like every other text; metadata["notes"] keeps, at the same
# index, each text's author_role and created_at. metadata["fields"] holds the
# typed columns (status, types, dates, amounts) the facts builder grounds on.


@dataclass(frozen=True)
class NotesSpec:
    table: str
    key: str  # foreign key to the case id
    id_column: str
    text: str
    role: str  # column holding author_role
    order_by: str


@dataclass(frozen=True)
class CaseSpec:
    case_type: str
    table: str
    id_column: str
    group_prefix: str
    messages: tuple[tuple[str, str], ...]  # (column, author_role)
    fields: tuple[str, ...]
    notes: NotesSpec | None = None


def parse_case_specs(raw: Any) -> list[CaseSpec]:
    if raw is None:
        return []
    if not isinstance(raw, dict) or not raw:
        raise SqlConfigError("'cases' must be a non-empty mapping of case_type -> spec")
    specs: list[CaseSpec] = []
    for case_type, spec in raw.items():
        ct = _ident(case_type, "case type")
        if not isinstance(spec, dict):
            raise SqlConfigError(f"case '{ct}' must be a mapping")
        messages = spec.get("messages")
        if not isinstance(messages, list) or not messages:
            raise SqlConfigError(f"case '{ct}': 'messages' must list {{column, role}} entries")
        notes = spec.get("notes")
        notes_spec = None
        if notes is not None:
            if not isinstance(notes, dict):
                raise SqlConfigError(f"case '{ct}': 'notes' must be a mapping")
            notes_spec = NotesSpec(
                table=_ident(notes.get("table"), f"{ct}.notes.table"),
                key=_ident(notes.get("key"), f"{ct}.notes.key"),
                id_column=_ident(notes.get("id"), f"{ct}.notes.id"),
                text=_ident(notes.get("text"), f"{ct}.notes.text"),
                role=_ident(notes.get("role"), f"{ct}.notes.role"),
                order_by=_ident(notes.get("order_by"), f"{ct}.notes.order_by"),
            )
        specs.append(
            CaseSpec(
                case_type=ct,
                table=_ident(spec.get("table"), f"{ct}.table"),
                id_column=_ident(spec.get("id"), f"{ct}.id"),
                group_prefix=_ident(spec.get("group_prefix", ct), f"{ct}.group_prefix"),
                messages=tuple(
                    (_ident(m.get("column"), f"{ct}.messages.column"), _ident(m.get("role"), f"{ct}.messages.role"))
                    for m in messages
                ),
                fields=tuple(_ident(c, f"{ct}.fields") for c in spec.get("fields", [])),
                notes=notes_spec,
            )
        )
    return specs


def build_case_select(spec: CaseSpec, single_row: bool = False) -> sql.Composed:
    """Main rows having at least one non-empty message column."""
    columns = dict.fromkeys([spec.id_column, *(c for c, _ in spec.messages), *spec.fields])
    non_empty = sql.SQL(" OR ").join(
        sql.SQL("coalesce(btrim({}::text), '') <> ''").format(sql.Identifier(c)) for c, _ in spec.messages
    )
    query = sql.SQL("SELECT {cols} FROM {table} WHERE ({non_empty})").format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        table=sql.Identifier(spec.table),
        non_empty=non_empty,
    )
    if single_row:
        return query + sql.SQL(" AND {} = %s").format(sql.Identifier(spec.id_column))
    return query + sql.SQL(" ORDER BY {}").format(sql.Identifier(spec.id_column))


def build_notes_select(notes: NotesSpec, single_case: bool = False) -> sql.Composed:
    query = sql.SQL("SELECT {key}, {id}, {text}, {role}, {order} FROM {table}").format(
        key=sql.Identifier(notes.key), id=sql.Identifier(notes.id_column), text=sql.Identifier(notes.text),
        role=sql.Identifier(notes.role), order=sql.Identifier(notes.order_by), table=sql.Identifier(notes.table),
    )
    if single_case:
        query += sql.SQL(" WHERE {} = %s").format(sql.Identifier(notes.key))
    return query + sql.SQL(" ORDER BY {}, {}, {}").format(
        sql.Identifier(notes.key), sql.Identifier(notes.order_by), sql.Identifier(notes.id_column)
    )


def case_to_document(
    source_name: str, source_type: str, spec: CaseSpec, row: dict[str, Any], notes: list[dict[str, Any]]
) -> RawDocument:
    """One case -> one RawDocument (pure: no database access)."""
    pk = row[spec.id_column]
    item_id = f"{spec.table}:{pk}"
    texts: list[str] = []
    parts: list[dict[str, Any]] = []
    for column, role in spec.messages:
        value = row.get(column)
        if value is not None and str(value).strip():
            texts.append(str(value).strip())
            parts.append({"author_role": role, "created_at": None})
    if spec.notes is not None:
        for note in notes:
            text = note.get(spec.notes.text)
            if text is not None and str(text).strip():
                texts.append(str(text).strip())
                parts.append({"author_role": str(note.get(spec.notes.role)),
                              "created_at": _jsonable(note.get(spec.notes.order_by))})
    return RawDocument(
        id=f"{source_name}:{item_id}",
        source_type=source_type,
        source_uri=item_id,  # "table:pk"
        content=MESSAGE_SEPARATOR.join(texts),
        mime_type="text/plain",
        metadata={
            "kind": "case",  # builder dispatch: structured facts -> grounded Q/A
            "case_type": spec.case_type,
            "thread_id": f"{spec.group_prefix}:{pk}",
            "fields": {c: _jsonable(row.get(c)) for c in spec.fields},
            "notes": parts,  # aligned with the content's messages
        },
    )


@register
class SqlConnector(SourceConnector):
    """Read-only PostgreSQL source. Never writes to the database."""

    source_type = "sql"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        dsn = config.get("dsn")
        if not isinstance(dsn, str) or not dsn:
            raise SqlConfigError("'dsn' is required")
        self._dsn = dsn
        self.specs = parse_table_specs(config["tables"]) if config.get("tables") is not None else []
        self.case_specs = parse_case_specs(config.get("cases"))
        if not self.specs and not self.case_specs:
            raise SqlConfigError("configure 'tables' (one document per row) and/or 'cases' (case bundles)")
        self._specs_by_table = {s.table: s for s in self.specs}
        self._cases_by_table = {c.table: c for c in self.case_specs}
        self._cache: dict[str, RawDocument] = {}

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        conn = psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=10)
        conn.read_only = True
        return conn

    def test_connection(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                tables = [s.table for s in self.specs] + [c.table for c in self.case_specs]
                tables += [c.notes.table for c in self.case_specs if c.notes is not None]
                for table in tables:
                    cur.execute("SELECT to_regclass(%s) AS t", (table,))
                    row = cur.fetchone()
                    if row is None or row["t"] is None:
                        logger.error("sql_table_missing source=%s table=%s", self.name, table)
                        return False
            return True
        except psycopg.Error as exc:
            # Never log the message: it can contain the DSN host / user.
            logger.error("sql_connection_failed source=%s error=%s", self.name, type(exc).__name__)
            return False

    def list_items(self) -> Iterator[ItemRef]:
        """Stream every table once; rows are cached so fetch() needs no extra query."""
        self._cache.clear()
        with self._connect() as conn:
            for spec in self.specs:
                count = 0
                with conn.cursor(name=f"list_{spec.table}") as cur:
                    cur.execute(build_select(spec))
                    for row in cur:
                        doc = row_to_document(self.name, self.source_type, spec, row)
                        item_id = doc.source_uri
                        self._cache[item_id] = doc
                        count += 1
                        yield ItemRef(
                            source_type=self.source_type,
                            item_id=item_id,
                            uri=item_id,
                            hints={"thread_id": doc.metadata["thread_id"]},
                        )
                logger.info("sql_table_listed source=%s table=%s rows=%d", self.name, spec.table, count)
            for case in self.case_specs:
                notes_by_case: dict[Any, list[dict[str, Any]]] = {}
                if case.notes is not None:
                    with conn.cursor() as cur:
                        cur.execute(build_notes_select(case.notes))
                        for note in cur.fetchall():
                            notes_by_case.setdefault(note[case.notes.key], []).append(note)
                count = 0
                with conn.cursor(name=f"cases_{case.case_type}") as cur:
                    cur.execute(build_case_select(case))
                    for row in cur:
                        doc = case_to_document(self.name, self.source_type, case, row,
                                               notes_by_case.get(row[case.id_column], []))
                        self._cache[doc.source_uri] = doc
                        count += 1
                        yield ItemRef(source_type=self.source_type, item_id=doc.source_uri, uri=doc.source_uri,
                                      hints={"thread_id": doc.metadata["thread_id"], "case_type": case.case_type})
                logger.info("sql_cases_listed source=%s case_type=%s cases=%d", self.name, case.case_type, count)

    def fetch(self, ref: ItemRef) -> RawDocument:
        cached = self._cache.pop(ref.item_id, None)
        if cached is not None:
            return cached
        table, _, pk = ref.item_id.partition(":")
        case = self._cases_by_table.get(table)
        if case is not None:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(build_case_select(case, single_row=True), (pk,))
                case_row = cur.fetchone()
                notes: list[dict[str, Any]] = []
                if case.notes is not None and case_row is not None:
                    cur.execute(build_notes_select(case.notes, single_case=True), (pk,))
                    notes = list(cur.fetchall())
            if case_row is None:
                raise LookupError(f"case not found: {ref.item_id}")
            return case_to_document(self.name, self.source_type, case, case_row, notes)
        spec = self._specs_by_table[table]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(build_select(spec, single_row=True), (pk,))
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"row not found: {ref.item_id}")
        return row_to_document(self.name, self.source_type, spec, row)
