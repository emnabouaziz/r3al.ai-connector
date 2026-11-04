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
        self.specs = parse_table_specs(config.get("tables"))
        self._specs_by_table = {s.table: s for s in self.specs}
        self._cache: dict[str, RawDocument] = {}

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        conn = psycopg.connect(self._dsn, row_factory=dict_row, connect_timeout=10)
        conn.read_only = True
        return conn

    def test_connection(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                for spec in self.specs:
                    cur.execute("SELECT to_regclass(%s) AS t", (spec.table,))
                    row = cur.fetchone()
                    if row is None or row["t"] is None:
                        logger.error("sql_table_missing source=%s table=%s", self.name, spec.table)
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

    def fetch(self, ref: ItemRef) -> RawDocument:
        cached = self._cache.pop(ref.item_id, None)
        if cached is not None:
            return cached
        table, _, pk = ref.item_id.partition(":")
        spec = self._specs_by_table[table]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(build_select(spec, single_row=True), (pk,))
            row = cur.fetchone()
        if row is None:
            raise LookupError(f"row not found: {ref.item_id}")
        return row_to_document(self.name, self.source_type, spec, row)
