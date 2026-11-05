"""Create the synthetic CRM schema and load deterministic fake data into Postgres.

    python scripts/seed_crm.py --reset

DESTRUCTIVE: drops and recreates the 5 CRM tables in the target database.
Refuses to run without --reset if any of those tables already exists.
The DSN comes from --dsn or the POSTGRES_DSN environment variable / .env.
"""

import argparse
import logging
import os
import sys
from dataclasses import astuple
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from src.synthetic.crm import CrmDataset, generate_crm, write_pii_manifest

logger = logging.getLogger("seed_crm")

TABLES = ("customers", "policies", "claims", "claim_notes", "invoices")


def _insert(cur: psycopg.Cursor[tuple[object, ...]], table: str, rows: list[object]) -> None:
    if not rows:
        return
    width = len(astuple(rows[0]))  # type: ignore[call-overload]
    placeholders = ", ".join(["%s"] * width)
    query = f"INSERT INTO {table} VALUES ({placeholders})"  # table names are constants
    cur.executemany(query, [astuple(r) for r in rows])  # type: ignore[call-overload]


def seed(dsn: str, schema_sql: str, data: CrmDataset, reset: bool) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = ANY(%s)",
            (list(TABLES),),
        )
        row = cur.fetchone()
        existing = int(row[0]) if row else 0
        if existing and not reset:
            raise SystemExit(f"{existing} CRM table(s) already exist; rerun with --reset to replace them")
        cur.execute(schema_sql.encode("utf-8"))
        _insert(cur, "customers", list(data.customers))
        _insert(cur, "policies", list(data.policies))
        _insert(cur, "claims", list(data.claims))
        _insert(cur, "claim_notes", list(data.claim_notes))
        _insert(cur, "invoices", list(data.invoices))
        conn.commit()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=None, help="defaults to $POSTGRES_DSN (.env is loaded)")
    parser.add_argument("--schema", default="db/schema.sql")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true", help="drop and recreate the CRM tables")
    parser.add_argument(
        "--manifest",
        default="data/generated/pii_manifest.json",
        help="ground-truth PII index of the injected values (git-ignored)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")

    load_dotenv(override=False)
    dsn = args.dsn or os.environ.get("POSTGRES_DSN")
    if not dsn:
        logger.error("seed_failed reason=missing_dsn")
        return 1

    data = generate_crm(seed=args.seed)
    seed(dsn, Path(args.schema).read_text(encoding="utf-8"), data, args.reset)
    n_values = write_pii_manifest(data, args.seed, args.manifest)
    logger.info("manifest_written path=%s values=%d", args.manifest, n_values)
    logger.info(
        "seed_ok customers=%d policies=%d claims=%d claim_notes=%d invoices=%d texts=%d",
        len(data.customers), len(data.policies), len(data.claims),
        len(data.claim_notes), len(data.invoices), len(data.text_origin),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
