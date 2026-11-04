# r3al.ai-connector

Pipeline that ingests a Belgian insurer's customer-support data (SQL, files, e-mail; content in Dutch / French / English) and produces `train.jsonl` / `eval.jsonl` for fine-tuning a customer-support model, plus a data-quality report.

## Setup

Python 3.11.

```
py -3.11 -m pip install -e ".[dev]"
cp .env.example .env   # then fill in real values
```

## Run

```
py -3.11 run.py --config config.yaml              # demo, in-memory fake source
py -3.11 run.py --config config.production.yaml   # real sources (see .env)
```

Outputs go to `out/`: `train.jsonl`, `eval.jsonl`, `quality_report.json`, `quality_report.md`.
Exit codes: `0` success, `2` train/eval leakage detected, `1` any other failure.

## Local database (synthetic data)

PostgreSQL runs in Docker (`app-postgres`, port 5432). To create the CRM tables and load invented data:

```
py -3.11 scripts/seed_crm.py --reset
```

This drops and recreates the `customers`, `policies`, `claims`, `claim_notes` and `invoices` tables.

## Pipeline

```
ingest -> parse -> clean -> dedup_exact -> anonymize -> dedup_near -> tag -> split -> build -> verify -> report
```

- One failing item is dropped and counted; it never fails the run.
- Leakage between train and eval, an invalid config or a broken counter invariant fail the run.
- Every shortcut still in the code is listed in [FAKES.md](FAKES.md).

## Tests

```
py -3.11 -m pytest -q
py -3.11 -m mypy src
```
