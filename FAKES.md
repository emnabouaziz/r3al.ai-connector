# Assumed shortcuts

One line per shortcut currently in the code, with the reason. Remove a line when the real implementation lands.

- **Builder user/assistant split** ([src/dataset/builder.py](src/dataset/builder.py)): user = first half of the text, assistant = second half, cut at the whitespace closest to the middle. Reason: the fake corpus has no real question/answer pairs; this only exercises the output format.
- **anonymize stub** ([src/processing/pipeline.py](src/processing/pipeline.py)): identity function that sets `pii_counts` to zero for every configured PII type. Reason: real anonymisation is a later task; the counters and report slots already exist.
- **dedup_near stub** ([src/processing/pipeline.py](src/processing/pipeline.py)): identity, `duplicates_near` stays 0. Reason: a later task. The blocking cross-split near-duplicate check in `verify` remains active.
- **Identity parse** ([src/processing/pipeline.py](src/processing/pipeline.py)): one RawDocument -> one Record, group_id = `thread_id` metadata. Reason: the fake connector already emits clean text; real parsers will emit 1..n records.
- **FakeConnector** ([src/connectors/fake.py](src/connectors/fake.py)): 60 hand-written in-memory documents (incl. 3 exact duplicates, 2 near duplicates in the same thread, 1 empty, 1 ten-character, 1 off-topic). Reason: run the pipe end to end before any real source exists. The near duplicates share their original's thread so they cannot leak across the split.
- **Small split floors in config.yaml** (`eval_floor_per_domain: 2`, `eval_floor_per_cell: 1`): production values are 25 / 5 (config.production.yaml). Reason: with 60 documents the production floors would send every group to eval.
- **superseded counter always 0**: no document versioning exists yet.
- **Synthetic CRM data in Postgres** ([src/synthetic/crm.py](src/synthetic/crm.py), loaded by `scripts/seed_crm.py`): invented customers, policies, claims, notes and invoices, texts composed from per-language sentence pools. Reason: no real client data yet. The real schema will differ; only the `tables:` mapping in config.production.yaml should need to change.
- **files / email sources disabled** in config.production.yaml (`enabled: false`): connectors not implemented yet. Hence `compliance` and `onboarding` are reported unmeasurable: the CRM tables do not contain those topics.
