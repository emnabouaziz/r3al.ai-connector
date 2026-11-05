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
Outputs are first written to `out/.staging/`. They are promoted to `out/` only after both PII scans pass; on failure they stay in `out/.staging/` for inspection, so a false positive never destroys a completed run.

Exit codes: `0` success, `1` any other failure, `2` train/eval leakage detected, `3` PII found by the pattern scan, `4` PII found by the ground-truth scan (see below).

## Local database (synthetic data)

PostgreSQL runs in Docker (`app-postgres`, port 5432). To create the CRM tables and load invented data:

```
py -3.11 scripts/seed_crm.py --reset
```

This drops and recreates the `customers`, `policies`, `claims`, `claim_notes` and `invoices` tables.

## Documents (synthetic folder)

```
py -3.11 scripts/generate_documents.py
```

This writes 14 PDF / DOCX files into `data/generated/documents/` (git-ignored). The seed, the folder (`${DOCS_DIR}`) and the PII manifest path come from `config.production.yaml`, and the output is byte-reproducible. The folder is deliberately messy: long policy conditions with numbered sections, a FR v2 and NL v3 version of the same procedure with contradictory deadlines, a tariff table, a scanned PDF without a text layer, a DOCX with mojibake and mid-sentence line breaks, and an almost empty template.

**Parsing** dispatches on `mime_type` through a single table (`src/processing/parsers/`). The pipeline never branches on the source:
- **PDF** is read with pypdf. A PDF without a text layer yields `""` and is dropped with reason `pdf_no_text_layer`.
- **DOCX** is read with python-docx. Heading styles become section markers, and tables are serialised row by row as `cell | cell | cell`.
- **For both formats**, encoding is repaired with ftfy and lines broken mid-sentence are rejoined.

**Chunking.** A 30-page policy document is not one training example. Texts longer than `parse.max_chunk_chars` (1200) are split at headings: numbered sections in PDFs, Heading styles in DOCX. A section that is still too long is split by grouping paragraphs, and a lone heading is merged into the next section. Short texts (CRM rows, e-mails) stay one unchanged chunk. `group_id` is the document, not the chunk, so all chunks of a document land on the same side of the train/eval split.

**Supersede.** If two versions of the same procedure give contradictory answers (a 5-day vs an 8-day deadline), training on both teaches the model to answer either way at random. That is worse than losing the older version. So a file `<base>_v<N>` is marked `superseded` when a sibling `<base>_v<M>` (M > N) exists, is not older, and has a similar title (≥ 0.8). The pipeline drops it with reason `superseded` and counts it in `Counters.superseded`. Detection is limited to the same base filename, i.e. the same language. The FR v2 / NL v3 pair is therefore logged as unmatched rather than guessed; see [FAKES.md](FAKES.md).

**Next step: OCR.** Scanned PDFs are dropped today. An OCR pass (e.g. Tesseract with `nld`, `fra` and `eng`) would recover them, and its output would then need the same anonymisation and recall checks as any other text.

## Pipeline

```
ingest -> parse -> clean (+ chunk) -> dedup_exact -> anonymize -> dedup_near -> tag -> split -> build -> verify -> report
```

- One failing item is dropped and counted; it never fails the run.
- Leakage between train and eval, an invalid config or a broken counter invariant fail the run.
- The pipeline never branches on the source: parsers are chosen by `mime_type`, and cleaning, chunking and deduplication apply to every record.
- Items the pipeline refuses to decide (e.g. two versions of a procedure it cannot match) are listed under **"Needs client review"** in the quality report.
- Every shortcut still in the code is listed in [FAKES.md](FAKES.md).

## Support tickets (synthetic export)

```
py -3.11 scripts/generate_tickets.py
```

This writes `data/generated/tickets/tickets_export.csv` (59 threads, ~190 messages) and `data/generated/tickets/emails/*.eml`. The .eml files re-export 15 of those threads from the mailbox, with other ids, quoted history, other signatures and HTML parts. The output is byte-reproducible.

**Email connector** (`src/connectors/email.py`): one document per **thread**, not per message, built from both the CSV and the .eml folder. Messages are concatenated in chronological order and each message's direction is kept in metadata.
- **The `group_id` is derived from content**, not from the export. It combines the normalised subject (`Re:`/`Fwd:`/`Antw:` stripped) with a hash of the opening of the first customer message, after cleaning. The same conversation arriving through the CSV and through the mailbox therefore gets the same group and can never be split between train and eval. Only hashes enter the id, never text.

**Cleaning** (`src/processing/clean.py`) runs per message. Each operation is counted in the report:
- quoted history (`>` lines and `On … wrote:` / `Op … schreef:` / `Le … a écrit :` lines);
- the legal disclaimer;
- signature blocks (a `--` line or a sign-off line followed by fewer than 6 lines);
- automatic replies: a thread containing only auto-replies is dropped as `auto_acknowledgement`.

A text that becomes too short after cleaning is dropped as `too_short_after_clean`. Chunking happens after cleaning, so boilerplate never decides where a document is cut.

**Near-duplicates** (`src/processing/dedup_near.py`) are detected with MinHash + LSH and confirmed with exact Jaccard (≥ 0.85 on character 5-grams).
- **It runs after anonymisation.** Two messages that differ only by a name or an IBAN are the same example once masked. The counter-argument, written in the module docstring: two genuinely different claims written from the same template may merge. We accept that, because they teach the model the same thing.
- **Effect on the dataset:** the groups of a cluster are merged and one representative is kept (the longest text). The other members are dropped as `near_duplicate` and counted in `duplicates_near`.
- **On the synthetic data:** exactly the 15 CSV/.eml pairs are clustered.

## Training examples (build stage)

Order: `... -> tag -> build -> split -> verify`. Examples are built **before** the split and carry the `group_id` of their thread / document / case, so every pair from one source object lands on the same side. Each output line keeps the client schema, and `meta.source_ids` holds the real ids of the source documents (e.g. `support_mailbox:csv:THR-0042`, `crm_postgres:claims:59`, `knowledge_docs:faq_customers_nl.docx`).

The three sources are three kinds of material, so each has its own builder. The builder is chosen by the record's **kind** (carried in metadata), never by its source:

| kind | material | builder | what it produces |
|---|---|---|---|
| `thread` | conversations (tickets) | tickets | the last customer message before an agent reply + that reply |
| `case` | structured facts (CRM) | facts | extracted customer -> agent pairs, plus a templated factual Q/A grounded in the fields |
| `chunk` | reference knowledge (documents) | documents | one question per section, answered by that section's own text |

- **Tickets**. Rejected and counted: `no_agent_reply`, `auto_acknowledgement`, `refers_to_attachment` ("zie bijlage", "ci-joint", "attached"), `answer_is_a_question`, `answer_too_short`, `question_too_short`, `placeholder_heavy`.
- **CRM cases.** The SQL connector assembles one *case bundle* per claim / policy / invoice:
  - its texts, in time order, each with an `author_role` (`customer`, `agent`, `internal`);
  - its typed `fields` (status, claim / product type, dates, amounts, references).

  The texts travel in the content, so they are cleaned and anonymised like every other text, and field values go through the anonymiser too. From each bundle the facts builder makes:
  - **B1**: a customer note followed by a note written *to* the customer becomes a pair, with the ticket rules. An **internal note is never an answer**.
  - **B2**: one templated question/answer per case, keyed by (case_type, status, language), e.g. "What is the status of my claim CLM-2024-00845 for collision?". For an open claim, the answer adds a customer-facing next step chosen from the *category* of the latest internal note, never its text.

  Rejected and counted: `no_template_for_state` (no catch-all: a rejected claim has no template on purpose), `fields_insufficient`, `answer_not_grounded` (every date, amount and reference in the answer must be a field value), `placeholder_heavy`.
- **Documents**: the question is anchored on the heading, FAQ question or title, with the document's title as context. **Every answer is contained in the chunk.** Rejected and counted: `no_question_anchor`, `chunk_too_short`, `answer_not_grounded`, `placeholder_heavy`.
- **Example-level dedup**: identical template replies across threads would otherwise put the same pair in both train and eval (`duplicate_example`).

**Result on the synthetic sources: 675 candidates → 642 accepted → 599 examples (422 train / 177 eval), every domain MEASURABLE with ≥ 25 eval examples.** The report shows the rejections **per builder**, with an acceptance rate for each, and a builder stuck at 0% is flagged at the top. That is how the previous defect (the CRM yielding nothing) was found. The high rates on synthetic data are explained in [FAKES.md](FAKES.md): the rules did not change, the CRM's denominator did.

`placeholder_heavy` is measured on the answer, with a threshold of 0.12. On the synthetic data, real answers stay ≤ 0.09 (a greeting); form examples reach 0.13–0.17.

**System prompt**: "You are the support assistant of `<build.insurer_name>`…", written **in the example's language** (`build.system_prompt_language: example_language`), so the model learns to answer in the customer's language. Set it to `english` for a single English prompt. The insurer's name is never hardcoded.

**No LLM.** Questions come from local templates (`src/dataset/question_gen.py`). Client content must never reach an external LLM, the same rule the anonymiser enforces. `QuestionGenerator` is the extension point: an approved local model would plug in there without touching the pipeline, and its answers would still have to pass the grounding check.

## Known limitations

- **Cross-source semantic duplicates.** 3 ticket threads discuss claims that also exist in the CRM `claim_notes`: same customer, same claim number, different wording. Near-duplicate detection is lexical, so it does not see them, and both versions stay in the dataset. Resolving this needs entity linking (claim number + customer) or embeddings. It is deliberately not attempted here.
- **Cross-language document versions** (FR v2 / NL v3 of the same procedure) are escalated for review, not matched (see [FAKES.md](FAKES.md)).

## Anonymisation

Every record is anonymised before tagging, splitting and writing. PII is replaced by structural placeholders, in English even inside Dutch or French text: `[PERSON] [EMAIL] [PHONE] [IBAN] [NATIONAL_ID] [ADDRESS]`.

1. **Structured PII, regex + validator** (`src/processing/anonymize/`). Patterns are deliberately wide; a match is replaced only if its validator confirms it: IBAN mod 97, Belgian national register number checksum (pre-2000 and post-2000 branches) plus a plausible birth date, Belgian phone numbering plan, e-mail structure, Belgian postcode range for addresses. Rejected candidates are counted per pattern in the quality report, which makes each pattern's precision visible.
2. **Names, NER**: runs last, on the already-masked text, and never inside an existing placeholder.
3. **Pattern scan** (`src/report/pii_scan.py`): `run.py` re-scans the staged `train.jsonl` / `eval.jsonl` with every validated pattern. Any confirmed hit stops the run (exit code `3`). Findings carry file, line and PII type, never the value.
4. **Ground-truth recall scan** (`src/report/ground_truth_scan.py`). The pattern scan reuses the anonymiser's own patterns, so it cannot see PII those patterns do not cover: on its own, its "0 findings" proves nothing about recall. `scripts/seed_crm.py` therefore writes `data/generated/pii_manifest.json` (git-ignored: it is a PII index). It holds every PII value injected into free text, with the texts it appears in. The scan normalises each output line (casefold, letters and digits only) and searches each value as a literal substring, with **no regex**. It is independent of formatting: an IBAN without spaces, a value broken across a line break, or a name cut between the user and assistant turns are all found. Recall is measured only over values that entered the pipeline. Any hit stops the run (exit code `4`), and misses are reported as a sha256 prefix only. Without a manifest (real client data) the scan is skipped and the report says so explicitly.

   On the synthetic CRM: **424 injected values, 0 leaked, 100% recall for every type**. A negative control with the anonymiser switched off gives 13% recall and blocks the run, which shows the scan does detect leaks. Known limit: PERSON ground truth is full names, so a surname left alone after a partially masked name would not count as a miss.

**NER model choice: spaCy `xx_ent_wiki_sm`, not Presidio.**
- One small multilingual model (Dutch, French, English) instead of three per-language models.
- Installed from a pinned wheel (see `pyproject.toml`), loaded offline, deterministic, loaded once per process.
- Presidio would have added an orchestration layer on top of per-language spaCy models (large English models by default) with no gain for this corpus. Its value lies in recognisers that layer 1 already covers with stricter checksum validation.

**Trade-off: recall over precision.** Masking an innocent word costs a little training signal, while a leaked name in a fine-tuning set cannot be taken back. Every `PER` entity is masked, and cue rules (honorifics, letter sign-offs, contact phrases, record headers) catch names the model misses. The only precision guard is edge trimming: words that cannot be names (stop words, salutations, taxonomy terms) are trimmed from the edges of a `PER` span. On the synthetic CRM (615 texts, real names known) this gives **0 leaked names**, and over-masked spans fall from 222 to 35.

## Tests

```
py -3.11 -m pytest -q
py -3.11 -m mypy src
```
