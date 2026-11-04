-- Synthetic CRM schema for local development (database r3alai).
-- Mirrors the tables named in the client spec: customers, policies, claims,
-- claim_notes, invoices. Free-text columns hold multilingual content.

DROP TABLE IF EXISTS invoices, claim_notes, claims, policies, customers CASCADE;

CREATE TABLE customers (
    customer_id      integer PRIMARY KEY,
    first_name       text        NOT NULL,
    last_name        text        NOT NULL,
    email            text        NOT NULL,
    phone            text        NOT NULL,
    national_number  text        NOT NULL,
    iban             text        NOT NULL,
    language         char(2)     NOT NULL CHECK (language IN ('nl', 'fr', 'en')),
    city             text        NOT NULL,
    created_at       date        NOT NULL
);

CREATE TABLE policies (
    policy_id         integer PRIMARY KEY,
    policy_number     text          NOT NULL UNIQUE,
    customer_id       integer       NOT NULL REFERENCES customers (customer_id),
    product           text          NOT NULL CHECK (product IN ('car', 'home', 'family_liability')),
    start_date        date          NOT NULL,
    annual_premium    numeric(10,2) NOT NULL,
    status            text          NOT NULL CHECK (status IN ('active', 'cancelled')),
    coverage_summary  text          -- free text, nullable
);

CREATE TABLE claims (
    claim_id       integer PRIMARY KEY,
    claim_number   text    NOT NULL UNIQUE,
    policy_id      integer NOT NULL REFERENCES policies (policy_id),
    incident_date  date    NOT NULL,
    status         text    NOT NULL CHECK (status IN ('open', 'in_review', 'closed')),
    description    text    NOT NULL
);

CREATE TABLE claim_notes (
    note_id      integer PRIMARY KEY,
    claim_id     integer   NOT NULL REFERENCES claims (claim_id),
    author_type  text      NOT NULL CHECK (author_type IN ('customer', 'agent')),
    created_at   timestamp NOT NULL,
    body         text      NOT NULL
);

CREATE TABLE invoices (
    invoice_id      integer PRIMARY KEY,
    invoice_number  text          NOT NULL UNIQUE,
    policy_id       integer       NOT NULL REFERENCES policies (policy_id),
    issue_date      date          NOT NULL,
    due_date        date          NOT NULL,
    amount          numeric(10,2) NOT NULL,
    status          text          NOT NULL CHECK (status IN ('paid', 'unpaid', 'overdue')),
    customer_note   text          -- free text, nullable
);

CREATE INDEX claim_notes_claim_id_idx ON claim_notes (claim_id);
CREATE INDEX claims_policy_id_idx ON claims (policy_id);
CREATE INDEX invoices_policy_id_idx ON invoices (policy_id);
