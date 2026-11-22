-- One row per ingested source (a file, for now), per tenant.
CREATE TABLE sources (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id   text NOT NULL CHECK (tenant_id <> ''),
    source_key  text NOT NULL,          -- path relative to the tenant's folder
    title       text NOT NULL,
    fingerprint text NOT NULL,          -- see ingest/fingerprint.py
    ingested_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, source_key),
    -- Target of the composite foreign key below.
    UNIQUE (id, tenant_id)
);

-- The searchable pieces. Every chunk carries its tenant.
CREATE TABLE chunks (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id       bigint NOT NULL,
    tenant_id       text NOT NULL CHECK (tenant_id <> ''),
    ordinal         integer NOT NULL,   -- position within the source
    text            text NOT NULL,
    language        text NOT NULL CHECK (language IN ('ar', 'en')),
    heading_path    text[] NOT NULL,
    embedding       vector(1024) NOT NULL,
    embedding_model text NOT NULL,
    UNIQUE (source_id, ordinal),
    -- The foreign key includes tenant_id, so the database itself refuses a
    -- chunk whose tenant differs from its source's tenant. Deleting a source
    -- deletes its chunks.
    FOREIGN KEY (source_id, tenant_id)
        REFERENCES sources (id, tenant_id) ON DELETE CASCADE
);

-- Every read filters by tenant first.
CREATE INDEX chunks_tenant_id_idx ON chunks (tenant_id);
