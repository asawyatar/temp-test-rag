-- Chat history. A conversation belongs to one resident (tenant + user); every
-- read and delete filters on both, so ids from another resident act as unknown.
CREATE TABLE conversations (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id  text NOT NULL CHECK (tenant_id <> ''),
    user_id    text NOT NULL CHECK (user_id <> ''),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    -- Target of the composite foreign key below.
    UNIQUE (id, tenant_id)
);

-- A resident's conversations, newest first.
CREATE INDEX conversations_owner_idx ON conversations (tenant_id, user_id, updated_at DESC);

CREATE TABLE conversation_messages (
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id uuid NOT NULL,
    tenant_id       text NOT NULL,
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    content         text NOT NULL,
    -- User turns: the standalone question actually used for retrieval.
    search_query    text,
    -- Assistant turns: [{"source_key", "title", "cited_text"}, ...]
    citations       jsonb NOT NULL DEFAULT '[]'::jsonb,
    grounded        boolean,
    created_at      timestamptz NOT NULL DEFAULT now(),
    -- Includes tenant_id: the database refuses a message filed under a
    -- different tenant than its conversation. Deleting a conversation
    -- deletes its messages.
    FOREIGN KEY (conversation_id, tenant_id)
        REFERENCES conversations (id, tenant_id) ON DELETE CASCADE
);

CREATE INDEX conversation_messages_order_idx ON conversation_messages (conversation_id, id);
