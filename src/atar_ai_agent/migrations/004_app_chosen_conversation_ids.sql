-- Apps choose their own conversation ids ("1234", a UUID, ...). That id is
-- `external_id`, unique per resident (tenant + user), so two residents using
-- the same id get two different conversations. `id` stays the internal key
-- that messages reference.
ALTER TABLE conversations ADD COLUMN external_id text;
UPDATE conversations SET external_id = id::text;
ALTER TABLE conversations ALTER COLUMN external_id SET NOT NULL;
ALTER TABLE conversations
    ADD CONSTRAINT conversations_external_id_length
        CHECK (char_length(external_id) BETWEEN 1 AND 100),
    ADD CONSTRAINT conversations_owner_external_id
        UNIQUE (tenant_id, user_id, external_id);
