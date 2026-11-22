-- pgvector adds the `vector` column type and distance operators used for
-- similarity search. The tables that use it arrive with the ingestion slice.
CREATE EXTENSION IF NOT EXISTS vector;
