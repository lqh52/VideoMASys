-- Reference schema for environments that do not use AUTO_CREATE_SCHEMA.
-- The prototype CLI applies the equivalent SQLAlchemy metadata plus these indexes.
CREATE EXTENSION IF NOT EXISTS vector;

-- Run `videomasys-init-db` for the complete initial schema. Embeddings are vector(1024).
-- Future production migrations should be managed with Alembic before schema evolution.

