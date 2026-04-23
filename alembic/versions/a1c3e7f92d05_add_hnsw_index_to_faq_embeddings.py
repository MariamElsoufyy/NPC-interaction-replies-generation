"""add hnsw index to faq embeddings

Revision ID: a1c3e7f92d05
Revises: b82311e312f3
Create Date: 2026-04-23

HNSW (Hierarchical Navigable Small World) turns the cosine similarity search
from a full sequential scan O(n) into an approximate nearest-neighbor search
O(log n), cutting query time significantly as the FAQ table grows.

Parameters:
  m = 16             — connections per layer; higher = better recall, more memory
  ef_construction=64 — candidate list size during build; higher = better index quality
"""
from typing import Sequence, Union
from alembic import op


revision: str = 'a1c3e7f92d05'
down_revision: Union[str, Sequence[str], None] = '90c4b5bc4d56'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE INDEX IF NOT EXISTS faq_embedding_hnsw_idx
        ON frequently_asked_questions
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS faq_embedding_hnsw_idx")
