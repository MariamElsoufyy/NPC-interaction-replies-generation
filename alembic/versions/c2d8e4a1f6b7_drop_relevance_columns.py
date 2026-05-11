"""drop relevance_pass and relevance_score from past_questions

The cosine-similarity verification layer was removed; these columns were
populated only by that layer and are no longer written to.

Revision ID: c2d8e4a1f6b7
Revises: b1c5f7a3d9e2
Create Date: 2026-05-10

"""
from alembic import op
import sqlalchemy as sa

revision = 'c2d8e4a1f6b7'
down_revision = 'b1c5f7a3d9e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column('past_questions', 'relevance_score')
    op.drop_column('past_questions', 'relevance_pass')


def downgrade() -> None:
    op.add_column('past_questions', sa.Column('relevance_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('relevance_score', sa.Float(), nullable=True))
