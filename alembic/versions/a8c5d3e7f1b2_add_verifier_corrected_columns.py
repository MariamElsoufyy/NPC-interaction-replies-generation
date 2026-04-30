"""add verifier corrected_answer and corrected_emotion columns to past_questions

Revision ID: a8c5d3e7f1b2
Revises: f5b3e9d2a7c1
Create Date: 2026-04-30

"""
from alembic import op
import sqlalchemy as sa

revision = 'a8c5d3e7f1b2'
down_revision = 'f5b3e9d2a7c1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('past_questions', sa.Column('verifier_corrected_answer', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_corrected_emotion', sa.String(length=50), nullable=True))


def downgrade() -> None:
    op.drop_column('past_questions', 'verifier_corrected_emotion')
    op.drop_column('past_questions', 'verifier_corrected_answer')
