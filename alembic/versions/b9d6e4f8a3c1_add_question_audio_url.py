"""add question_audio_url column to past_questions

Revision ID: b9d6e4f8a3c1
Revises: a8c5d3e7f1b2
Create Date: 2026-04-30

"""
from alembic import op
import sqlalchemy as sa

revision = 'b9d6e4f8a3c1'
down_revision = 'a8c5d3e7f1b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('past_questions', sa.Column('question_audio_url', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('past_questions', 'question_audio_url')
