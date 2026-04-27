"""add verifier result and content filter result columns to past_questions

Revision ID: f5b3e9d2a7c1
Revises: e4a7c2d8f1b9
Create Date: 2026-04-26

"""
from alembic import op
import sqlalchemy as sa

revision = 'f5b3e9d2a7c1'
down_revision = 'e4a7c2d8f1b9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('past_questions', sa.Column('content_filter_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('content_filter_flagged', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_historical_accuracy', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_appropriateness', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_modern_references', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_in_character', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('past_questions', 'verifier_in_character')
    op.drop_column('past_questions', 'verifier_modern_references')
    op.drop_column('past_questions', 'verifier_appropriateness')
    op.drop_column('past_questions', 'verifier_historical_accuracy')
    op.drop_column('past_questions', 'verifier_pass')
    op.drop_column('past_questions', 'content_filter_flagged')
    op.drop_column('past_questions', 'content_filter_pass')
