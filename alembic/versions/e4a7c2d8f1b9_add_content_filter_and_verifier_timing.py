"""add content_filter and verifier columns to past_questions

Revision ID: e4a7c2d8f1b9
Revises: d3f1a2b4c5e6
Create Date: 2026-04-26

"""
from alembic import op
import sqlalchemy as sa

revision = 'e4a7c2d8f1b9'
down_revision = 'd3f1a2b4c5e6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('past_questions', sa.Column('content_filter_s', sa.Float(), nullable=True))
    op.add_column('past_questions', sa.Column('content_filter_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('content_filter_flagged', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('verifier_s', sa.Float(), nullable=True))
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
    op.drop_column('past_questions', 'verifier_s')
    op.drop_column('past_questions', 'content_filter_flagged')
    op.drop_column('past_questions', 'content_filter_pass')
    op.drop_column('past_questions', 'content_filter_s')
