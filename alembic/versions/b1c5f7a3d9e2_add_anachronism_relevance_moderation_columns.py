"""add anachronism and moderation result columns to past_questions

Revision ID: b1c5f7a3d9e2
Revises: b9d6e4f8a3c1
Create Date: 2026-05-10

"""
from alembic import op
import sqlalchemy as sa

revision = 'b1c5f7a3d9e2'
down_revision = 'b9d6e4f8a3c1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tier 1 — regex anachronism
    op.add_column('past_questions', sa.Column('anachronism_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('anachronism_reasons', sa.Text(), nullable=True))

    # Tier 2 — OpenAI Moderation API (question + answer, separately)
    op.add_column('past_questions', sa.Column('moderation_q_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('moderation_q_categories', sa.Text(), nullable=True))
    op.add_column('past_questions', sa.Column('moderation_a_pass', sa.Boolean(), nullable=True))
    op.add_column('past_questions', sa.Column('moderation_a_categories', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('past_questions', 'moderation_a_categories')
    op.drop_column('past_questions', 'moderation_a_pass')
    op.drop_column('past_questions', 'moderation_q_categories')
    op.drop_column('past_questions', 'moderation_q_pass')
    op.drop_column('past_questions', 'anachronism_reasons')
    op.drop_column('past_questions', 'anachronism_pass')
