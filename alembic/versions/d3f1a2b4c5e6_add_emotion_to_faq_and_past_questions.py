"""add emotion column to faq and past_questions

Revision ID: d3f1a2b4c5e6
Revises: c8d2f1a4b037
Create Date: 2026-04-26

"""
from alembic import op
import sqlalchemy as sa

revision = 'd3f1a2b4c5e6'
down_revision = 'c8d2f1a4b037'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('frequently_asked_questions', sa.Column('emotion', sa.String(50), nullable=True))
    op.add_column('past_questions', sa.Column('emotion', sa.String(50), nullable=True))


def downgrade() -> None:
    op.drop_column('frequently_asked_questions', 'emotion')
    op.drop_column('past_questions', 'emotion')
