"""create past_questions table

Revision ID: c8d2f1a4b037
Revises: a1c3e7f92d05
Create Date: 2026-04-23

Stores every completed pipeline interaction — question, answer, source (faq/llm),
audio URL, and full timing breakdown. Written as a background fire-and-forget task
so it never blocks the pipeline.
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID


revision: str = 'c8d2f1a4b037'
down_revision: Union[str, Sequence[str], None] = 'a1c3e7f92d05'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'past_questions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('character_id', sa.String(10), nullable=False),
        sa.Column('question', sa.Text, nullable=False),
        sa.Column('answer', sa.Text, nullable=True),
        sa.Column('audio_url', sa.Text, nullable=True),
        sa.Column('source', sa.String(10), nullable=False, server_default='llm'),
        sa.Column('faq_hit', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('preprocess_s', sa.Float, nullable=True),
        sa.Column('stt_s', sa.Float, nullable=True),
        sa.Column('faq_lookup_s', sa.Float, nullable=True),
        sa.Column('llm_s', sa.Float, nullable=True),
        sa.Column('tts_first_chunk_s', sa.Float, nullable=True),
        sa.Column('tts_total_s', sa.Float, nullable=True),
        sa.Column('time_to_first_audio_s', sa.Float, nullable=True),
        sa.Column('total_s', sa.Float, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    )
    op.create_index('ix_past_questions_character_id', 'past_questions', ['character_id'])
    op.create_index('ix_past_questions_created_at', 'past_questions', ['created_at'])


def downgrade() -> None:
    op.drop_index('ix_past_questions_created_at', table_name='past_questions')
    op.drop_index('ix_past_questions_character_id', table_name='past_questions')
    op.drop_table('past_questions')
