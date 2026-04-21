import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small dimensions


class FAQ(Base):
    __tablename__ = "frequently_asked_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    character_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # S1 / S2 / P1
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)   # Supabase Storage URL
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")  # 'en' or 'ar'
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    tag = mapped_column(String(50), nullable=True)  # optional tag for categorization
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<FAQ id={self.id} character={self.character_id} question={self.question[:40]!r}>"
