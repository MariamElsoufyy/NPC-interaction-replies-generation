import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import PastQuestion


async def create_past_question(db: AsyncSession, data: dict) -> PastQuestion:
    record = PastQuestion(**data)
    db.add(record)
    await db.commit()
    return record


async def get_past_questions_by_character(db: AsyncSession, character_id: str) -> list[PastQuestion]:
    result = await db.execute(
        select(PastQuestion)
        .where(PastQuestion.character_id == character_id)
        .order_by(PastQuestion.created_at.desc())
    )
    return result.scalars().all()
