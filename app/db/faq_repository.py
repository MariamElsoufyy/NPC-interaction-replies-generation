import uuid
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector
from app.core.config import SIMILARITY_THRESHOLD
from app.db.models import FAQ




async def get_all_faqs(db: AsyncSession) -> list[FAQ]:
    result = await db.execute(select(FAQ).order_by(FAQ.created_at.desc()))
    return result.scalars().all()


async def get_faq_by_id(db: AsyncSession, faq_id: uuid.UUID) -> FAQ | None:
    result = await db.execute(select(FAQ).where(FAQ.id == faq_id))
    return result.scalar_one_or_none()


async def get_faqs_by_character(db: AsyncSession, character_id: str) -> list[FAQ]:
    result = await db.execute(
        select(FAQ).where(FAQ.character_id == character_id).order_by(FAQ.created_at.desc())
    )
    return result.scalars().all()


async def create_faq(db: AsyncSession, faq_data: dict) -> FAQ:
    faq = FAQ(**faq_data)
    db.add(faq)
    await db.commit()
    await db.refresh(faq)
    return faq


async def update_faq(db: AsyncSession, faq_id: uuid.UUID, updates: dict) -> FAQ | None:
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        return None
    for key, value in updates.items():
        setattr(faq, key, value)
    await db.commit()
    await db.refresh(faq)
    return faq


async def delete_faq(db: AsyncSession, faq_id: uuid.UUID) -> bool:
    result = await db.execute(delete(FAQ).where(FAQ.id == faq_id))
    await db.commit()
    return result.rowcount > 0


async def search_similar_faq(
    db: AsyncSession,
    embedding: list[float],
    character_id: str,
    threshold: float = SIMILARITY_THRESHOLD,
    limit: int = 1,
) -> FAQ | None:
    """Find the most similar FAQ for a given character using cosine similarity.

    The <=> operator is pgvector's cosine distance (0 = identical, 2 = opposite).
    We convert to similarity: similarity = 1 - distance.
    """
    # Cast embedding to vector for the query
    vector_col = FAQ.embedding.cast(Vector(1536))

    result = await db.execute(
        select(FAQ, (1 - vector_col.op("<=>")(embedding)).label("similarity"))
        .where(FAQ.character_id == character_id)
        .where(FAQ.embedding.is_not(None))
        .order_by(vector_col.op("<=>")(embedding))
        .limit(limit)
    )
    rows = result.all()

    if not rows:
        return None

    faq, similarity = rows[0]
    if similarity >= threshold:
        return faq

    return None
