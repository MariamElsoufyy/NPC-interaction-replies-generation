from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
import os
from dotenv import load_dotenv

load_dotenv()

# Auto-fix URL to use asyncpg driver
_raw_url = os.getenv("DATABASE_URL", "")
if _raw_url.startswith("postgresql://"):
    DATABASE_URL = _raw_url.replace("postgresql://", "postgresql+asyncpg://", 1)
elif _raw_url.startswith("postgres://"):
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql+asyncpg://", 1)
else:
    DATABASE_URL = _raw_url


class Base(DeclarativeBase):
    pass


def get_engine():
    return create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,   # verify connection is alive before handing it to a worker (prevents stale-connection errors)
    )


def get_session_factory(engine):
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    engine = get_engine()
    AsyncSessionLocal = get_session_factory(engine)
    async with AsyncSessionLocal() as session:
        yield session
