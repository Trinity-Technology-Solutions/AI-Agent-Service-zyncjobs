"""Async database engine and session factory - lazy imports for compatibility."""
from recruitment_ai.config.settings import settings
import os
import logging

logger = logging.getLogger(__name__)

# Lazy init - import only when called
_engine = None
_session_factory = None
_Base = None
_initialized = False


async def init_db():
    global _engine, _session_factory, _Base, _initialized
    if _initialized:
        return

    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
        from recruitment_ai.database.base import Base

        USE_SQLITE = os.getenv("USE_SQLITE", "true").lower() == "true"

        if USE_SQLITE:
            SQLITE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "recruitment_ai.db")
            db_url = f"sqlite+aiosqlite:///{SQLITE_PATH}"
        else:
            db_url = settings.DATABASE_URL

        _engine = create_async_engine(db_url, echo=settings.DEBUG)
        _session_factory = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)

        from recruitment_ai.database.models import User, JobPost, Resume, Conversation, KnowledgeChunk
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        _initialized = True
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database init failed (non-critical): {e}")


async def get_db():
    if not _initialized:
        await init_db()
    if _session_factory is None:
        raise RuntimeError("Database not available")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
