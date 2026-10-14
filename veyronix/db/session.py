"""Engine + session lifecycle for the operational store."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from veyronix.core.config import get_settings
from veyronix.core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, echo=False, future=True, connect_args=connect_args)
    return engine


@lru_cache
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False, future=True)


def get_session() -> Session:
    """Return a new Session. Caller is responsible for close()."""
    return _session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on error, always close."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(drop: bool = False) -> None:
    """Create all tables (dev/local/CI convenience; prod uses Alembic).

    Importing models registers them on ``Base.metadata`` before create_all.
    """
    from veyronix.db import models  # noqa: F401  (register mappers)
    from veyronix.db.base import Base

    engine = get_engine()
    if drop:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    log.info("db.init", url=str(engine.url), tables=len(Base.metadata.tables), drop=drop)
