"""
Database engine and session management using SQLAlchemy.

Provides the database engine, session factory, and initialization
function. Uses SQLite by default.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level engine and session factory
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # Handle sqlite:/// path format
        db_url = settings.database_url
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
            echo=False,
        )
        # Enable foreign keys for SQLite
        if "sqlite" in db_url:
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info(f"Database engine created: {db_url}")
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )
    return _SessionLocal


def get_db() -> Session:
    """
    Get a database session.

    Yields:
        A SQLAlchemy Session that is automatically closed.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.

    This is safe to call multiple times — existing tables
    will not be dropped or modified.
    """
    from app.database.models import Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


def reset_engine() -> None:
    """Reset the engine and session factory (for testing)."""
    global _engine, _SessionLocal
    if _engine:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
