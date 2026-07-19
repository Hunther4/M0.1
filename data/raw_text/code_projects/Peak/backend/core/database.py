import logging
from pathlib import Path

from sqlalchemy import event
from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "peak.db"
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_and_tables():
    """Create SQLModel-declared tables, then apply versioned migrations.

    Migration errors propagate to the caller — we deliberately do NOT wrap
    ``run_migrations`` in a try/except. A broken migration must abort
    application startup, not be silently downgraded to a warning.
    """
    SQLModel.metadata.create_all(engine)
    from migrations.runner import run_migrations

    run_migrations(engine)


def get_session():
    with Session(engine) as session:
        yield session
