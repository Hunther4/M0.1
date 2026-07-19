"""
Fixtures compartidos para todos los tests de Peak Practice.

Estrategia:
- DB en memoria con StaticPool para que TODAS las conexiones
  compartan la misma base de datos (crítico para SQLite :memory:)
- Parcheamos el engine global ANTES de importar módulos de la app
- autouse setup_db crea tablas antes de cada test, las destruye después
- IA mockeada: _executor.submit es no-op
"""
import os
import sys
from pathlib import Path

# Set rate limit for tests early — before any route module imports.
# This ensures the @limiter.limit() decorator reads the test value.
os.environ["RATE_LIMIT_PER_MINUTE"] = "9999"

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, delete

# StaticPool: UNA sola conexión compartida entre fixtures y TestClient.
# Sin esto, SQLite :memory: crea una DB DISTINTA por cada connection.
TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    echo=False,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

# Register the same PRAGMA listener that core/database.py uses on the real
# engine.  Since conftest replaces core.database.engine with TEST_ENGINE,
# the event listener registered at import time would be lost.
@event.listens_for(TEST_ENGINE, "connect")
def _set_test_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

# Parchear engine global ANTES de importar cualquier módulo de la app
import core.database  # noqa: E402

core.database.engine = TEST_ENGINE

import models.models  # noqa: F401, E402


@pytest.fixture(autouse=True)
def setup_db():
    """Crea tablas antes de cada test, las destruye después."""
    SQLModel.metadata.create_all(TEST_ENGINE)
    yield
    # Limpieza física estricta para evitar leaks entre tests debido a StaticPool
    with Session(TEST_ENGINE) as session:
        for table in reversed(SQLModel.metadata.sorted_tables):
            try:
                session.exec(delete(table))
            except Exception:
                pass
        session.commit()
    SQLModel.metadata.drop_all(TEST_ENGINE)
    # Reset auth manager cache so each test starts clean
    from core.auth import api_key_manager
    api_key_manager.reset()


@pytest.fixture(autouse=True)
def mock_ai():
    """
    Evita que background tasks ejecuten IA real.
    ThreadPoolExecutor.submit pasa a ser no-op.

    Parcheamos core.tasks.background_executor.submit directamente para
    que cualquier módulo que importe ``background_executor`` desde
    ``core.tasks`` reciba el mock (todos comparten la misma instancia).
    """
    with patch("core.tasks.background_executor.submit") as mock_submit:
        mock_submit.return_value = None
        yield


@pytest.fixture(name="engine")
def engine_fixture():
    """Retorna el engine de test (misma instancia siempre)."""
    return TEST_ENGINE


@pytest.fixture(name="session")
def db_session_fixture():
    """Sesión de SQLModel para manipular DB directamente."""
    with Session(TEST_ENGINE) as s:
        yield s


@pytest.fixture(name="client")
def client_fixture():
    """
    TestClient de FastAPI con dependency_overrides.
    Cada request abre su propia Session pero todas comparten
    el mismo TEST_ENGINE con StaticPool → misma DB.
    """
    from core.database import get_session
    from main import app

    def override_get_session():
        with Session(TEST_ENGINE) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def disable_auth(monkeypatch):
    """Disables auth for backward compatibility with existing tests."""
    monkeypatch.setenv("DISABLE_AUTH", "1")
    yield


@pytest.fixture(name="api_key")
def api_key_fixture(session):
    """Sets up a known API key in the AppSetting table and caches it on
    the module-level api_key_manager singleton.

    Uses upsert to avoid UNIQUE constraint conflicts with any key that
    was already inserted during TestClient lifespan initialization.
    """
    import bcrypt
    from sqlmodel import select

    from core.auth import api_key_manager
    from models.models import AppSetting

    raw = "test-api-key"
    hashed = bcrypt.hashpw(raw.encode(), bcrypt.gensalt()).decode()
    existing = session.exec(
        select(AppSetting).where(AppSetting.key == "api_key_hash")
    ).first()
    if existing:
        existing.value = hashed
    else:
        session.add(AppSetting(key="api_key_hash", value=hashed))
    session.commit()
    api_key_manager._cached_hash = hashed
    return raw


@pytest.fixture(name="auth_headers")
def auth_headers_fixture(api_key):
    """Returns valid X-API-Key header dict."""
    return {"X-API-Key": api_key}


@pytest.fixture(name="skill_factory")
def skill_factory_fixture(session):
    """Crea un Skill de prueba con valores por defecto."""
    from models.models import Skill

    def _create(
        slug="test-skill",
        name="Test Skill",
        domain="memory",
        skill_type="staircase",
        config_path="skills/test.yaml",
        level=1.0,
    ):
        skill = Skill(
            slug=slug,
            name=name,
            domain=domain,
            skill_type=skill_type,
            config_path=config_path,
            current_level=level,
        )
        session.add(skill)
        session.commit()
        session.refresh(skill)
        return skill

    return _create


@pytest.fixture(name="session_factory")
def session_factory_fixture(session, skill_factory):
    """Crea un PracticeSession de prueba con valores por defecto."""
    from datetime import datetime, timezone

    from models.models import Session as PracticeSession

    def _create(
        skill_id=None,
        entry_mode="quick",
        duration=15,
        difficulty=3,
        practiced="test practice session with enough detail",
        error="test micro error found",
    ):
        if skill_id is None:
            skill = skill_factory()
            skill_id = skill.id
        s = PracticeSession(
            skill_id=skill_id,
            entry_mode=entry_mode,
            duration_minutes=duration,
            difficulty=difficulty,
            what_i_practiced=practiced,
            micro_error_found=error,
            correction_applied="some correction",
            ai_fields_status="pending",
            created_at=datetime.now(timezone.utc),
        )
        session.add(s)
        session.commit()
        session.refresh(s)
        return s

    return _create
