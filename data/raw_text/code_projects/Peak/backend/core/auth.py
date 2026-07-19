"""
Core authentication module — API key management and FastAPI auth middleware.

Architecture:
  - LazyAPIKeyManager: singleton that bootstraps the API key hash on first
    call (startup), caches it, and provides constant-time verification.
  - auth_middleware: FastAPI ASGI middleware that checks X-API-Key header
    against the cached hash before allowing requests through.
"""
import hmac
import logging
import os
import secrets

import bcrypt
from fastapi import Request
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

import core.database
from models.models import AppSetting

logger = logging.getLogger(__name__)


class LazyAPIKeyManager:
    """Manages a single API key with bcrypt-hashed storage and constant-time verify.

    The key is bootstrapped once at startup (via initialize()) and cached
    in memory for the lifetime of the process. verify() uses hmac.compare_digest
    for constant-time comparison against the cached hash.
    """

    def __init__(self):
        self._cached_hash: str | None = None

    def initialize(self) -> str | None:
        """Bootstrap the API key from env or DB, or generate a random one.

        Priority:
          1. PEAK_API_KEY env var → hash with bcrypt → upsert in AppSetting table
          2. Existing ``api_key_hash`` in AppSetting table → use it
          3. Neither → generate random 32-char hex key → hash → store in DB

        Returns:
            The raw (unhashed) key string when newly generated, None otherwise.
            The caller SHOULD log the returned value at startup.
        """
        env_key = os.environ.get("PEAK_API_KEY")

        with Session(core.database.engine) as session:
            if env_key:
                hashed = bcrypt.hashpw(env_key.encode(), bcrypt.gensalt()).decode()
                setting = session.exec(
                    select(AppSetting).where(AppSetting.key == "api_key_hash")
                ).first()
                if setting:
                    setting.value = hashed
                else:
                    setting = AppSetting(key="api_key_hash", value=hashed)
                    session.add(setting)
                session.commit()
                self._cached_hash = hashed
                return None

            setting = session.exec(
                select(AppSetting).where(AppSetting.key == "api_key_hash")
            ).first()
            if setting is not None:
                self._cached_hash = setting.value
                return None

            raw_key = secrets.token_hex(16)  # 32 hex chars
            hashed = bcrypt.hashpw(raw_key.encode(), bcrypt.gensalt()).decode()
            setting = AppSetting(key="api_key_hash", value=hashed)
            session.add(setting)
            session.commit()
            self._cached_hash = hashed
            logger.info("No API key found — generated new key")
            return raw_key

    def verify(self, provided_key: str) -> bool:
        """Constant-time comparison of *provided_key* against the cached hash.

        Uses hmac.compare_digest over bcrypt-hashed candidates to prevent
        timing side-channel attacks.

        Returns False when no hash is cached (not yet initialized) or when
        the key does not match.
        """
        if self._cached_hash is None:
            return False
        try:
            candidate = bcrypt.hashpw(
                provided_key.encode(), self._cached_hash.encode()
            )
            return hmac.compare_digest(candidate, self._cached_hash.encode())
        except Exception:
            return False

    def reset(self) -> None:
        """Drop the cached hash (useful for test isolation)."""
        self._cached_hash = None


# Module-level singleton — imported once and shared across the app.
api_key_manager = LazyAPIKeyManager()


async def auth_middleware(request: Request, call_next):
    """FastAPI middleware that enforces X-API-Key authentication.

    Skips auth when:
      - DISABLE_AUTH environment variable is set to ``1``.
      - The request method is OPTIONS (CORS preflight — handled before auth).
      - The request path starts with ``/api/health``.
    """
    if os.environ.get("DISABLE_AUTH") == "1":
        return await call_next(request)

    # Bypass auth for CORS preflight requests
    if request.method == "OPTIONS":
        return await call_next(request)

    if request.url.path.startswith("/api/health"):
        return await call_next(request)

    # FastAPI auto-generated docs are bypass-listed so developers can explore
    # the API during development. In production, set DISABLE_DOCS=1 or remove
    # this block to enforce auth on /docs and /openapi.json as well.
    if request.url.path in ("/", "/docs", "/openapi.json", "/redoc"):
        return await call_next(request)

    api_key = request.headers.get("X-API-Key")
    if not api_key or not api_key_manager.verify(api_key):
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key"},
        )

    return await call_next(request)
