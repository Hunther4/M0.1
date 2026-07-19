import json
import logging
import os
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from core.ai import generate_structured_json as local_generate_structured_json
from core.settings import get_setting, set_setting
from core.utils import parse_structured_json

logger = logging.getLogger(__name__)

load_dotenv()

# Las keys se leen dinámicamente en execute_with_router para soportar hot-reload del .env

T = TypeVar('T', bound=BaseModel)

# Lazy-accessor cache for API keys.
# Each key is read from the environment on first call to its accessor and cached.
_key_cache: dict[str, str] = {}


def _get_groq_key() -> str:
    """Return GROQ_API_KEY, reading from env on first call only."""
    if "groq" not in _key_cache:
        _key_cache["groq"] = os.getenv("GROQ_API_KEY", "")
    return _key_cache["groq"]


def _get_openrouter_key() -> str:
    """Return OPENROUTER_API_KEY, reading from env on first call only."""
    if "openrouter" not in _key_cache:
        _key_cache["openrouter"] = os.getenv("OPENROUTER_API_KEY", "")
    return _key_cache["openrouter"]


def get_ai_mode() -> str:
    """Devuelve 'local' o 'api'. Por defecto 'local'."""
    return get_setting("ai_mode", "local")

def set_ai_mode(mode: str) -> None:
    if mode not in ["local", "api"]:
        mode = "local"
    set_setting("ai_mode", mode)

def _clean_and_parse_json(content: str, response_model: Type[T]) -> Optional[T]:
    return parse_structured_json(content, response_model)


def _build_augmented_system(system_prompt: str, response_model: Type[T]) -> str:
    """Build JSON enforcement block — single source of truth for all providers."""
    schema = response_model.model_json_schema()
    required = schema.get("required", [])
    fields = ", ".join(f'"{k}"' for k in required)

    # Generic positive example — works for ANY schema
    example_values = {}
    for k, v in schema.get("properties", {}).items():
        t = v.get("type", "any")
        if t == "boolean":
            example_values[k] = True
        elif t == "number" or t == "integer":
            example_values[k] = 50
        elif t == "array":
            example_values[k] = []
        else:
            example_values[k] = "..."
    example_json = json.dumps(example_values, ensure_ascii=False)

    return (
        f"{system_prompt}\n\n"
        f"REGLAS DE SALIDA (OBLIGATORIAS):\n"
        f"1. Respondé SOLO un objeto JSON válido. Nada más.\n"
        f"2. Campos requeridos ({len(required)}): {fields}\n"
        f"3. NO uses markdown, NO expliques, NO repitas el esquema.\n"
        f"4. Ejemplo de FORMATO (valores son placeholders):\n"
        f"{example_json}\n"
        f"5. Si un campo es opcional y no tenés dato, usá null o [] según tipo."
    )


def _call_groq(system_prompt: str, user_prompt: str, response_model: Type[T], model_id: str) -> Optional[T]:
    groq_api_key = _get_groq_key()
    if not groq_api_key:
        logger.warning("Groq omitido (Sin API Key)")
        return None

    try:
        from groq import Groq
        client = Groq(api_key=groq_api_key, timeout=30.0)

        augmented_system = _build_augmented_system(system_prompt, response_model)

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content
        return _clean_and_parse_json(content, response_model)
    except Exception as e:
        logger.error("Groq Falló: %s: %s", type(e).__name__, e)
        return None

def _call_openrouter(system_prompt: str, user_prompt: str, response_model: Type[T], model_id: str) -> Optional[T]:
    openrouter_api_key = _get_openrouter_key()
    if not openrouter_api_key:
        logger.warning("OpenRouter omitido (Sin API Key)")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            timeout=30.0
        )

        augmented_system = _build_augmented_system(system_prompt, response_model)

        # Opcional: headers sugeridos por OpenRouter
        extra_headers = {
            "HTTP-Referer": "http://localhost:5173",
            "X-Title": "Peak Practice"
        }

        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": augmented_system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            extra_headers=extra_headers
        )
        content = response.choices[0].message.content
        return _clean_and_parse_json(content, response_model)
    except Exception as e:
        logger.error("OpenRouter Falló: %s: %s", type(e).__name__, e)
        return None

def execute_with_router(task_type: str, system_prompt: str, user_prompt: str, response_model: Type[T]) -> Optional[T]:
    """Ejecuta la llamada a la IA con orden de preferencia configurable.

    Orden de fallback (nuevo, configurable):
      1. LM Studio local (1 intento, sin retry de timeout)
      2. Si LM Studio responde pero da error de parseo/structured output
         → 1 retry adicional en LM Studio (segunda oportunidad)
      3. Si LM Studio no responde o falla el retry → API (Groq → OpenRouter)
      4. Si API falla → None (fail fast, sin más reintentos)

    Esto prioriza LM Studio sobre API cuando está disponible, pero no se
    queda colgado esperando timeouts largos.
    """
    load_dotenv()
    mode = get_ai_mode()

    if mode == "local":
        logger.info("Ruteando con preferencia LM Studio → API fallback (modo local con resiliencia)...")
        # In local mode, prefer LM Studio but fall back to API if it's not
        # available. Without this, the call would hang for up to 120s
        # (60s timeout × max_retries=1) when LM Studio is down — the
        # audit, mental, and quick-log paths would silently return a
        # fallback result with no signal to the caller.
        lm_result = _try_lm_studio_with_retry(system_prompt, user_prompt, response_model)
        if lm_result is not None:
            return lm_result
        logger.info("LM Studio no disponible en modo local — intentando API fallback")
        return _try_api_fallback(task_type, system_prompt, user_prompt, response_model)

    # Modo API (con preferencia LM Studio primero)
    logger.info("Ruteando con preferencia LM Studio → API fallback...")

    # Check for manual model selection (override del flujo automático)
    selected_raw = get_setting("selected_model", "")
    selected = None
    if selected_raw:
        try:
            selected = json.loads(selected_raw)
        except (json.JSONDecodeError, TypeError):
            selected = None

    is_auto = selected is None or selected.get("auto", True)

    if not is_auto:
        provider = selected.get("provider")
        model_id = selected.get("model_id")
        model_name = selected.get("model_name", "")

        if provider == "groq" and model_id:
            logger.info("Manual: usando Groq (%s)", model_name)
            return _call_groq(system_prompt, user_prompt, response_model, model_id) or _try_api_fallback(task_type, system_prompt, user_prompt, response_model)
        elif provider == "openrouter" and model_id:
            logger.info("Manual: usando OpenRouter (%s)", model_name)
            return _call_openrouter(system_prompt, user_prompt, response_model, model_id) or _try_api_fallback(task_type, system_prompt, user_prompt, response_model)
        elif provider == "lm_studio":
            logger.info("Manual: usando LM Studio (%s)", model_name)
            return _try_lm_studio_with_retry(system_prompt, user_prompt, response_model) or _try_api_fallback(task_type, system_prompt, user_prompt, response_model)

    # Modo API
    logger.info("Ruteando a API Mode...")

    # Smart selection: LM Studio (1 retry) → API Groq → API OpenRouter
    logger.info("Orden: LM Studio → API Groq → API OpenRouter")

    # 1. Intentar LM Studio (1 intento, 1 retry si responde con error)
    lm_result = _try_lm_studio_with_retry(system_prompt, user_prompt, response_model)
    if lm_result is not None:
        return lm_result

    # 2-3. Fallback API: prueba TODOS los modelos Groq, luego TODOS los OpenRouter
    return _try_api_fallback(task_type, system_prompt, user_prompt, response_model)


def _is_lm_studio_reachable() -> bool:
    """Quick TCP check (2s) — is LM Studio listening? Avoids 60s timeout waste."""
    import socket
    from urllib.parse import urlparse
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 1234
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except (OSError, TimeoutError):
        return False


def _try_lm_studio_with_retry(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
) -> Optional[T]:
    """Intenta LM Studio una vez. Si el servidor responde pero da error
    de parseo, hace 1 retry adicional. Si no hay servidor, retorna None.
    """
    from openai import APIConnectionError, APITimeoutError

    # Fast health check: skip 60s timeout if port is closed
    if not _is_lm_studio_reachable():
        logger.info("LM Studio no disponible (health check falló) — saltando directo a API")
        return None

    # Intento 1
    try:
        result = local_generate_structured_json(
            system_prompt, user_prompt, response_model, force_local=True, max_retries=0
        )
        if result is not None:
            return result
        # Servidor respondió pero no pudimos parsear → 1 retry
        logger.warning("LM Studio respondió pero output inválido — retry 1 vez")
    except (APIConnectionError, APITimeoutError):
        logger.warning("LM Studio no responde (servidor caído)")
        return None
    except Exception as e:
        # BadRequestError with "No models loaded" = LM Studio running but empty
        if "No models loaded" in str(e):
            logger.warning("LM Studio corriendo pero sin modelo cargado")
            return None
        logger.warning("LM Studio error inesperado: %s", e)
        return None

    # Intento 2 (retry único por respuesta inválida)
    try:
        result = local_generate_structured_json(
            system_prompt, user_prompt, response_model, force_local=True, max_retries=0
        )
        if result is not None:
            return result
        logger.warning("LM Studio retry también dio output inválido")
    except Exception as e:
        if "No models loaded" in str(e):
            logger.warning("LM Studio sin modelo — saltando a API")
        else:
            logger.warning("LM Studio retry error: %s", e)

    return None


def _try_api_fallback(
    task_type: str,
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
) -> Optional[T]:
    """Fallback a API: prueba TODOS los modelos de Groq (ordenados por score),
    luego TODOS los de OpenRouter. Return en el primero que funcione."""
    from core.model_registry import get_available_models

    # 1. Groq — rápido, prioridad
    groq_models = get_available_models(provider="groq")
    groq_models.sort(key=lambda m: m.score, reverse=True)
    for model in groq_models:
        logger.info("Intentando Groq: %s (%s)", model.name, model.model_id)
        res = _call_groq(system_prompt, user_prompt, response_model, model.model_id)
        if res:
            return res

    # 2. OpenRouter — más lento, más modelos
    or_models = get_available_models(provider="openrouter")
    or_models.sort(key=lambda m: m.score, reverse=True)
    for model in or_models:
        logger.info("Intentando OpenRouter: %s (%s)", model.name, model.model_id)
        res = _call_openrouter(system_prompt, user_prompt, response_model, model.model_id)
        if res:
            return res

    return None
