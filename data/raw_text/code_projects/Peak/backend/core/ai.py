import json
import logging
import os
from typing import Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from core.utils import parse_structured_json, sanitize_prompt

logger = logging.getLogger(__name__)

# Lazy client — not initialized at import time.
# First call to get_client() reads env vars and caches the instance.
_client: Optional[OpenAI] = None

T = TypeVar('T', bound=BaseModel)


def get_client() -> OpenAI:
    """Return the OpenAI client, creating it lazily on first call.

    Reads LM_STUDIO_API_KEY, LM_STUDIO_BASE_URL, and LM_STUDIO_TIMEOUT
    from the environment on first invocation; caches the instance thereafter.
    """
    global _client
    if _client is None:
        base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        timeout_str = os.getenv("LM_STUDIO_TIMEOUT", "60")
        try:
            timeout = int(timeout_str)
        except (ValueError, TypeError):
            timeout = 60
        _client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    return _client


# Backward-compatible module-level attribute for existing code and mock patches.
# "core.ai.client" resolves to the lazy getter via __getattr__.
# This __getattr__ is only called on attribute access, NOT at import time.
def __getattr__(name):
    if name == "client":
        return get_client()
    raise AttributeError(f"module 'core.ai' has no attribute '{name}'")


class QuickLogCompletion(BaseModel):
    correction_applied: str
    hypothesis_tomorrow: str


def generate_structured_json(system_prompt: str, user_prompt: str, response_model: Type[T], max_retries: int = 1, force_local: bool = False, task_type: str = "audit") -> Optional[T]:
    """
    Llama a la IA pidiendo JSON estructurado.
    Si force_local=True, saltea el router y ejecuta directo en LM Studio.
    Si force_local=False (default), SIEMPRE delega al router — incluso en modo
    "local". Esto garantiza el fallback completo: LM Studio (1 intento + 1 retry)
    → API Groq → API OpenRouter. El modo "local" ahora significa "preferí LM
    Studio" pero sin colgarse si el servidor local está caído.
    task_type permite propagar la capability al router (e.g., "audit", "assessment", "reasoning").

    force_local es usado por el router para evitar recursión infinita
    cuando todos los providers API fallan y se cae a local.
    """
    if not force_local:
        from core.router import execute_with_router
        return execute_with_router(task_type, system_prompt, user_prompt, response_model)

    for attempt in range(max_retries + 1):
        try:
            from core.router import _build_augmented_system
            augmented_system = _build_augmented_system(system_prompt, response_model)

            client = get_client()
            response = client.chat.completions.create(
                model="local-model",  # LM Studio ignora el nombre del modelo
                messages=[
                    {"role": "system", "content": augmented_system},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
            )

            if not response.choices:
                logger.warning("Intento %d: choices vacío", attempt + 1)
                continue

            content = response.choices[0].message.content
            if not content:
                logger.warning("Intento %d: respuesta vacía", attempt + 1)
                continue

            parsed = parse_structured_json(content, response_model)
            if parsed:
                return parsed

        except Exception as e:
            # "No models loaded" means LM Studio is running but empty —
            # propagate so the router can skip immediately to API.
            if "No models loaded" in str(e):
                raise
            logger.error("Error en intento %d: %s: %s", attempt + 1, type(e).__name__, e)
            if attempt == max_retries:
                return None
    return None


def generate_quick_log_completions(what_i_practiced: str, micro_error_found: str, domain: str) -> dict:
    """
    Genera `correction_applied` y `hypothesis_tomorrow` para el modo Quick Log.
    Delega al router en background.
    """
    system_prompt = (
        f"Coach práctica deliberada — Quick Log (dominio: {domain}).\n"
        f"Usuario fatigado. Completá 2 campos basándote en su objetivo + error.\n\n"
        f"EJEMPLOS de buenos outputs:\n"
        f'- Objetivo: "compás 14-16 piano" | Error: "pulgar tarde en negro"\n'
        f'  → correction_applied: "ralentice 50% y aislé pulgar 5 min"\n'
        f'  → hypothesis_tomorrow: "mañana metrónomo 60bpm, foco pulgar"\n\n'
        f'- Objetivo: "función parse_json" | Error: "no maneja keys faltantes"\n'
        f'  → correction_applied: "agregué .get() con default y test edge cases"\n'
        f'  → hypothesis_tomorrow: "mañana refactor a try/except y benchmarkeo"\n\n'
        f"REGLAS:\n"
        f"- NO uses genéricos ('practiqué más', 'revisaré', 'seguiré intentando')\n"
        f"- SÍ: acción concreta + detalle técnico del dominio\n"
        f"- Máx 2 oraciones c/u\n"
    )

    user_prompt = (
        f"Objetivo: {sanitize_prompt(what_i_practiced)}\n"
        f"Error: {sanitize_prompt(micro_error_found)}\n\n"
        f"Completá JSON:"
    )

    from core.router import get_ai_mode
    if get_ai_mode() == "api":
        from core.router import execute_with_router
        result = execute_with_router("quick_log", system_prompt, user_prompt, QuickLogCompletion)
    else:
        from core.router import _build_augmented_system
        augmented = _build_augmented_system(system_prompt, QuickLogCompletion)
        result = generate_structured_json(augmented, user_prompt, QuickLogCompletion, task_type="quick_log")

    if result:
        return result.model_dump()
    return {
        "correction_applied": "No se pudo autogenerar — LM Studio/Router no respondió.",
        "hypothesis_tomorrow": "Revisar sesión manualmente en la UI."
    }
