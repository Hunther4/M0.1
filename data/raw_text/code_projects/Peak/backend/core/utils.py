import logging
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def sanitize_prompt(text: str, max_len: int = 2000) -> str:
    """Sanitize user text before AI prompt interpolation.

    Strips ASCII control characters (\\x00-\\x1F) except \\t, \\n, \\r,
    then truncates to max_len characters. Returns empty string on None/empty input.

    Args:
        text: User-supplied text to sanitize.
        max_len: Maximum character length (default 2000).

    Returns:
        Sanitized string, or empty string if input is None/empty.
    """
    if not text:
        return ""
    # Strip control characters except \t (0x09), \n (0x0A), \r (0x0D)
    cleaned = "".join(ch for ch in text if ord(ch) >= 0x20 or ch in "\t\n\r")
    # Truncate to max_len
    return cleaned[:max_len]


def clean_json_response(content: str) -> str:
    """Limpia markdown de la respuesta de la IA y extrae el JSON."""
    if not content:
        return ""
    content = content.strip()
    # Intentar extraer bloque ```json ... ```
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', content, re.DOTALL)
    if json_match:
        content = json_match.group(1).strip()
    # Si arranca con "json", sacarlo
    for prefix in ["json", "JSON", "Json"]:
        if content.startswith(prefix):
            content = content[len(prefix):].strip()
            break
    # Buscar el primer { y último } como fallback
    if not content.startswith("{"):
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            content = content[start:end+1]
    return content


def parse_structured_json(content: str, response_model: Type[T]) -> Optional[T]:
    """Limpia content y parsea con Pydantic."""
    cleaned = clean_json_response(content)
    try:
        return response_model.model_validate_json(cleaned)
    except Exception as e:
        logger.error("Error parseando Pydantic: %s", e)
        return None
