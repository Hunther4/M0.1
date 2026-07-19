"""
MentalRep + Challenge generation para Peak (Fase 5).
Generación IA de representaciones mentales y desafíos personalizados.

Flujo MentalRep:
  1. Trigger detectado (level milestone / 20 sesiones / manual)
  2. Se pasa la representación anterior + resumen de sesiones
  3. IA compara y genera nueva descripción: "Antes veía X, ahora veo Y"
  4. IA decide si el cambio es real o no (distance check)

Flujo Challenge:
  1. Después de sesión deliberada, o después de assessment
  2. IA genera desafío concreto + difficulty_target
  3. Si hay RAG con libros, lo enriquece con citas
"""
import logging
from typing import Optional

from pydantic import BaseModel, Field

from core.ai import generate_structured_json
from core.rag import query_books
from core.utils import sanitize_prompt

logger = logging.getLogger(__name__)


# --- Modelos Pydantic para IA ---

class MentalRepResult(BaseModel):
    description: str = Field(..., description="Formato: 'Antes veía X, ahora veo Y'")
    is_real_shift: bool = Field(..., description="True si el cambio es genuino vs el anterior")
    reasoning: str = Field(..., description="Por qué esto es un cambio real en la comprensión")
    key_insight: str = Field(..., description="El insight principal que cambió tu entendimiento")


class ChallengeResult(BaseModel):
    description: str = Field(..., description="Desafío concreto y medible")
    difficulty_target: int = Field(ge=1, le=5, description="Dificultad objetivo 1-5")
    rationale: str = Field(..., description="Por qué este desafío ahora")


# --- Generación de MentalRep ---

def generate_mental_rep(
    skill_name: str,
    domain: str,
    skill_type: str,
    current_level: float,
    prev_description: Optional[str],
    prev_version: int,
    session_summary: str,
) -> Optional[MentalRepResult]:
    """
    Genera una nueva representación mental comparando la comprensión anterior
    con la actual, basada en el resumen de sesiones recientes.
    """
    # Consultar RAG si hay libros
    rag_chunks = query_books(f"{skill_name} mental model representation deliberate practice", top_k=2)
    rag_context = ""
    if rag_chunks:
        rag_lines = [
            f"Fragmento de referencia:\n{c['text'][:400]}\n---\nFuente: {c['book_title']}"
            for c in rag_chunks
        ]
        rag_context = "\n\nCONTEXTO DE LIBROS:\n" + "\n\n".join(rag_lines)

    system_prompt = f"""Coach Mental Representations (Ericsson). Dominio: {domain} | Tipo: {skill_type}

Tu trabajo: articular CÓMO cambió la comprensión del usuario.
FORMATO: "Antes veía X, ahora veo Y"

EJEMPLOS DE CAMBIO REAL:
- "Antes veía recursión como 'función que se llama a sí misma', ahora la veo como 'dividir problema en subproblemas idénticos + caso base'"
- "Antes veía escalas como 'dedos corriendo', ahora las veo como 'patrones de intervalos transferibles a cualquier tonalidad'"

EJEMPLOS DE NO-CAMBIO (is_real_shift=false):
- "Antes sabía 3 escalas, ahora sé 5" → MÁS CONOCIMIENTO, no distinta comprensión
- "Antes tardaba 10 min, ahora 5" → MÁS VELOCIDAD, no distinto modelo

REGLAS:
- is_real_shift=true SOLO si el MODELO MENTAL cambió, no si solo practicó más
- Sé honesto: si no hay shift real, marcá false
- NO inventes insights.{rag_context}"""

    clean_summary = sanitize_prompt(session_summary)
    if prev_description:
        user_prompt = f"""Habilidad: {skill_name}
Nivel actual: {current_level}/100
Versión anterior: v{prev_version}
Representación anterior: "{prev_description}"

Resumen de sesiones recientes:
{clean_summary}

Generá la nueva representación mental. Si el cambio es genuino, describilo.
Si no hay un cambio real en la comprensión, marcá is_real_shift=false."""
    else:
        user_prompt = f"""Habilidad: {skill_name}
Nivel actual: {current_level}/100
Esta es la PRIMERA representación mental (v1).

Resumen de sesiones recientes:
{clean_summary}

Generá la representación mental inicial del usuario sobre esta habilidad.
¿Cómo entiende él/ella esta habilidad en este momento?
Formato: "Veo/entiendo X como Y"."""

    from core.router import _build_augmented_system
    augmented = _build_augmented_system(system_prompt, MentalRepResult)
    result = generate_structured_json(augmented, user_prompt, MentalRepResult, task_type="assessment")
    return result


# --- Generación de Challenge ---

def generate_challenge(
    skill_name: str,
    domain: str,
    skill_type: str,
    current_level: float,
    last_session: str,
    difficulty_override: Optional[int] = None,
) -> Optional[ChallengeResult]:
    """
    Genera un desafío concreto para la próxima sesión.
    Se enriquece con RAG si hay libros indexados.
    """
    rag_chunks = query_books(f"{skill_name} deliberate practice next challenge", top_k=2)
    rag_context = ""
    source_book = None
    if rag_chunks:
        rag_lines = []
        for c in rag_chunks:
            rag_lines.append(
                f"Fragmento de referencia:\n{c['text'][:400]}\n---\nFuente: {c['book_title']}"
            )
        rag_context = "\n\nCONTEXTO DE LIBROS (usá estos fragmentos para inspirar el desafío):\n" + "\n\n".join(rag_lines)
        source_book = rag_chunks[0]["book_title"]

    system_prompt = f"""Coach práctica deliberate. Generá UN desafío CONCRETO + MEDIBLE para PRÓXIMA sesión.
Dominio: {domain} | Tipo: {skill_type}

REGLA DE ORO: "Practicar X" = MAL. "Hacer Y específico con criterio Z" = BIEN.

EJEMPLOS:
✗ "Practicar escalas"
✓ "Tocar escala mayor Do 2 octavas a 100bpm sin errores 3x seguidas"

✗ "Mejorar código"
✓ "Refactor parse_json: eliminar nested ifs, añadir 3 tests edge case, <50ms"

✗ "Escribir mejor"
✓ "Reescribir intro artículo: 3 frases max, voz activa, eliminar 5 adjetivos"

DIFICULTAD objetivo:
- <30 → 2-3
- 30-60 → 3-4
- >60 → 4-5
{rag_context}"""

    difficulty_guide = ""
    if difficulty_override:
        difficulty_guide = f"\nDificultad sugerida por el usuario: {difficulty_override}/5"
    elif current_level < 30:
        difficulty_guide = "\nDificultad sugerida: 2-3 (principiante/intermedio)"
    elif current_level < 60:
        difficulty_guide = "\nDificultad sugerida: 3-4 (intermedio/avanzado)"
    else:
        difficulty_guide = "\nDificultad sugerida: 4-5 (avanzado)"

    user_prompt = f"""Habilidad: {skill_name}
Nivel actual: {current_level}/100
Última sesión: {sanitize_prompt(last_session)}{difficulty_guide}

Generá el próximo desafío."""
    from core.router import _build_augmented_system
    augmented = _build_augmented_system(system_prompt, ChallengeResult)
    result = generate_structured_json(augmented, user_prompt, ChallengeResult, task_type="reasoning")

    if result and source_book:
        result.description = f"[{source_book}] {result.description}"

    return result


# --- Generación de Challenge Post-Assessment ---

def generate_challenge_from_assessment(
    skill_name: str,
    domain: str,
    skill_type: str,
    current_level: float,
    assessment_score: float,
    assessment_type: str,
    assessment_notes: str,
) -> Optional[ChallengeResult]:
    """
    Genera un challenge basado en los resultados de un assessment.
    """
    session_context = f"Assessment {assessment_type}: score={assessment_score}/100. Notas: {assessment_notes}"
    return generate_challenge(
        skill_name=skill_name,
        domain=domain,
        skill_type=skill_type,
        current_level=current_level,
        last_session=session_context,
    )
