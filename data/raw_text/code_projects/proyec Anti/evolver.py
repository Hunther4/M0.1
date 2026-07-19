"""
Skill Evolver for Anti-Agent.
Refactored to use requests instead of openai library.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import httpx
from datetime import datetime
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{1,}$")
_DYN_RE = re.compile(r"^dyn-(\d+)$")


def _sanitize_for_prompt(text: str, max_len: int = 1000) -> str:
    """
    Escape user-controlled log content before LLM prompt injection.
    Neutralizes backticks, JSON braces, and instruction-mimicking patterns.
    """
    if not text:
        return ""
    t = text[:max_len]
    # Neutralize markdown code fences and backtick injection
    t = t.replace("```", "'''")
    # Neutralize JSON braces used for prompt boundary confusion
    t = t.replace("{", "〔").replace("}", "〕")
    # Neutralize angle brackets used for XML/HTML tag injection
    t = t.replace("<", "〈").replace(">", "〉")
    return t

class SkillEvolver:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:1234/v1",
        model: str = "local-model",
        max_new_skills: int = 1,
        max_completion_tokens: int = 3000,
    ):
        self.max_new_skills = max_new_skills
        self.max_completion_tokens = max_completion_tokens
        self._model = model
        self.prm_url = f"{base_url.rstrip('/')}/chat/completions"

    async def evolve(
        self,
        failed_logs: list,
        current_skills: list,
    ) -> list[dict]:
        """
        Analyse failed_logs and propose new skills.
        """
        if not failed_logs:
            return []

        prompt = self._build_analysis_prompt(failed_logs, current_skills)

        try:
            response = await self._call_llm(prompt)
            raw_skills = self._parse_skills_response(response)
            skills = self._finalise_names(raw_skills)
            return skills[: self.max_new_skills]

        except Exception as e:
            logger.error(f"[SkillEvolver] LLM call failed: {e}", exc_info=True)
            return []

    async def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_completion_tokens,
            "temperature": 0.7
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(self.prm_url, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _build_analysis_prompt(
        self,
        failed_logs: list,
        current_skills: list,
    ) -> str:
        failure_blocks = []
        for i, log in enumerate(failed_logs[:5]):
            task = _sanitize_for_prompt(log.get("task", ""))
            result = _sanitize_for_prompt(log.get("result", ""))
            score = log.get("score", 0.0)
            
            # Extraer telemetría estructurada del sandbox si existe
            telemetry_info = ""
            if result and "[TELEMETRIA: SANDBOX_FAIL]" in result:
                match = re.search(r"\[TELEMETRIA: SANDBOX_FAIL\] exit_code=(\d+), error_type=(\w+)", result)
                if match:
                    exit_code = match.group(1)
                    error_type = match.group(2)
                    telemetry_info = f"\n⚠️ **ERROR TÉCNICO EN EJECUCIÓN (Sandbox/Local):** Exit Code `{exit_code}` ({error_type})\n"
            
            failure_blocks.append(
                f"### Falla {i + 1}  (Puntaje={score})\n"
                f"**Tarea/Instruccion:**\n{task}\n"
                f"{telemetry_info}"
                f"**Respuesta del Agente:**\n{result[:500]}...\n"
            )

        existing = [s.get("name", "") for s in current_skills]

        return (
            "Sos un Arquitecto de Inteligencia para un sistema agéntico avanzado.\n"
            "Tu trabajo: analizar las experiencias del agente (exitosas y fallidas) y generar reglas de ORO para mejorar su rendimiento.\n\n"
            "CRITERIOS DE EVOLUCIÓN:\n"
            "1. SÍNTESIS EXTREMA: ¿Cómo puede el agente decir lo mismo con menos palabras pero más datos?\n"
            "2. PROCESAMIENTO PREVIO: ¿Qué pasos de razonamiento faltaron para 'destilar' la info antes de responder?\n"
            "3. CALIDAD DE FUENTES: ¿Cómo evitar redundancia entre fuentes similares?\n"
            "4. MITIGACIÓN DE EXCEPCIONES: Si detectás un error de Sandbox (`[TELEMETRIA: SANDBOX_FAIL]`), diseña una habilidad conductual correctiva para evitar ese error específico (ej: ModuleNotFoundError, SyntaxError, FileNotFoundError) enseñándole al agente a validar la sintaxis, verificar precondiciones o importar librerías necesarias.\n\n"
            "---\n"
            "## Experiencias Recientes\n\n"
            + "\n\n".join(failure_blocks)
            + "\n\n---\n"
            "## Habilidades Existentes (NO duplicar)\n\n"
            + json.dumps(existing, indent=2)
            + "\n\n---\n"
            "## Instrucciones de Salida\n\n"
            f"Genera **1 a {self.max_new_skills}** nuevas reglas o habilidades. Enfocate en la DENSIDAD INFORMATIVA, EFICIENCIA de búsqueda y AUTOCORRECCIÓN de errores.\n\n"
            "Formato JSON:\n"
            "- `name`: slug (ej: `sintesis-de-fuentes`).\n"
            "- `description`: cuándo aplicar.\n"
            "- `content`: Guía Markdown (10-15 líneas) con: Objetivo, Pasos para 'destilar' info, y un **Anti-patrón** (ej: copiar y pegar resúmenes sin analizar).\n"
            "- `category`: `investigacion`, `codigo`, o `general`.\n\n"
            "**Salida:** Devuelve SOLO el array JSON."
        )

    def _parse_skills_response(self, response: str) -> list[dict]:
        clean = re.sub(r"```(?:json)?\s*", "", response).strip()
        j_start = clean.find("[")
        j_end = clean.rfind("]") + 1
        if j_start == -1 or j_end <= j_start:
            return []

        try:
            skills = json.loads(clean[j_start:j_end])
        except json.JSONDecodeError:
            return []

        valid = []
        for s in skills:
            missing = [k for k in ("name", "description", "content") if not s.get(k)]
            if not missing:
                valid.append(s)
        return valid

    def _finalise_names(self, skills: list[dict]) -> list[dict]:
        seen = set()
        result = []
        dyn_counter = 1

        for skill in skills:
            updated = dict(skill)
            name = skill.get("name", "").strip().lower()

            if _SLUG_RE.match(name) and name not in seen:
                pass 
            else:
                name = f"dyn-{dyn_counter:03d}"
                dyn_counter += 1

            seen.add(name)
            updated["name"] = name
            updated["category"] = skill.get("category", "general").strip()
            result.append(updated)

        return result

    # --- Sprint 3: Engram Merge (Dual Evolution) ---
    
    def _engram_similarity(self, content_a: str, content_b: str) -> float:
        """Compute similarity between two engram contents using token overlap."""
        tokens_a = set(re.findall(r'\w+', content_a.lower()))
        tokens_b = set(re.findall(r'\w+', content_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union) if union else 0.0
    
    async def _synthesize_merge(self, old_content: str, new_content: str, topic: str) -> Optional[str]:
        """Call LLM to synthesize two engram observations into one higher-density observation."""
        prompt = (
            "Eres un sistema de síntesis de memoria. Tu tarea es fusionar DOS observaciones "
            "relacionadas sobre el mismo tema en una ÚNICA observación de mayor densidad.\n\n"
            "REGLAS:\n"
            "1. Conserva TODA la información factual de ambas observaciones.\n"
            "2. Elimina redundancias — si ambos dicen lo mismo, dilo una vez.\n"
            "3. Mantén un tono neutral y factual.\n"
            "4. La salida debe ser un párrafo denso de 2-4 oraciones.\n\n"
            f"Tema: {topic}\n\n"
            f"Observación A:\n{old_content}\n\n"
            f"Observación B:\n{new_content}\n\n"
            "Observación fusionada (2-4 oraciones):"
        )
        
        try:
            response = await self._call_llm(prompt)
            merged = response.strip()
            return merged if len(merged) > 20 else None
        except Exception as e:
            logger.error(f"[SkillEvolver] Merge synthesis failed: {e}")
            return None
    
    def _check_duplicate_engram(self, new_content: str, engrams_dir: str):
        """
        Check for duplicate engrams. If similarity > 0.7, return (True, filename, old_data, similarity).
        Used by _merge_engram to trigger the merge flow.
        """
        new_tokens = set(re.findall(r'\w+', new_content.lower()))
        if not new_tokens:
            return False, None, None, 0.0
        
        if not os.path.exists(engrams_dir):
            return False, None, None, 0.0
        
        for filename in os.listdir(engrams_dir):
            if not filename.endswith(".json"):
                continue
            try:
                filepath = os.path.join(engrams_dir, filename)
                with open(filepath, "r") as f:
                    old_data = json.load(f)
                similarity = self._engram_similarity(new_content, old_data.get("content", ""))
                if similarity > 0.7:
                    return True, filename, old_data, similarity
            except Exception:
                continue
        
        return False, None, None, 0.0
    
    async def merge_engram(self, new_content: str, engrams_dir: str) -> dict:
        """
        Attempt to merge a new engram with existing ones.
        
        Returns:
            {"action": "merged", "filename": ..., "content": ...} if merged,
            {"action": "saved_new", "filename": ...} if no merge needed,
            {"action": "skipped", "reason": ...} otherwise.
        """
        is_dup, filename, old_data, similarity = self._check_duplicate_engram(new_content, engrams_dir)
        
        if not is_dup:
            # Save as new engram (handled by caller)
            return {"action": "saved_new", "filename": None, "similarity": similarity}
        
        # Merge: synthesize old + new
        topic = old_data.get("topic", "untitled")
        merged = await self._synthesize_merge(old_data.get("content", ""), new_content, topic)
        
        if merged:
            old_path = os.path.join(engrams_dir, filename)
            # Backup before removal — caller saves merged content next.
            # If caller's save fails, the backup can be restored.
            try:
                import shutil
                bak_path = old_path + ".bak"
                shutil.copy2(old_path, bak_path)
            except OSError:
                bak_path = None

            try:
                os.remove(old_path)
            except OSError:
                pass

            return {
                "action": "merged",
                "filename": filename,
                "topic": topic,
                "content": merged,
                "similarity": similarity,
            }
        
        # Synthesis failed — save new separately
        return {"action": "saved_new", "filename": None, "similarity": similarity, "note": "merge_synthesis_failed"}

    async def extract_engrams(self, logs: List[Dict]) -> List[Dict]:
        """
        Analyze logs to extract factual knowledge (Engrams).
        """
        if not logs:
            return []

        prompt = self._build_engram_prompt(logs)

        try:
            response = await self._call_llm(prompt)
            return self._parse_engrams_response(response)
        except Exception as e:
            logger.error(f"[SkillEvolver] Engram extraction failed: {e}", exc_info=True)
            return []

    def _build_engram_prompt(self, logs: list) -> str:
        blocks = []
        # Filter for successful tasks where information was found
        successful = [l for l in logs if l.get("success", False) and l.get("score", 0) > 0]
        
        for i, log in enumerate(successful[-5:]):
            blocks.append(f"### Tarea:\n{_sanitize_for_prompt(log.get('task', ''))}\n### Resultado:\n{_sanitize_for_prompt(log.get('result', ''), max_len=1000)}\n")

        return (
            "Sos el Hipocampo de un sistema de IA. Tu tarea es extraer CONOCIMIENTO FACTUAL (Engrams) de las recientes investigaciones.\n\n"
            "Un Engram es un dato duro, permanente y valioso que la IA debería recordar para no tener que buscarlo de nuevo en el futuro "
            "(ej: 'DeepSeek R1 fue lanzado en Enero 2025 y tiene 67B de parametros', 'El comando para reiniciar el servidor web es docker-compose restart').\n\n"
            "---\n## Investigaciones Recientes:\n\n"
            + "\n\n".join(blocks)
            + "\n\n---\n"
            "## Instrucciones\n"
            "Extrae hasta 3 Engrams clave basados SOLO en la información anterior.\n"
            "Devuelve SOLO un array JSON válido, donde cada objeto tenga:\n"
            "- `topic`: slug corto del tema (ej: 'deepseek-r1-specs').\n"
            "- `content`: Resumen denso y directo de los hechos.\n\n"
            "Si no hay datos duros valiosos, devuelve un array vacío []."
        )

    def _parse_engrams_response(self, response: str) -> list[dict]:
        clean = re.sub(r"```(?:json)?\s*", "", response).strip()
        j_start = clean.find("[")
        j_end = clean.rfind("]") + 1
        if j_start == -1 or j_end <= j_start:
            return []
        try:
            engrams = json.loads(clean[j_start:j_end])
            return [e for e in engrams if e.get("topic") and e.get("content")]
        except json.JSONDecodeError:
            return []
