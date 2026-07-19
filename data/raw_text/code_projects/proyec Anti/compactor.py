"""
HybridCompactor - Sistema híbrido de compactación para Engram v0.6 FINAL

Este archivo contiene todas las mejoras:
- v0.4: HybridCompactor original
- v0.5: Jaccard deduplication, Matriz de Integridad
- v0.5.1: Role-Aware + Context Guard (FIX)
- v0.6: Presión Adaptativa (Sentinel), Semantic Truncate
"""

import re
import os
import json
import asyncio
import logging
from typing import Optional, Callable, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# === Optional: TF-IDF / Cosine Similarity dependencies (LAZY) ===
_HAS_SKLEARN = False
_TFIDF_VECTORIZER = None

def _check_sklearn():
    """Lazy-check for sklearn availability."""
    global _HAS_SKLEARN
    if not _HAS_SKLEARN:
        try:
            import sklearn.feature_extraction.text
            _HAS_SKLEARN = True
        except (ImportError, Exception):
            _HAS_SKLEARN = False
    return _HAS_SKLEARN

# === Optional: SentenceTransformer (preferred, but heavy) ===
_HAS_SENTENCE_TRANSFORMER = False
_SENTENCE_MODEL = None

def _check_sentence_transformer():
    """Lazy-check for sentence_transformers availability."""
    global _HAS_SENTENCE_TRANSFORMER
    if not _HAS_SENTENCE_TRANSFORMER:
        try:
            import sentence_transformers
            _HAS_SENTENCE_TRANSFORMER = True
        except (ImportError, Exception):
            _HAS_SENTENCE_TRANSFORMER = False
    return _HAS_SENTENCE_TRANSFORMER


def _get_embedding_model():
    """Lazy-load the embedding model — SentenceTransformer preferred, TF-IDF fallback."""
    global _SENTENCE_MODEL, _TFIDF_VECTORIZER

    if _check_sentence_transformer() and _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using SentenceTransformer 'all-MiniLM-L6-v2' for embeddings.")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")

    if _SENTENCE_MODEL is not None:
        return "sentence_transformer", _SENTENCE_MODEL

    if _check_sklearn():
        if _TFIDF_VECTORIZER is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            _TFIDF_VECTORIZER = TfidfVectorizer(stop_words="english", max_features=5000)
        return "tfidf", _TFIDF_VECTORIZER

    return "none", None


class HybridCompactor:
    """Sistema híbrido de compactación."""
    
    DEFAULT_GUIDELINES = """
## PRESERVE (alta prioridad)
- Instrucciones del sistema
- Transacciones recientes (últimas 20)
- Categorías configuradas del usuario
- Balance actual
- Preferencias de formato

## DISCARD (puede comprimir)
- Historial antiguo (>30 días)
- Logs de debug
- Estados UI temporales
- Mensajes repetitivos
"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # Configuración sliding window
        self.max_messages = self.config.get("max_messages", 50)
        self.summary_threshold = self.config.get("summary_threshold", 30)
        
        # Configuración overlap
        self.overlap_size = self.config.get("overlap_size", 3)
        
        # Guidelines dinámicos
        self.guidelines = self.DEFAULT_GUIDELINES
        
        # Tracking de failures/successes
        self.failures = []
        self.successes = []
        
        # Preservation layer
        self.storage_path = self.config.get("storage_path", "./.engram_compacted")
        
        # Métricas
        self.compaction_count = 0
        self.total_tokens_saved = 0
        self.messages_deduplicated = 0
        
        # v0.6: Presión Adaptativa (Sentinel)
        self.adaptive_thresholds = {
            "safe": 1.0,      # < 50%: inactivo
            "warning": 0.7,   # 50-85%: normal
            "critical": 0.4,   # 85-95%: agresivo
        }
        
        # TF-IDF corpus buffer for meaningful IDF weights
        self._tfidf_corpus = []
        self._tfidf_max_corpus = 100
    
    # ==================== TOKEN COUNTING ====================
    
    def count_tokens(self, text: str) -> int:
        """Cuenta tokens de forma precisa usando regex."""
        if not text:
            return 0
        words = re.findall(r"\b[\w']+\b", text)
        punct = re.findall(r"[^\w\s]", text)
        return len(words) + len(punct)
    
    def count_tokens_estimate(self, text: str) -> int:
        """Estimación rápida - divide por 4 chars."""
        return len(text) // 4
    
    def _messages_tokens(self, messages: List[Dict]) -> int:
        """Cuenta tokens total de mensajes."""
        return sum(self.count_tokens(m.get("content", "")) for m in messages)
    
    # ==================== SLIDING WINDOW ====================
    
    async def sliding_window(self, messages: List[Dict]) -> List[Dict]:
        """Aplica sliding window simple."""
        if len(messages) <= self.max_messages:
            return messages
        
        excess = messages[:-self.max_messages]
        summary = await self._auto_summary(excess)
        
        result = [{"role": "system", "content": f"[RESUMEN: {summary}]"}]
        result.extend(messages[-self.max_messages:])
        
        return result
    
    # ==================== SPRINT 3: LLM-BASED SUMMARY ====================
    
    async def _call_llm(self, prompt: str, max_tokens: int = 256) -> Optional[str]:
        """Call a local LLM via HTTP endpoint (LM Studio / LocalAI compatible)."""
        import requests

        base_url = self.config.get("llm_base_url", "http://127.0.0.1:1234/v1")
        model = self.config.get("llm_model", "local-model")
        url = f"{base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }

        max_retries = 3

        for attempt in range(max_retries):
            try:
                resp = await asyncio.to_thread(
                    lambda: requests.post(url, json=payload, timeout=30)
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if e.response is not None else 0
                if status in (429, 503) and attempt < max_retries - 1:
                    wait = (2 ** attempt) * 0.5
                    logger.warning(
                        f"[Compactor] LLM returned {status}, "
                        f"retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.warning(f"[Compactor] LLM summary call failed: {e}")
                return None
            except Exception as e:
                logger.warning(f"[Compactor] LLM summary call failed: {e}")
                return None

        return None
    
    async def _auto_summary(self, messages: List[Dict]) -> str:
        """
        Genera 'State of the Conversation' summary (Sprint 3).
        Uses LLM if available; falls back to heuristic summary.
        """
        if not messages:
            return "Sin historial"
        
        # Try LLM-based summary first
        transcript = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')[:300]}"
            for m in messages[-10:]
        )
        
        summary_prompt = (
            "Resume el 'Estado de la Conversación' en 2-3 oraciones.\n"
            "Incluye: decisiones clave tomadas, estado actual, y cualquier información "
            "crítica que el asistente necesite saber para continuar.\n\n"
            f"Conversación:\n{transcript}\n\n"
            "Resumen (2-3 oraciones):"
        )
        
        llm_summary = await self._call_llm(summary_prompt)
        if llm_summary:
            return llm_summary
        
        # Fallback: heuristic summary
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        
        topics = set()
        for msg in messages:
            content = msg.get("content", "")
            words = re.findall(r"\b(\w{5,})\b", content.lower())
            topics.update(words)
        
        summary = f"{len(messages)} mensajes, {len(user_msgs)} del usuario, {len(assistant_msgs)} del asistente"
        
        if topics:
            top_topics = list(topics)[:5]
            summary += f", temas: {', '.join(top_topics)}"
        
        return summary
    
    # ==================== FAILURE-DRIVEN ====================
    
    def record_result(self, task: str, success: bool, info_used: List[str]):
        """Registra resultado de una tarea."""
        entry = {
            "task": task,
            "info_used": info_used,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            self.successes.append(entry)
        else:
            self.failures.append(entry)
        
        if len(self.failures) > 100:
            self.failures = self.failures[-100:]
        if len(self.successes) > 100:
            self.successes = self.successes[-100:]
    
    def optimize_guidelines(self, llm: Optional[Callable] = None) -> str:
        """Optimiza guidelines basándose en failures."""
        if not self.failures:
            return self.guidelines
        
        if llm is None:
            missing_info = set()
            for f in self.failures:
                missing_info.update(f.get("info_used", []))
            
            new_guideline = f"\n## FAILURE-DRIVEN PRESERVE\n"
            new_guideline += f"- Info fallida: {', '.join(list(missing_info)[:10])}\n"
            
            self.guidelines += new_guideline
        else:
            prompt = f"""
Analiza estos failures:
{json.dumps(self.failures[-10:], indent=2)}

Y estos successes:
{json.dumps(self.successes[-10:], indent=2)}

Mejora los compression guidelines.
"""
            new_guidelines = llm(prompt)
            if new_guidelines:
                self.guidelines = new_guidelines
        
        return self.guidelines
    
    # ==================== COMPRESSION ====================
    
    async def compress(self, messages: List[Dict], budget_tokens: int) -> List[Dict]:
        """Aplica compresión híbrida."""
        if not messages:
            return messages
        
        total = self._messages_tokens(messages)
        
        if total <= budget_tokens:
            return messages
        
        if len(messages) > self.max_messages:
            messages = await self.sliding_window(messages)
            self.compaction_count += 1
        
        total = self._messages_tokens(messages)
        
        if total <= budget_tokens:
            return messages
        
        messages = self._filter_by_guidelines(messages)
        
        total = self._messages_tokens(messages)
        
        if total > budget_tokens:
            messages = self._hard_truncate(messages, budget_tokens)
        
        return messages
    
    def _filter_by_guidelines(self, messages: List[Dict]) -> List[Dict]:
        """Filtra mensajes basándose en guidelines."""
        preserve_roles = {"system", "user"}
        
        preserve_keywords = self._extract_keywords("PRESERVE")
        discard_keywords = self._extract_keywords("DISCARD")
        
        preserve = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role in preserve_roles:
                preserve.append(msg)
                continue
            
            # Preservar respuestas del asistente que siguen a mensajes del usuario
            if role == "assistant" and i > 0 and messages[i-1].get("role") == "user":
                preserve.append(msg)
                continue
                
            # Preservar mensajes con indicadores de tool results
            if "[RESULT" in content or "[ERROR]" in content:
                preserve.append(msg)
                continue

            content_lower = content.lower()
            
            should_preserve = True
            
            for kw in preserve_keywords:
                if kw.lower() in content_lower:
                    should_preserve = True
                    break
            
            for kw in discard_keywords:
                if kw.lower() in content_lower:
                    should_preserve = False
                    break
            
            if should_preserve:
                preserve.append(msg)
        
        return preserve
    
    def _extract_keywords(self, section: str) -> List[str]:
        """Extrae keywords de una sección de guidelines."""
        keywords = []
        
        lines = self.guidelines.split("\n")
        in_section = False
        
        for line in lines:
            line = line.strip()
            
            if line == f"## {section}":
                in_section = True
                continue
            
            if in_section:
                if line.startswith("## "):
                    break
                if line.startswith("-"):
                    kw = line.lstrip("- ").strip()
                    if kw and len(kw) > 2:
                        keywords.append(kw)
        
        return keywords
    
    def _hard_truncate(self, messages: List[Dict], budget_tokens: int) -> List[Dict]:
        """Truncate directo cuando todo lo demás falla."""
        result = []
        remaining = budget_tokens
        
        for msg in messages:
            content = msg.get("content", "")
            tokens = self.count_tokens(content)
            
            if tokens <= remaining:
                result.append(msg)
                remaining -= tokens
            else:
                chars = remaining * 4
                result.append({
                    "role": msg.get("role", "assistant"),
                    "content": content[:chars] + "\n[TRUNCATED]"
                })
                break
        
        return result
    
    # ==================== PRESERVATION LAYER ====================
    
    def preserve_to_disk(self, messages: List[Dict], label: str) -> str:
        """Persiste mensajes a filesystem."""
        os.makedirs(self.storage_path, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.storage_path}/{label}_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, default=str)
        
        return filename
    
    # ==================== U-SHAPE ====================
    
    def ushape_order(self, messages: List[Dict]) -> List[Dict]:
        """U-shape ordering para máxima atención."""
        if len(messages) <= 3:
            return messages
        
        system_msgs = [m for m in messages if m.get("role") == "system"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        other_msgs = [m for m in messages if m.get("role") not in ["system", "user"]]
        
        high_priority = other_msgs[-2:] if len(other_msgs) > 2 else other_msgs
        low_priority = other_msgs[:-2] if len(other_msgs) > 2 else []
        
        result = []
        result.extend(system_msgs)
        
        for msg in low_priority:
            result.append(msg)
        
        for msg in reversed(high_priority):
            result.append(msg)
        
        result.extend(user_msgs)
        
        return result
    
    # ==================== METRICS ====================
    
    def get_stats(self) -> dict:
        """Retorna estadísticas."""
        failure_rate = 0
        if self.failures or self.successes:
            failure_rate = len(self.failures) / (len(self.failures) + len(self.successes))
        
        return {
            "failure_rate": round(failure_rate, 3),
            "total_failures": len(self.failures),
            "total_successes": len(self.successes),
            "compaction_count": self.compaction_count,
            "tokens_saved": self.total_tokens_saved,
            "messages_deduplicated": self.messages_deduplicated,
        }
    
    # ==================== SPRINT 3: SEMANTIC DEDUPLICATION ====================
    
    def _get_text_embedding(self, text: str):
        """Get embedding vector for text."""
        if not text:
            return None
        kind, model = _get_embedding_model()
        
        if kind == "sentence_transformer":
            return model.encode([text], convert_to_tensor=False)[0]
        
        if kind == "tfidf" and model is not None:
            # Maintain a growing corpus for meaningful IDF weights
            self._tfidf_corpus.append(text)
            if len(self._tfidf_corpus) > self._tfidf_max_corpus:
                self._tfidf_corpus = self._tfidf_corpus[-self._tfidf_max_corpus:]
            
            # Fit on accumulated corpus (gives non-zero IDF)
            model.fit(self._tfidf_corpus)
            return model.transform([text])
        
        return None
    
    def _cosine_similarity_vectors(self, vec1, vec2) -> float:
        """Compute cosine similarity between two vectors."""
        if vec1 is None or vec2 is None:
            return 0.0
        
        # SentenceTransformer → numpy arrays
        if hasattr(vec1, "shape") and hasattr(vec2, "shape") and len(vec1.shape) == 1:
            import numpy as np
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        
        # TF-IDF → sparse matrices
        if hasattr(vec1, "toarray") and _check_sklearn():
            from sklearn.metrics.pairwise import cosine_similarity
            return float(cosine_similarity(vec1, vec2)[0][0])
        
        return 0.0
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity via embeddings (cosine)."""
        if not text1 or not text2:
            return 0.0
        
        if not _check_sklearn() and not _check_sentence_transformer():
            # Fallback to Jaccard if no ML libs available
            set1 = set(text1.lower().split())
            set2 = set(text2.lower().split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union if union > 0 else 0.0
        
        kind, model = _get_embedding_model()
        
        if kind == "sentence_transformer":
            vec1 = self._get_text_embedding(text1)
            vec2 = self._get_text_embedding(text2)
            return self._cosine_similarity_vectors(vec1, vec2)
        
        if kind == "tfidf" and _check_sklearn():
            from sklearn.metrics.pairwise import cosine_similarity
            # Fit once on first call, reuse for subsequent comparisons
            if not hasattr(model, "vocabulary_"):
                model.fit([text1, text2])
            vectors = model.transform([text1, text2])
            return float(cosine_similarity(vectors[0], vectors[1])[0][0])
        
        return 0.0
    
    def deduplicate_messages(self, messages: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """
        Semantic deduplication using cosine similarity (Sprint 3).
        
        Uses SentenceTransformer if available, TF-IDF fallback.
        Context Guard: last 4 messages are always preserved.
        """
        if len(messages) <= 6:
            return messages
        
        # Context Guard: últimos 4 protegidos
        preserved = messages[-4:]
        to_process = messages[:-4]
        
        if not to_process:
            return messages
        
        unique = [to_process[0]]
        removed_count = 0
        
        for current in to_process[1:]:
            role = current.get("role", "")
            content = current.get("content", "")
            is_duplicate = False
            
            for existing in unique:
                if role == existing.get("role"):
                    sim = self.semantic_similarity(content, existing.get("content", ""))
                    if sim >= threshold:
                        is_duplicate = True
                        removed_count += 1
                        break
            
            if not is_duplicate:
                unique.append(current)
        
        self.messages_deduplicated += removed_count
        return unique + preserved
    
    # ==================== v0.6: SEMANTIC ADAPTATIVE ====================
    
    def get_adaptive_threshold(self, usage_percent: float) -> float:
        """Retorna umbral de similitud semántica según nivel de carga."""
        if usage_percent < 50:
            return 1.0       # No dedup
        elif usage_percent < 85:
            return 0.85      # Semantic default (Sprint 3)
        else:
            return 0.65      # Aggressive semantic
    
    def deduplicate_adaptive(self, messages: List[Dict], usage_percent: float) -> List[Dict]:
        """Deduplicación semántica adaptativa según carga."""
        threshold = self.get_adaptive_threshold(usage_percent)
        
        if threshold >= 1.0:
            return messages
        
        return self.deduplicate_messages(messages, threshold=threshold)
    
    # ==================== v0.6: SEMANTIC TRUNCATE ====================
    
    def semantic_truncate(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """Semantic Truncate con Mid-Truncation inteligente."""
        if not messages:
            return messages
        
        current_tokens = self._messages_tokens(messages)
        if current_tokens <= max_tokens:
            return messages
        
        # New: simple per-message truncation (no complex blocks)
        result = []
        remaining = max_tokens
        
        for msg in messages:
            tokens = self.count_tokens(msg.get("content", ""))
            
            if tokens <= remaining:
                result.append(msg)
                remaining -= tokens
            elif remaining > 20:
                # Mid-Truncation: inicio + [...] + fin
                content = msg.get("content", "")
                chars = remaining * 4
                half = chars // 2
                
                # Always use mid-truncation if there's enough content
                new_content = content[:half] + "[...SEMANTIC TRUNCATED...]" + content[-half:]
                result.append({"role": msg.get("role", "assistant"), "content": new_content})
                remaining = 0
                break
            else:
                # Simple truncate
                content = msg.get("content", "")
                chars = remaining * 4
                result.append({"role": msg.get("role", "assistant"), "content": content[:chars]})
                break
        
        return result