import os
import json
import queue
import re
import logging
import asyncio
import concurrent.futures
import threading
from datetime import datetime

from src.logger import AppLogger
from src.exceptions import MemoryStorageError

logger = logging.getLogger(__name__)
app_logger = AppLogger(__name__)

from src.skills import SkillManager
from src.archive import ArchiveManager
import hashlib

import math
import string

STOP_WORDS = frozenset({
    "the", "is", "and", "or", "of", "to", "in", "for", "on", "at",
    "by", "an", "as", "it", "be", "do", "if", "no", "not", "are",
    "was", "were", "been", "has", "have", "had", "this", "that",
    "with", "from", "will", "can", "should", "would", "could",
    "el", "la", "los", "las", "de", "en", "un", "una", "por",
    "para", "con", "sin", "sobre", "entre", "cuando", "donde"
})

class TFIDFContextRanker:
    """
    Ranker TF-IDF nativo de cero dependencias para búsquedas semánticas.
    """
    def __init__(self, stopwords=None):
        if stopwords is None:
            self.stopwords = STOP_WORDS
        else:
            self.stopwords = set(stopwords)

    def _tokenize(self, text: str) -> list:
        if not text:
            return []
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = text.split()
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def rank(self, query: str, documents: list, top_k: int = 5) -> list:
        """
        Rankea documentos según similitud coseno con la query usando pesos TF-IDF.
        Cada documento espera: [{"id": x, "content": "..."}]
        Retorna los documentos ordenados con una clave adicional "semantic_score".
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or not documents:
            return []

        total_docs = len(documents)
        doc_tfs = []  
        word_df = {}  

        for doc in documents:
            content = doc.get("content", "")
            tokens = self._tokenize(content)
            
            tf = {}
            if tokens:
                for token in tokens:
                    tf[token] = tf.get(token, 0) + 1
                for token in tf:
                    tf[token] = tf[token] / len(tokens)
                    word_df[token] = word_df.get(token, 0) + 1
            doc_tfs.append(tf)

        query_idf = {}
        for token in set(query_tokens):
            df = word_df.get(token, 0)
            query_idf[token] = math.log(1 + (total_docs / (1 + df)))

        query_tf = {}
        for token in query_tokens:
            query_tf[token] = query_tf.get(token, 0) + 1
        for token in query_tf:
            query_tf[token] = (query_tf[token] / len(query_tokens)) * query_idf[token]

        ranked_docs = []
        for idx, doc in enumerate(documents):
            tf = doc_tfs[idx]
            doc_tfidf = {}
            for token in tf:
                if token in query_idf:
                    doc_tfidf[token] = tf[token] * query_idf[token]

            dot_product = 0.0
            doc_magnitude_sq = 0.0
            query_magnitude_sq = sum(val * val for val in query_tf.values())

            for token, q_val in query_tf.items():
                d_val = doc_tfidf.get(token, 0.0)
                dot_product += q_val * d_val

            for val in doc_tfidf.values():
                doc_magnitude_sq += val * val

            doc_magnitude = math.sqrt(doc_magnitude_sq)
            query_magnitude = math.sqrt(query_magnitude_sq)

            score = 0.0
            if doc_magnitude > 0 and query_magnitude > 0:
                score = dot_product / (doc_magnitude * query_magnitude)

            doc_copy = dict(doc)
            doc_copy["semantic_score"] = score
            ranked_docs.append(doc_copy)

        ranked_docs.sort(key=lambda x: x["semantic_score"], reverse=True)
        return ranked_docs[:top_k]

_default_ranker = TFIDFContextRanker()

class MemoryManager:
    # Default: keep max 5000 log lines (rotate older)
    MAX_LOG_LINES = 5000
    
    def __init__(self, memory_path, workspace_path=None, event_loop=None):
        self.memory_path = memory_path
        self.logs_path = os.path.join(memory_path, "logs.jsonl")
        self.patterns_path = os.path.join(memory_path, "patterns.md")
        self.engrams_path = os.path.join(memory_path, "engrams")
        self.skills_dir = os.path.join(memory_path, "skills")
        self.usage_stats_path = os.path.join(memory_path, "usage_stats.json")
        self.workspace_path = workspace_path
        self.last_retrieved_topics = []
        
        # Store the orchestrator's event loop for thread-safe async calls
        self._event_loop = event_loop
        self._log_lock = asyncio.Lock()

        # Background worker for fire-and-forget async operations
        self._async_queue: queue.Queue = queue.Queue(maxsize=100)
        self._worker_started = False
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        
        if not os.path.exists(self.engrams_path):
            os.makedirs(self.engrams_path)
            
        if not os.path.exists(self.skills_dir):
            os.makedirs(self.skills_dir)
            
        self.skills = SkillManager(self.skills_dir)
        
        # Cold Path Initialization
        self.archive = ArchiveManager(os.path.join(memory_path, "cold_archive.db"))

    # ── Background Worker for Async Operations ─────────────────────────

    def _ensure_worker(self):
        """Start the background worker thread on first call."""
        with self._worker_lock:
            if self._worker_started:
                return
            self._worker_started = True

        def _worker_loop():
            while True:
                coro = self._async_queue.get()
                if coro is None:
                    break  # sentinel — shutdown
                try:
                    asyncio.run(coro)
                except Exception:
                    app_logger.exception("[Memory] Worker: coroutine failed")

        t = threading.Thread(target=_worker_loop, daemon=True)
        t.start()
        self._worker_thread = t
        app_logger.debug("[Memory] Background worker started")

    def _shutdown_worker(self):
        """Signal the worker to shut down gracefully."""
        if self._worker_started:
            self._async_queue.put(None)
            app_logger.debug("[Memory] Background worker shutdown signaled")

    def close(self):
        """Clean up resources — shutdown background worker."""
        self._shutdown_worker()
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
            app_logger.debug("[Memory] Background worker thread joined")

    def _run_async(self, coro):
        """
        Run async coroutine synchronously using the orchestrator's event loop.
        Uses run_coroutine_threadsafe to avoid creating new loops (fixes leak).
        Falls back to asyncio.run() if no loop is available.
        
        DEADLOCK GUARD: If called from the event loop thread itself, 
        future.result() would block the loop permanently. We detect this 
        and fire-and-forget the coroutine, returning None. Callers must 
        tolerate None returns for background operations (logging, eviction).
        """
        # DEADLOCK DETECTION: If we're ON the event loop thread, 
        # future.result() would block forever waiting for itself
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
            
        # Case 1: We have a reference to the orchestrator's loop
        if self._event_loop and self._event_loop.is_running():
            if running_loop is self._event_loop:
                # DEADLOCK DETECTED: we're on the event loop thread.
                # Enqueue the coroutine for the background worker.
                self._ensure_worker()
                try:
                    self._async_queue.put_nowait(coro)
                except queue.Full:
                    app_logger.warning(
                        "[Memory] _run_async: worker queue full — "
                        "dropping background operation"
                    )
                return None

            # Case 1b: Not on event loop thread — safe to use future.result()
            future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
            try:
                return future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                future.cancel()
                app_logger.warning("[Memory] _run_async timed out after 30s")
                return None
            except Exception as e:
                app_logger.error(f"[Memory] _run_async failed: {e}")
                raise MemoryStorageError(f"Async operation failed: {e}") from e

        # Case 2: No stored loop — try get_running_loop, then fallback
        if running_loop is not None:
            # DEADLOCK DETECTED: we're on the event loop thread.
            # Enqueue the coroutine for the background worker.
            self._ensure_worker()
            try:
                self._async_queue.put_nowait(coro)
            except queue.Full:
                app_logger.warning(
                    "[Memory] _run_async: worker queue full — "
                    "dropping background operation"
                )
            return None

        # Case 3: No running loop — use ThreadPoolExecutor to avoid
        # "cannot call asyncio.run from a running event loop" errors
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            try:
                return future.result(timeout=30)
            except concurrent.futures.TimeoutError:
                future.cancel()
                app_logger.warning("[Memory] _run_async timed out after 30s (thread pool)")
                return None

    # --- Workspace helpers ---

    def list_workspace_files(self):
        if self.workspace_path and os.path.exists(self.workspace_path):
            return os.listdir(self.workspace_path)
        return []

    def count_workspace_files(self):
        return len(self.list_workspace_files())

    def count_engrams(self):
        if os.path.exists(self.engrams_path):
            return len(os.listdir(self.engrams_path))
        return 0

    # --- Logging ---

    async def log_experience(self, task, result, success, score=None, votes=None):
        from src.logger import get_request_id
        entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": get_request_id(),
            "task": task,
            "result": result[:2000],
            "success": success,
            "score": score,
            "votes": votes
        }
        
        async with self._log_lock:
            # Rotate logs if exceeding limit — count lines without loading all
            if os.path.exists(self.logs_path):
                with open(self.logs_path, "r") as f:
                    line_count = sum(1 for _ in f)
                if line_count >= self.MAX_LOG_LINES:
                    # Read only the second half for rotation
                    with open(self.logs_path, "r") as f:
                        lines = f.readlines()
                    tmp_path = self.logs_path + ".tmp"
                    with open(tmp_path, "w") as f:
                        f.writelines(lines[len(lines) // 2:])
                    os.replace(tmp_path, self.logs_path)
                    logger.info(f"[Memory] Rotated logs, kept {len(lines) // 2} entries")
            
            with open(self.logs_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            
        # Also log to Cold Path for permanent history
        await self.archive.log_to_history(entry)

    def get_recent_logs(self, limit=10):
        if not os.path.exists(self.logs_path):
            return []
        with open(self.logs_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            results = []
            for line in lines[-limit:]:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"[Memory] Skipping malformed log line: {line[:80]}")
            return results

    # --- Patterns ---

    def save_pattern(self, content):
        tmp_path = self.patterns_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(f"Patrones y Lecciones Aprendidas\nActualizado: {datetime.now()}\n\n{content}")
        os.replace(tmp_path, self.patterns_path)

    def load_patterns(self):
        if os.path.exists(self.patterns_path):
            with open(self.patterns_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def forget(self):
        if os.path.exists(self.patterns_path):
            os.remove(self.patterns_path)
        if os.path.exists(self.logs_path):
            os.remove(self.logs_path)

    # --- Engrams ---

    def save_engram(self, topic, content):
        """Guarda el engram directamente en la base de datos unificada SQLite."""
        clean_topic = re.sub(r'[^a-zA-Z0-9_-]', '_', topic.lower())
        
        try:
            # Guardar en SQLite (Hot & Cold unificado)
            self._run_async(self.archive.archive_engram(topic, content))
        except MemoryStorageError as e:
            app_logger.error(f"Failed to save engram '{topic}': {e}")
            return f"Error al guardar Engram '{topic}'."
        
        # Phase 3: Auto-extract entities para el Knowledge Graph
        self._auto_extract_entities(clean_topic, content)
        
        return f"Engram '{topic}' guardado en SQLite y extraído al Knowledge Graph."

    def _auto_extract_entities(self, topic, content):
        """
        Auto-extracts entities from saved observation and stores them.
        Creates entity entries in the knowledge graph.
        
        Args:
            topic: Topic/identifier for the observation (used as observation_id)
            content: Content to extract entities from
        """
        from collections import Counter

        # Usar el topic directamente como observation_id para consistencia
        obs_id = topic
        
        words = re.findall(r'[a-zA-Z0-9]{4,}', content.lower())
        filtered = [w for w in words if w not in STOP_WORDS]
        
        # Top 5 palabras más frecuentes como entidades
        word_counts = Counter(filtered)
        top_entities = word_counts.most_common(5)
        
        # Extraer patrones: TODO/NOTA/FIXME/BUG
        patterns = {
            "TODO": r'(?i)(?:TODO|FIXME|TASK):\s*(.+)',
            "BUG": r'(?i)(?:BUG|ERROR|ISSUE):\s*(.+)',
            "DECISION": r'(?i)(?:DECISION|CHOOSE|PICKED):\s*(.+)',
            "PATTERN": r'(?i)(?:PATTERN|CONVENTION):\s*(.+)'
        }
        
        entity_type_map = {
            "TODO": "task",
            "BUG": "bugfix", 
            "DECISION": "decision",
            "PATTERN": "pattern"
        }
        
        try:
            # Guardar entidades en el archivo de archive
            for entity_val, count in top_entities:
                if len(entity_val) >= 4:  # Ignorar entidades muy cortas
                    success = self._run_async(self.archive.add_entity(obs_id, "keyword", entity_val))
            
            # Buscar patrones especiales
            for pat_name, pat_regex in patterns.items():
                matches = re.findall(pat_regex, content)
                for match in matches:
                    if len(match.strip()) >= 4:
                        entity_type = entity_type_map.get(pat_name, "mention")
                        self._run_async(self.archive.add_entity(obs_id, entity_type, match.strip()[:200]))
                        
        except Exception as e:
            app_logger.exception(f"[Memory] Error en auto-extracción para '{topic}'")

    def update_usage_stats(self, topic, is_success):
        stats = {}
        if os.path.exists(self.usage_stats_path):
            try:
                with open(self.usage_stats_path, "r") as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                app_logger.warning(f"[Memory] Error reading usage stats: {e}")
        
        if topic not in stats:
            stats[topic] = {"usos": 0, "fallos": 0, "ultimo_uso": ""}
            
        stats[topic]["usos"] += 1
        if not is_success:
            stats[topic]["fallos"] += 1
        stats[topic]["ultimo_uso"] = datetime.now().isoformat()
        
        tmp_path = self.usage_stats_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(stats, f, indent=4)
        os.replace(tmp_path, self.usage_stats_path)

    async def _purge_old_engrams_from_db(self, topics_to_purge: list):
        """Elimina engrams de la base de datos SQLite por topic."""
        if not topics_to_purge:
            return
        
        conn = await self.archive._get_conn()
        await conn.executemany(
            "DELETE FROM engram_archive WHERE topic = ?",
            [(t,) for t in topics_to_purge]
        )
        await conn.commit()

    async def decay_old_engrams(self, max_fallos=3):
        """Elimina engrams obsoletos de SQLite y usage_stats.json."""
        if not os.path.exists(self.usage_stats_path):
            return 0
        try:
            with open(self.usage_stats_path, "r") as f:
                stats = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            app_logger.warning(f"[Memory] Error reading stats for decay: {e}")
            return 0

        topics_to_purge = []
        for topic, stat in list(stats.items()):
            should_delete = False
            if stat.get("fallos", 0) >= max_fallos:
                should_delete = True

            ultimo = stat.get("ultimo_uso")
            if ultimo:
                delta = datetime.now() - datetime.fromisoformat(ultimo)
                if delta.days > 30:
                    should_delete = True

            if should_delete:
                topics_to_purge.append(topic)

        if not topics_to_purge:
            return 0

        # Purge from SQLite first; only update stats if DB operation succeeds
        try:
            await self._purge_old_engrams_from_db(topics_to_purge)
        except Exception as e:
            app_logger.error(f"[Memory] Error purging engrams from DB: {e}")
            return 0

        deleted_count = 0
        for topic in topics_to_purge:
            if topic in stats:
                del stats[topic]
                deleted_count += 1

        try:
            with open(self.usage_stats_path, "w") as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            app_logger.error(f"[Memory] Error writing usage stats: {e}")
            return 0

        return deleted_count

    def search_engrams(self, query):
        """
        Busca engrams usando el backend unificado FTS5 (SQLite) + TF-IDF semántico.
        Incluye fallback a engrams recientes cuando FTS5 no encuentra resultados.
        Returns:
            tuple: (formatted_results: str, retrieved_topics: list)
        """
        retrieved_topics = []
        if not query.strip():
            return ("Consulta vacia.", retrieved_topics)
            
        # 1. Obtener candidatos de SQLite via FTS5
        candidates = self._run_async(self.archive.search_archive(query, limit=20))
        
        # 2. FALLBACK: Si FTS5 no encontró nada, traer los 5 más recientes
        if not candidates:
            try:
                import asyncio
                async def _recent():
                    async with self.archive._lock:
                        conn = await self.archive._get_conn()
                        async with conn.execute(
                            """SELECT id, topic, content, timestamp, score, importance_score
                               FROM engram_archive
                               ORDER BY timestamp DESC LIMIT 5"""
                        ) as cursor:
                            rows = await cursor.fetchall()
                        return [
                            {"id": r[0], "topic": r[1], "content": r[2],
                             "timestamp": r[3], "score": r[4] or 0.0}
                            for r in rows
                        ]
                candidates = self._run_async(_recent())
            except Exception:
                pass
        
        if not candidates:
            return ("No hay engrams almacenados aún.", retrieved_topics)

        # 3. Utilizar el ranker TF-IDF para ordenar semánticamente los candidatos
        docs = []
        for c in candidates:
            docs.append({
                "original": c,
                "content": f"{c.get('topic', '')} {c.get('content', '')}"
            })
            
        ranker = _default_ranker
        ranked = ranker.rank(query, docs, top_k=5)
        
        # 4. Si el TF-IDF no encontró nada relevante, usar los candidatos raw
        if not ranked or (ranked and ranked[0].get("semantic_score", 0) < 0.01):
            # Sin re-ranking: tomar los primeros 5 candidatos tal cual
            ranked = [{"original": c, "semantic_score": 0.5} for c in candidates[:5]]
        
        # 5. U-Shape Ordering del top 5
        final_results = []
        for i, doc in enumerate(ranked):
            c = doc["original"]
            topic = c['topic']
            retrieved_topics.append(topic)
            content = c['content']
            snippet = content[:1000] + ("..." if len(content) > 1000 else "")
            
            semantic_weight = doc["semantic_score"]
            formatted = f"--- Engram: {topic} (Relevancia Semántica: {semantic_weight:.2f}) ---\n{snippet}"
            
            if i % 2 == 0:
                final_results.append(formatted)
            else:
                final_results.insert(0, formatted)

        return ("\n\n".join(final_results), retrieved_topics)

    def cleanup_engrams(self):
        count = 0
        for f in os.listdir(self.engrams_path):
            full_path = os.path.join(self.engrams_path, f)
            try:
                if os.path.isfile(full_path):
                    os.remove(full_path)
                    count += 1
            except OSError as e:
                app_logger.warning(f"[Memory] Error removing {f}: {e}")
        return count

    def retrieve_omni_context(self, query: str) -> str:
        """
        Scans Skills, Engrams, and Workspace .md files for relevant context.
        Returns a formatted string to inject as Latent Memory.
        Now includes project-context boosting and recent memory fallback.
        """
        omni_parts = []
        
        # 1. Skills (Behavioral)
        relevant_skills = self.skills.retrieve_relevant(query, top_k=3)
        skills_formatted = self.skills.format_for_prompt(relevant_skills)
        if skills_formatted.strip():
            omni_parts.append(f"### SKILLS ACTIVADAS:\n{skills_formatted}")

        # 2. Engrams (Factual) — improved with fallback
        engrams_raw, retrieved_topics = self.search_engrams(query)
        if "No se encontraron engrams" not in engrams_raw and "Consulta vacia" not in engrams_raw and "No hay engrams" not in engrams_raw:
            omni_parts.append(f"### ENGRAMS (Conocimiento Previo):\n{engrams_raw}")

        # 2.5. Recent memories (always include last 3 regardless of query match)
        # This ensures the agent has context about recent work
        try:
            import asyncio
            async def _recent_for_context():
                async with self.archive._lock:
                    conn = await self.archive._get_conn()
                    async with conn.execute(
                        """SELECT topic, content, timestamp FROM engram_archive
                           ORDER BY timestamp DESC LIMIT 3"""
                    ) as cursor:
                        return await cursor.fetchall()
            recent = self._run_async(_recent_for_context())
            if recent:
                recent_formatted = "\n".join([
                    f"- {r[0]} ({r[2][:10]}): {r[1][:200]}..."
                    for r in recent
                ])
                # Only add if not already covered by search results
                if not retrieved_topics or len(retrieved_topics) < 2:
                    omni_parts.append(f"### MEMORIAS RECIENTES:\n{recent_formatted}")
        except Exception:
            pass

        # 3. Workspace Files (Recursive & Semantic TF-IDF)
        if self.workspace_path and os.path.exists(self.workspace_path):
            valid_extensions = {".md", ".py", ".go", ".js", ".ts", ".json", ".rs", ".sh"}
            ignored_dirs = {".git", "__pycache__", "venv", "node_modules", "build", "dist", ".engram", "extras"}
            
            workspace_docs = []
            for root, dirs, files in os.walk(self.workspace_path):
                dirs[:] = [d for d in dirs if d not in ignored_dirs]
                
                rel_path = os.path.relpath(root, self.workspace_path)
                if rel_path != "." and len(rel_path.split(os.sep)) > 4:
                    continue
                    
                for file in files:
                    _, ext = os.path.splitext(file)
                    if ext.lower() in valid_extensions:
                        filepath = os.path.join(root, file)
                        try:
                            if os.path.getsize(filepath) < 102400: # Max 100KB to keep memory usage low
                                with open(filepath, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    if content.strip():
                                        workspace_docs.append({
                                            "filename": os.path.relpath(filepath, self.workspace_path),
                                            "content": content
                                        })
                        except Exception as e:
                            app_logger.debug(f"Error reading workspace file {filepath}: {e}")
                            pass
            
            if workspace_docs:
                ranker = _default_ranker
                ranked_workspace = ranker.rank(query, workspace_docs, top_k=3)
                
                md_output = []
                for doc in ranked_workspace:
                    if doc["semantic_score"] > 0.05: # Minimum relevance threshold
                        fname = doc["filename"]
                        score = doc["semantic_score"]
                        content = doc["content"]
                        snippet = content[:800] + "..." if len(content) > 800 else content
                        md_output.append(f"- Archivo '{fname}' (Similitud Semántica: {score:.2f}):\n{snippet}")
                
                if md_output:
                    omni_parts.append(f"### ARCHIVOS LOCALES RELEVANTES:\n" + "\n".join(md_output))

        # 4. Cold Archive (Long-term Factual)
        archive_results = self._run_async(self.archive.search_archive(query, limit=3))
        if archive_results:
            archive_fmt = "\n".join([f"- {r['topic']} ({r['timestamp']}): {r['content'][:400]}..." for r in archive_results])
            omni_parts.append(f"### ARCHIVO HISTÓRICO (Cold Path):\n{archive_fmt}")

        if not omni_parts:
            return "No hay contexto latente relevante para esta consulta. Este es un proyecto nuevo o sin memorias previas."
            
        return "\n\n".join(omni_parts)

    # --- Knowledge Graph: Entity Extraction ---

    # Patrones regex para entidades
    PATTERNS = {
        "file": r'(?:src/[^\s/]+(?:\.[pf][ly][oa]?)|[^\s/]+\.(?:py|js|ts|md|go|rs|json|yaml|yml|toml)(?:\s|$|/))',
        "url": r'https?://[^\s<>"{}|\\\^`\[\]]+',
        "package": r'(?:npm|pip|go|cargo|poetry|bundler) install ([^\s]+)'
    }

    def _extract_entities(self, content: str) -> dict:
        """
        Extrae entidades de cualquier tipo usando regex.
        Retorna dict con listas por tipo: {"file": [...], "url": [...], "package": [...]}
        """
        entities = {"file": [], "url": [], "package": []}
        
        # Archivos: busca paths que terminen en extensiones comunes
        file_pattern = r'([a-zA-Z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|md|go|rs|json|yaml|yml|toml|sh|html|css|sql))'
        for match in re.finditer(file_pattern, content):
            path = match.group(1).strip()
            # Filtrar paths típicos que no son archivos
            if not path.startswith('/usr/') and not path.startswith('~'):
                entities["file"].append(path)
        
        # URLs
        url_pattern = r'(https?://[^\s<>"{}|\\\^`\[\]]+)'
        for match in re.finditer(url_pattern, content):
            url = match.group(1).strip()
            # Limpiar trailing punctuación
            url = re.sub(r'[.,;:)>\]}]$', '', url)
            if url not in entities["url"]:
                entities["url"].append(url)
        
# Paquetes: npm install, pip install, go get, cargo install
        package_pattern = r'(?:npm|pip|go|cargo|poetry|bundler) (?:install|get)\s+([^\s]+)'
        for match in re.finditer(package_pattern, content):
            pkg = match.group(1).strip()
            # Limpiar trailing puntuación o version - pero preservar slashes para paths
            pkg = re.sub(r'[;,)<>#].*$', '', pkg)
            entities["package"].append(pkg)
        
        # Eliminar duplicados preservando orden
        for etype in entities:
            seen = set()
            unique = []
            for item in entities[etype]:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            entities[etype] = unique[:20]  # Limitar a 20 por tipo
        
        logger.debug(f"[Memory] Extraídas entidades: {sum(len(v) for v in entities.values())} total")
        return entities

    def create_entity(self, observation_id: int, entity_type: str, value: str) -> dict:
        """
        Crea una entidad en el Knowledge Graph.
        Args:
            observation_id: ID de la observación/engram origen
            entity_type: "file", "url", "package"
            value: valor de la entidad
        Returns:
            {"success": bool, "entity_id": int or None, "message": str}
        """
        valid_types = {"file", "url", "package"}
        if entity_type not in valid_types:
            return {"success": False, "entity_id": None, "message": f"Tipo inválido. Usar: {valid_types}"}
        
        if not value or len(value.strip()) < 2:
            return {"success": False, "entity_id": None, "message": "Value demasiado corto"}
        
        try:
            entity_id = self._run_async(self.archive.add_entity(observation_id, entity_type, value.strip()))
            app_logger.info(f"[Memory] Entidad creada: {entity_type}={value[:50]} (id={entity_id})")
            return {"success": True, "entity_id": entity_id, "message": f"Entidad {entity_type} creada"}
        except MemoryStorageError:
            raise
        except Exception as e:
            app_logger.exception(f"[Memory] Error create_entity: {e}")
            return {"success": False, "entity_id": None, "message": str(e)}

    def create_edge(self, source_id: int, target_id: int, relation_type: str) -> dict:
        """
        Crea una relación (edge) entre dos entidades.
        Args:
            source_id: ID de la entidad origen
            target_id: ID de la entidad destino
            relation_type: tipo de relación (e.g., "imports", "references", "depends_on", "calls")
        Returns:
            {"success": bool, "edge_id": int or None, "message": str}
        """
        from src.archive import ArchiveManager
        
        if source_id == target_id:
            return {"success": False, "edge_id": None, "message": "source_id y target_id no pueden ser iguales"}
        
        if relation_type not in ArchiveManager.VALID_RELATION_TYPES:
            return {"success": False, "edge_id": None, "message": f"Relación inválida. Usar: {ArchiveManager.VALID_RELATION_TYPES}"}
        
        try:
            edge_id = self._run_async(self.archive.add_edge(source_id, target_id, relation_type))
            app_logger.info(f"[Memory] Edge creado: {source_id} --[{relation_type}]--> {target_id} (id={edge_id})")
            return {"success": True, "edge_id": edge_id, "message": f"Edge {relation_type} creado"}
        except MemoryStorageError:
            raise
        except Exception as e:
            app_logger.exception(f"[Memory] Error create_edge: {e}")
            return {"success": False, "edge_id": None, "message": str(e)}
