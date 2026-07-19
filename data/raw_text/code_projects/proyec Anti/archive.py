import aiosqlite
import json
import os
import math
import re
import asyncio
from datetime import datetime

from src.logger import AppLogger
from src.exceptions import MemoryStorageError

app_logger = AppLogger(__name__)

# Phase 2.2: Core Scoring Logic
# typeBonus mapping: bonus applied based on observation type
TYPE_BONUS = {
    "bugfix": 0.5,
    "decision": 0.5,
    "discovery": 0.3,
    "pattern": 0.1,
    # Default for unknown types
    None: 0.0,
    "": 0.0,
}

# Phase 2.3: Access Tracking
ACCESS_BONUS = 0.2


class ArchiveManager:
    """
    Gestiona el 'Cold Path' de la memoria de Anti.
    Almacena engrams antiguos y logs extensos en una base de datos SQLite
    para mantener el 'Hot Path' (JSONs) liviano y rápido.
    """

    VALID_RELATION_TYPES = frozenset({
        "references", "relates_to", "follows", "supersedes", "contradicts"
    })

    def __init__(self, db_path):
        self.db_path = db_path
        self._conn = None
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()

    async def _get_conn(self):
        """Returns a persistent connection with optimized PRAGMAs.
        Uses double-check locking to prevent race conditions."""
        if self._conn is not None:
            return self._conn
        async with self._init_lock:
            # Double-check after acquiring lock
            if self._conn is not None:
                return self._conn
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("PRAGMA journal_mode = WAL")
            await self._conn.execute("PRAGMA foreign_keys = ON")
            await self._conn.execute("PRAGMA synchronous = NORMAL")
            await self._conn.execute("PRAGMA busy_timeout = 5000")
            await self._init_db()
        return self._conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False  # Don't suppress exceptions

    async def close(self):
        """Cierra la conexión SQLite de forma segura. Idempotente."""
        if self._conn is not None:
            try:
                try:
                    await self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as e:
                    app_logger.debug(f"[Archive] Checkpoint failed during close: {e}")
                await self._conn.close()
            except Exception as e:
                app_logger.debug(f"[Archive] Error closing connection: {e}")
            finally:
                self._conn = None

    def __del__(self):
        self._conn = None

    async def _init_db(self):
        """Inicializa las tablas si no existen."""
        conn = await self._get_conn()
        # aiosqlite connections are used to execute directly or via cursor
        # In aiosqlite, conn.execute() returns a cursor
        
        # Tabla de Engrams Archivados
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS engram_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance_score REAL DEFAULT 0,
                tags TEXT,
                score REAL DEFAULT 1.0,
                last_accessed_at TIMESTAMP
            )
        ''')
        # Tabla de Logs Históricos
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS log_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task TEXT NOT NULL,
                result TEXT,
                success INTEGER,
                score REAL
            )
        ''')
        # Tabla de Entidades (Knowledge Graph)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT,
                entity_type TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        # Tabla de Relaciones (Edges)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            )
        ''')
        # Índices para mejor rendimiento
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_entities_observation_id ON entities(observation_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_source_id ON edges(source_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_edges_target_id ON edges(target_id)')
        
        # Tabla de Prompts de Proyecto (Capa 2)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS project_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                core_prompt TEXT NOT NULL,
                important_directives TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        # Índices de Project Prompts
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_project_prompts_name ON project_prompts(project_name)')
        
        # Tabla de Lecciones Aprendidas (Capa 2)
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS learned_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                project TEXT NOT NULL,
                category TEXT NOT NULL,
                pure_lesson TEXT NOT NULL,
                token_cost INTEGER DEFAULT 0
            )
        ''')
        # Índices de Learned Lessons (Sub-milisegundo query speed)
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_learned_lessons_project ON learned_lessons(project)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_learned_lessons_category ON learned_lessons(category)')
        
        # FTS5 (Full Text Search) para Engrams
        await conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS engram_fts USING fts5(
                topic,
                content,
                content='engram_archive',
                content_rowid='id'
            )
        ''')
        # Triggers de sincronización FTS5
        await conn.execute('''
            CREATE TRIGGER IF NOT EXISTS engram_ai AFTER INSERT ON engram_archive BEGIN
                INSERT INTO engram_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content);
            END;
        ''')
        await conn.execute('''
            CREATE TRIGGER IF NOT EXISTS engram_ad AFTER DELETE ON engram_archive BEGIN
                INSERT INTO engram_fts(engram_fts, rowid, topic, content) VALUES('delete', old.id, old.topic, old.content);
            END;
        ''')
        await conn.execute('''
            CREATE TRIGGER IF NOT EXISTS engram_au AFTER UPDATE OF topic, content ON engram_archive BEGIN
                INSERT INTO engram_fts(engram_fts, rowid, topic, content) VALUES('delete', old.id, old.topic, old.content);
                INSERT INTO engram_fts(rowid, topic, content) VALUES (new.id, new.topic, new.content);
            END;
        ''')
        # Rebuild FTS index to ensure consistency
        await conn.execute("INSERT INTO engram_fts(engram_fts) VALUES('rebuild')")
        await conn.commit()

    async def archive_engram(self, topic, content, importance=1.0, tags=""):
        """Mueve un engram al archivo frío."""
        async with self._lock:
            try:
                conn = await self._get_conn()
                await conn.execute(
                    "INSERT INTO engram_archive (topic, content, timestamp, importance_score, tags) VALUES (?, ?, ?, ?, ?)",
                    (topic, content, datetime.now().isoformat(), importance, tags)
                )
                await conn.commit()
                return True
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error archivando engram '{topic}'")
                raise MemoryStorageError(f"Failed to archive engram '{topic}': {e}") from e

    async def search_archive(self, query, limit=5):
        """
        Busca engrams en el archivo usando FTS5 para máxima velocidad semántica/palabras clave.
        Phase 2.3: Actualiza last_accessed_at, aplica accessBonus y recalcula score.
        """
        async with self._lock:
            try:
                conn = await self._get_conn()
                
                # Preprocesar query para FTS5
                clean_query = re.sub(r'[^\w\s]', '', query).strip()
                if not clean_query:
                    return []
                # FTS5 query: búsqueda OR de palabras clave para máxima recuperación
                fts_query = " OR ".join(f'"{word}"' for word in clean_query.split())
                
                async with conn.execute(
                    """
                    SELECT e.id, e.topic, e.content, e.timestamp, e.score, e.importance_score
                    FROM engram_archive e
                    JOIN engram_fts fts ON e.id = fts.rowid
                    WHERE engram_fts MATCH ?
                    ORDER BY bm25(engram_fts) LIMIT ?
                    """,
                    (fts_query, limit)
                ) as cursor:
                    results = await cursor.fetchall()
                
                if not results:
                    return []

                processed_results = []
                updates = []
                now = datetime.now().isoformat()

                for r in results:
                    engram_id = r[0]
                    # 3.1: Update last_accessed_at
                    # 3.2: Get current score and apply accessBonus
                    current_score = r[4] if r[4] is not None else 1.0
                    new_score = current_score + ACCESS_BONUS
                    # 3.3: Recalculate full score
                    new_importance = r[5] if r[5] is not None else 0.0
                    full_score = min(5.0, new_importance + new_score)
                    
                    updates.append((now, full_score, engram_id))
                    
                    processed_results.append({
                        "id": engram_id,
                        "topic": r[1],
                        "content": r[2],
                        "timestamp": r[3],
                        "score": full_score
                    })
                
                # Batch update using executemany
                await conn.executemany(
                    "UPDATE engram_archive SET last_accessed_at = ?, score = ? WHERE id = ?",
                    updates
                )
                await conn.commit()
                return processed_results
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error buscando en archivo: '{query}'")
                raise MemoryStorageError(f"Failed to search archive: {e}") from e

    async def log_to_history(self, entry):
        """Guarda un log detallado en el historial de largo plazo."""
        async with self._lock:
            try:
                conn = await self._get_conn()
                await conn.execute(
                    "INSERT INTO log_history (timestamp, task, result, success, score) VALUES (?, ?, ?, ?, ?)",
                    (entry.get('timestamp'), entry.get('task'), entry.get('result'), 
                     1 if entry.get('success') else 0, entry.get('score', 0))
                )
                await conn.commit()
                return True
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error guardando log histórico")
                raise MemoryStorageError(f"Failed to save history log: {e}") from e

    async def get_stats(self):
        """Retorna estadísticas del archivo."""
        try:
            conn = await self._get_conn()
            async with conn.execute("SELECT (SELECT COUNT(*) FROM engram_archive), (SELECT COUNT(*) FROM log_history)") as cursor:
                row = await cursor.fetchone()
            engrams, logs = row[0], row[1]
            return {"archived_engrams": engrams, "historical_logs": logs}
        except Exception as e:
            app_logger.exception("[Archive] Error obteniendo stats")
            raise MemoryStorageError(f"Failed to get archive stats: {e}") from e

    # --- Phase 2.4: Auto-Purge System ---

    async def _get_low_score_observations(self, threshold=1.0):
        """
        Retorna observaciones con score bajo el threshold especificado.
        
        Args:
            threshold: Score máximo para considerar como "bajo" (default: 1.0)
            
        Returns:
            Lista de diccionarios {
                "id": int,
                "topic": str,
                "content": str,
                "importance_score": float,
                "timestamp": str,
                "last_accessed_at": str or None
            }
        """
        try:
            conn = await self._get_conn()
            async with conn.execute(
                """
                SELECT id, topic, content, importance_score, timestamp, last_accessed_at 
                FROM engram_archive 
                WHERE importance_score < ? 
                ORDER BY importance_score ASC, timestamp DESC
                """,
                (threshold,)
            ) as cursor:
                rows = await cursor.fetchall()
                
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "topic": row[1],
                    "content": row[2],
                    "importance_score": row[3],
                    "timestamp": row[4],
                    "last_accessed_at": row[5]
                })
            return results
        except Exception as e:
            app_logger.exception("[Archive] Error en _get_low_score_observations")
            raise MemoryStorageError(f"Failed to get low score observations: {e}") from e

    async def update_observation_score(self, observation_id: int, score_delta: float):
        """
        Actualiza el score de importancia de una observación.
        
        Args:
            observation_id: ID de la observación
            score_delta: Incremento o decremento a aplicar (positivo o negativo)
            
        Returns:
            bool: True si se actualizó correctamente
        """
        async with self._lock:
            try:
                conn = await self._get_conn()
                # Primero verificar que existe
                async with conn.execute("SELECT id FROM engram_archive WHERE id = ?", (observation_id,)) as cursor:
                    if not await cursor.fetchone():
                        app_logger.warning(f"[Archive] Observación no encontrada: {observation_id}")
                        return False
                        
                # Actualizar score + actualizar last_accessed_at
                await conn.execute(
                    """
                    UPDATE engram_archive 
                    SET importance_score = MAX(0, importance_score + ?),
                        last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (score_delta, datetime.now().isoformat(), observation_id)
                )
                await conn.commit()
                return True
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error actualizando score para observación {observation_id}")
                raise MemoryStorageError(f"Failed to update observation score: {e}") from e

    async def purge_observations(self, observation_ids: list) -> int:
        """
        Elimina observaciones del archivo (auto-purge).
        
        Args:
            observation_ids: Lista de IDs a eliminar
            
        Returns:
            int: Número total de registros eliminados (observations + entities + edges)
        """
        if not observation_ids:
            return 0
            
        async with self._lock:
            try:
                conn = await self._get_conn()
                total_deleted = 0
                
                # Batch purge in chunks to avoid SQLITE_MAX_VARIABLE_NUMBER
                for i in range(0, len(observation_ids), 500):
                    chunk = observation_ids[i:i+500]
                    placeholders = ",".join("?" * len(chunk))
                    
                    # 1. Delete edges referencing entities that belong to purged observations
                    cursor = await conn.execute(f"DELETE FROM edges WHERE source_id IN (SELECT id FROM entities WHERE observation_id IN ({placeholders})) OR target_id IN (SELECT id FROM entities WHERE observation_id IN ({placeholders}))", tuple(chunk) * 2)
                    edges_deleted = cursor.rowcount
                    
                    # 2. Delete entities belonging to purged observations
                    cursor = await conn.execute(f"DELETE FROM entities WHERE observation_id IN ({placeholders})", tuple(chunk))
                    entities_deleted = cursor.rowcount
                    
                    # 3. Delete archive records
                    cursor = await conn.execute(
                        f"DELETE FROM engram_archive WHERE id IN ({placeholders})",
                        chunk
                    )
                    obs_deleted = cursor.rowcount
                    
                    total_deleted += (obs_deleted + entities_deleted + edges_deleted)
                
                await conn.commit()
                app_logger.info(f"[Archive] Auto-purged {len(observation_ids)} observations and their associated entities/edges")
                return total_deleted
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception("[Archive] Error en purge")
                raise MemoryStorageError(f"Failed to purge observations: {e}") from e

    async def add_entity(self, observation_id, entity_type: str, value: str):
        """
        Agrega una entidad al Knowledge Graph.
        Acepta observation_id como string (hash) o int.
        Retorna el ID de la entidad o None si falla.
        """
        async with self._lock:
            try:
                conn = await self._get_conn()
                
                # Validate observation_id: must be numeric (int or digit string)
                if not isinstance(observation_id, int) and not (isinstance(observation_id, str) and observation_id.isdigit()):
                    app_logger.warning(f"[Archive] Invalid observation_id: {observation_id}. Must be numeric.")
                    return None

                # Validate that observation_id exists in engram_archive
                async with conn.execute("SELECT id FROM engram_archive WHERE id = ?", (observation_id,)) as cursor:
                    if not await cursor.fetchone():
                        app_logger.warning(f"[Archive] observation_id {observation_id} not found in engram_archive, skipping entity insert")
                        return None
                
                cursor = await conn.execute(
                    "INSERT INTO entities (observation_id, entity_type, value, timestamp) VALUES (?, ?, ?, ?)",
                    (observation_id, entity_type, value, datetime.now().isoformat())
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error agregando entidad (type={entity_type})")
                raise MemoryStorageError(f"Failed to add entity: {e}") from e

    async def add_edge(self, source_id: int, target_id: int, relation_type: str) -> int:
        """
        Agrega una relación (edge) entre dos entidades.
        Retorna el ID del edge o None si falla.
        """
        async with self._lock:
            try:
                conn = await self._get_conn()
                # Validar que ambas entidades existan
                async with conn.execute("SELECT id FROM entities WHERE id IN (?, ?)", (source_id, target_id)) as cursor:
                    rows = await cursor.fetchall()
                
                # For self-loops, rows will have only 1 entry. 
                # We allow self-loops as long as the entity exists.
                if source_id == target_id:
                    if len(rows) != 1:
                        app_logger.warning(f"[Archive] Entity not found: {source_id}")
                        return None
                elif len(rows) != 2:
                    app_logger.warning(f"[Archive] Entities no encontradas: {source_id}, {target_id}")
                    return None
                
                cursor = await conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation_type, timestamp) VALUES (?, ?, ?, ?)",
                    (source_id, target_id, relation_type, datetime.now().isoformat())
                )
                await conn.commit()
                return cursor.lastrowid
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error agregando edge")
                raise MemoryStorageError(f"Failed to add edge: {e}") from e

    async def get_entities_by_type(self, entity_type: str, limit: int = 50) -> list:
        """取得指定类型的实体列表。"""
        try:
            conn = await self._get_conn()
            async with conn.execute(
                "SELECT id, observation_id, value, timestamp FROM entities WHERE entity_type = ? ORDER BY id DESC LIMIT ?",
                (entity_type, limit)
            ) as cursor:
                rows = await cursor.fetchall()
            return [{"id": r[0], "observation_id": r[1], "value": r[2], "timestamp": r[3]} for r in rows]
        except Exception as e:
            app_logger.exception(f"[Archive] Error obteniendo entidades tipo '{entity_type}'")
            raise MemoryStorageError(f"Failed to get entities by type: {e}") from e

    async def get_entity_edges(self, entity_id: int) -> list:
        """Obtiene todas las relaciones de una entidad."""
        try:
            conn = await self._get_conn()
            async with conn.execute(
                "SELECT e.id, e.source_id, e.target_id, e.relation_type, e.timestamp FROM edges e WHERE e.source_id = ? OR e.target_id = ?",
                (entity_id, entity_id)
            ) as cursor:
                rows = await cursor.fetchall()
            return [{"id": r[0], "source_id": r[1], "target_id": r[2], "relation_type": r[3], "timestamp": r[4]} for r in rows]
        except Exception as e:
            app_logger.exception(f"[Archive] Error obteniendo edges para entidad {entity_id}")
            raise MemoryStorageError(f"Failed to get entity edges: {e}") from e

    # --- Knowledge Graph Phase 3: Relations API ---
    
    async def mem_relate(self, source_id, target_id, relation_type):
        """
        Crea una relación entre dos entidades.
        
        Args:
            source_id: ID de la entidad fuente
            target_id: ID de la entidad objetivo
            relation_type: Uno de: references, relates_to, follows, supersedes, contradicts
            
        Returns:
            Tupla (success: bool, message: str)
        """
        if relation_type not in self.VALID_RELATION_TYPES:
            return (False, f"Tipo de relación inválido: {relation_type}. "
                           f"Válidos: {', '.join(self.VALID_RELATION_TYPES)}")
        
        async with self._lock:
            try:
                conn = await self._get_conn()
                # Verificar que ambas entidades existen
                async with conn.execute("SELECT id FROM entities WHERE id = ?", (source_id,)) as cursor:
                    if not await cursor.fetchone():
                        return (False, f"Entidad fuente no encontrada: {source_id}")
                async with conn.execute("SELECT id FROM entities WHERE id = ?", (target_id,)) as cursor:
                    if not await cursor.fetchone():
                        return (False, f"Entidad objetivo no encontrada: {target_id}")
                
                # Insertar la relación
                await conn.execute(
                    "INSERT INTO edges (source_id, target_id, relation_type, timestamp) VALUES (?, ?, ?, ?)",
                    (source_id, target_id, relation_type, datetime.now().isoformat())
                )
                await conn.commit()
                return (True, f"Relación creada: {source_id} --[{relation_type}]--> {target_id}")
            except Exception as e:
                if self._conn:
                    await self._conn.rollback()
                app_logger.exception(f"[Archive] Error en mem_relate")
                raise MemoryStorageError(f"Failed to create relation: {e}") from e


    async def mem_get_relations(self, observation_id, relation_type=None):
        """
        Obtiene relaciones de una observación.
        
        Args:
            observation_id: ID de la observación
            relation_type: Opcional, filtrar por tipo de relación
            
        Returns:
            Lista de diccionarios con {source_id, target_id, relation_type, target_value, source_value, direction}
        """
        try:
            conn = await self._get_conn()
            
            query = """
                SELECT 
                    e.source_id, 
                    e.target_id, 
                    e.relation_type, 
                    ent_target.value, 
                    ent_source.value,
                    CASE WHEN e.source_id = ent.id THEN 'outgoing' ELSE 'incoming' END
                FROM entities ent
                JOIN edges e ON (e.source_id = ent.id OR e.target_id = ent.id)
                JOIN entities ent_target ON e.target_id = ent_target.id
                JOIN entities ent_source ON e.source_id = ent_source.id
                WHERE ent.observation_id = ?
            """
            params = [observation_id]
            if relation_type:
                query += " AND e.relation_type = ?"
                params.append(relation_type)
            
            async with conn.execute(query, params) as cursor:
                rows = await cursor.fetchall()
            
            seen = set()
            results = []
            for r in rows:
                # Deduplicate result edges by (source_id, target_id, relation_type)
                edge_key = (r[0], r[1], r[2])
                if edge_key not in seen:
                    seen.add(edge_key)
                    results.append({
                        "source_id": r[0],
                        "target_id": r[1],
                        "relation_type": r[2],
                        "target_value": r[3],
                        "source_value": r[4],
                        "direction": r[5]
                    })
            return results
        except Exception as e:
            app_logger.exception(f"[Archive] Error en mem_get_relations para {observation_id}")
            raise MemoryStorageError(f"Failed to get relations: {e}") from e

    async def mem_graph(self, observation_id, depth=2):
        """
        Recorre el grafo de conocimiento usando Recursive CTE desde una observación.
        
        Args:
            observation_id: ID de la observación inicial
            depth: Profundidad máxima de traversal (default: 2)
            
        Returns:
            Diccionario con:
            - nodes: Lista de nodos visitados {id, observation_id, entity_type, value}
            - edges: Lista de relaciones encontradas {source_id, target_id, relation_type}
            - levels: Diccionario indicando nivel de cada nodo
        """
        try:
            conn = await self._get_conn()
            
            # Optimized Recursive CTE
            # We use a path to avoid cycles, but we'll try to make it as light as possible
            query = """
            WITH RECURSIVE traversal(entity_id, current_depth, path) AS (
                SELECT id, 0, '|' || CAST(id AS TEXT) || '|'
                FROM entities
                WHERE observation_id = ?
                
                UNION ALL
                
                SELECT 
                    CASE WHEN e.source_id = t.entity_id THEN e.target_id ELSE e.source_id END,
                    t.current_depth + 1,
                    t.path || CASE WHEN e.source_id = t.entity_id THEN e.target_id ELSE e.source_id END || '|'
                FROM edges e
                JOIN traversal t ON (e.source_id = t.entity_id OR e.target_id = t.entity_id)
                WHERE t.current_depth < ? 
                  AND t.path NOT LIKE '%|' || CASE WHEN e.source_id = t.entity_id THEN e.target_id ELSE e.source_id END || '|%'
            )
            SELECT entity_id, MIN(current_depth) as depth FROM traversal GROUP BY entity_id;
            """
            async with conn.execute(query, (observation_id, depth)) as cursor:
                reachable_entities = await cursor.fetchall()
            
            if not reachable_entities:
                return {
                    "nodes": [],
                    "edges": [],
                    "levels": {},
                    "root_id": observation_id,
                    "depth": depth,
                    "message": f"No se encontraron entidades para observación: {observation_id}"
                }
            
            entity_ids = [r[0] for r in reachable_entities]
            levels = {r[0]: r[1] for r in reachable_entities}
            
            # Fetch all node details
            placeholders = ",".join("?" * len(entity_ids))
            async with conn.execute(
                f"SELECT id, observation_id, entity_type, value FROM entities WHERE id IN ({placeholders})",
                entity_ids
            ) as cursor:
                node_rows = await cursor.fetchall()
                nodes = [{
                    "id": r[0],
                    "observation_id": r[1],
                    "entity_type": r[2],
                    "value": r[3]
                } for r in node_rows]
            
            # Fetch all edges between these nodes
            async with conn.execute(
                f"SELECT source_id, target_id, relation_type FROM edges WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})",
                entity_ids + entity_ids
            ) as cursor:
                edge_rows = await cursor.fetchall()
                edges = [{
                    "source_id": r[0],
                    "target_id": r[1],
                    "relation_type": r[2]
                } for r in edge_rows]
            
            return {
                "nodes": nodes,
                "edges": edges,
                "levels": levels,
                "root_id": observation_id,
                "depth": depth,
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
        except Exception as e:
            app_logger.exception(f"[Archive] Error en mem_graph para {observation_id}")
            raise MemoryStorageError(f"Failed to traverse graph: {e}") from e

    # --- Phase 2.2: Core Scoring Logic ---

    def _calculate_score(self, observation_type: str, edges_count: int) -> float:
        """
        Calcula el score base para una observación.
        
        Args:
            observation_type: Tipo de observación (bugfix, decision, discovery, pattern, etc.)
            edges_count: Cantidad de edges/conexiones de la observación
            
        Returns:
            Score en rango [0.0, 5.0]
        """
        # Base score
        score = 1.0
        
        # Type bonus
        type_bonus = TYPE_BONUS.get(observation_type, 0.0)
        score += type_bonus
        
        # Edges contribution: +0.3 per edge
        edges_contribution = edges_count * 0.3
        score += edges_contribution
        
        # Clamp to [0.0, 5.0]
        return max(0.0, min(5.0, score))

    def _apply_age_penalty(self, created_at: datetime) -> float:
        """
        Aplica penalización por edad a una observación.
        
        Penaliza observaciones más antiguas usando logaritmo.
        Formula: agePenalty = -0.05 * log(days_old + 1)
        
        Args:
            created_at: Timestamp de creación de la observación
            
        Returns:
            Valor negativo (penalización) o 0.0 si hay error
        """
        try:
            if created_at is None:
                return 0.0
            
            # Handle both datetime objects and strings
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            
            now = datetime.now()
            delta = now - created_at
            days_old = delta.total_seconds() / 86400.0  # Convert to days
            
            # agePenalty = -0.05 * log(days_old + 1)
            penalty = -0.05 * math.log(days_old + 1)
            
            return penalty
        except Exception as e:
            app_logger.warning(f"[Archive] Error calculando age penalty: {e}")
            return 0.0

    def _apply_recency_bonus(self, last_accessed: datetime) -> float:
        """
        Aplica bonus por recencia (acceso reciente).
        
        Bonus que aumenta para observaciones accedidas recientemente.
        Formula: recencyBonus = +0.1 * (1 / days_since_access)
        
        Args:
            last_accessed: Timestamp del último acceso
            
        Returns:
            Valor positivo (bonus) o 0.0 si nunca fue accedida o hay error
        """
        try:
            if last_accessed is None:
                return 0.0
            
            # Handle both datetime objects and strings
            if isinstance(last_accessed, str):
                last_accessed = datetime.fromisoformat(last_accessed)
            
            now = datetime.now()
            delta = now - last_accessed
            days_since_access = delta.total_seconds() / 86400.0  # Convert to days
            
            # Avoid division by zero and cap recency bonus
            if days_since_access < 0.001:  # Less than ~1.44 minutes
                days_since_access = 0.001
            
            # recencyBonus = +0.1 * (1 / days_since_access)
            bonus = 0.1 * (1.0 / days_since_access)
            
            # Cap recency bonus at 1.0 (for very recent access)
            return min(1.0, bonus)
        except Exception as e:
            app_logger.warning(f"[Archive] Error calculando recency bonus: {e}")
            return 0.0

    def calculate_final_score(
        self,
        observation_type: str,
        edges_count: int,
        created_at: datetime,
        last_accessed: datetime
    ) -> float:
        """
        Calcula el score final combinando todos los componentes.
        
        Formula: score = 1.0 + typeBonus + (edges * 0.3) - agePenalty + recencyBonus
        
        Args:
            observation_type: Tipo de observación
            edges_count: Cantidad de edges
            created_at: Timestamp de creación
            last_accessed: Timestamp del último acceso
            
        Returns:
            Score final en rango [0.0, 5.0]
        """
        base_score = self._calculate_score(observation_type, edges_count)
        age_penalty = abs(self._apply_age_penalty(created_at))  # Make positive for clarity
        recency_bonus = self._apply_recency_bonus(last_accessed)
        
        final_score = base_score - age_penalty + recency_bonus
        
        # Clamp final score to [0.0, 5.0]
        return max(0.0, min(5.0, final_score))

    # =========================================================================
    # CAPA 2: MEMORIA EVOLUTIVA Y CONTEXTO
    # =========================================================================

    async def save_project_prompt(self, project_name: str, core_prompt: str, important_directives: str = ""):
        """
        Guarda o actualiza la configuración de prompts para un proyecto específico.
        """
        async with self._lock:
            conn = await self._get_conn()
            now = datetime.now().isoformat()
            
            # Verificar si ya existe
            async with conn.execute("SELECT id FROM project_prompts WHERE project_name = ?", (project_name,)) as cursor:
                row = await cursor.fetchone()
            
            if row:
                await conn.execute('''
                    UPDATE project_prompts 
                    SET core_prompt = ?, important_directives = ?, timestamp = ?
                    WHERE project_name = ?
                ''', (core_prompt, important_directives, now, project_name))
            else:
                await conn.execute('''
                    INSERT INTO project_prompts (project_name, core_prompt, important_directives, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (project_name, core_prompt, important_directives, now))
                
            await conn.commit()

    async def get_project_context(self, project_name: str) -> dict:
        """
        Recupera el contexto core (prompt + directivas) de un proyecto, 
        y adjunta las últimas lecciones purificadas relacionadas.
        """
        async with self._lock:
            conn = await self._get_conn()
            
            context = {
                "project_name": project_name,
                "core_prompt": "",
                "important_directives": "",
                "lessons": []
            }
            
            # 1. Obtener core prompt y directivas
            async with conn.execute('''
                SELECT core_prompt, important_directives 
                FROM project_prompts 
                WHERE project_name = ?
            ''', (project_name,)) as cursor:
                row = await cursor.fetchone()
            
            if row:
                context["core_prompt"] = row[0]
                context["important_directives"] = row[1]
                
            # 2. Obtener lecciones aprendidas (max 10 más recientes)
            async with conn.execute('''
                SELECT category, pure_lesson 
                FROM learned_lessons 
                WHERE project = ? 
                ORDER BY date DESC LIMIT 10
            ''', (project_name,)) as cursor:
                rows = await cursor.fetchall()
                for cat, lesson in rows:
                    context["lessons"].append({"category": cat, "pure_lesson": lesson})
                
            return context

    async def save_lesson(self, project: str, category: str, pure_lesson: str, token_cost: int = 0):
        """
        Guarda una lección purificada por la IA Curadora.
        """
        async with self._lock:
            conn = await self._get_conn()
            now = datetime.now().isoformat()
            
            await conn.execute('''
                INSERT INTO learned_lessons (date, project, category, pure_lesson, token_cost)
                VALUES (?, ?, ?, ?, ?)
            ''', (now, project, category, pure_lesson, token_cost))
            await conn.commit()
