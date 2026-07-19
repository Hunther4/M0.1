"""
ContextManager - Gestor de contexto integrado para Engram v0.5

Combina:
- HybridCompactor
- U-shape ordering
- Priority pinning
- Overflow detection
- Matriz de Integridad
- Deduplicación Jaccard
"""

from typing import List, Dict, Optional
from .compactor import HybridCompactor


class ContextManager:
    """Gestor de contexto integrado v0.5."""
    
    def __init__(self, model_context_length: int, config: dict = None):
        self.model_context_length = model_context_length
        self.config = config or {}
        
        # Reservas (CORREGIDO v0.4: context-aware)
        self.reserved_system = self.config.get("reserved_system", 1500)
        self.reserved_completion = self.config.get("reserved_completion", 3000)
        
        # Budget usable (NUNCA negativo)
        self.usable = max(
            model_context_length - self.reserved_system - self.reserved_completion,
            0
        )
        self.threshold = int(self.usable * 0.85)  # 85% del usable
        self.threshold = max(self.threshold, int(self.usable * 0.5))  # Mínimo 50%
        
        # Componentes
        self.compactor = HybridCompactor(config)
        
        # Mensajes actuales
        self.messages: List[Dict] = []
        
        # Métricas
        self.token_count = 0
        self.compaction_triggered = 0
        
        # v0.6: Métricas Avanzadas
        self.tokens_saved = 0
        self.messages_deduplicated = 0
        self.maintenance_latency = 0.0  # ms
        
        # v0.5: Matriz de Integridad
        self.integrity_matrix = {
            "safe": 0.50,        # < 50% carga: normal
            "warning": 0.85,    # 50-85%: alerta
            "critical": 0.95,   # > 85%: crítico
        }
    
    async def add_message(self, role: str, content: str) -> dict:
        """
        Añade mensaje y verifica overflow.
        Retorna status y tokens count.
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        
        # Update token count (usa compactor correcto)
        self.token_count = self.compactor._messages_tokens(self.messages)
        
        # Check overflow
        overflow = self.token_count > self.threshold
        
        if overflow:
            result = await self._compact()
            self.compaction_triggered += 1
            return {
                "status": "COMPACTED",
                "messages": result,
                "tokens": self.token_count
            }
        
        return {
            "status": "OK",
            "tokens": self.token_count
        }
    
    async def _compact(self) -> List[Dict]:
        """Aplica compactación."""
        self.messages = await self.compactor.compress(self.messages, self.threshold)
        
        # Update token count después de compactar
        self.token_count = self.compactor._messages_tokens(self.messages)
        
        return self.messages
    
    async def build_with_ushape(self) -> List[Dict]:
        """Construye contexto con U-shape ordering."""
        # Primero compactar si es necesario
        if self.token_count > self.threshold:
            await self._compact()
        
        # Aplicar U-shape
        return self.compactor.ushape_order(self.messages)
    
    def pin_critical(self, content: str):
        """Marca contenido como crítico (no se elimina)."""
        pinned_msg = {"role": "system", "content": f"[PINNED] {content}"}
        
        # Insertar al inicio (después de cualquier system existing)
        insert_idx = 0
        for i, msg in enumerate(self.messages):
            if msg.get("role") == "system":
                insert_idx = i + 1
            else:
                break
        
        self.messages.insert(insert_idx, pinned_msg)
        
        # Update token count
        self.token_count = self.compactor._messages_tokens(self.messages)
    
    def get_info(self) -> dict:
        """Información del context manager."""
        return {
            "model_context": self.model_context_length,
            "usable": self.usable,
            "threshold": self.threshold,
            "current_tokens": self.token_count,
            "compaction_triggered": self.compaction_triggered,
            "stats": self.compactor.get_stats()
        }
        
    def get_advanced_stats(self) -> dict:
        """Métricas completas de salud del sistema."""
        base_stats = self.compactor.get_stats()
        
        return {
            **base_stats,
            "tokens_saved": self.tokens_saved,
            "messages_deduplicated": self.messages_deduplicated,
            "maintenance_latency_ms": self.maintenance_latency,
            "efficiency_score": self._calc_efficiency(),
        }
    
    def _calc_efficiency(self) -> float:
        """Calcula score de eficiencia (0-100)."""
        if self.usable <= 0: return 0.0
        
        usage = self.token_count / self.usable
        saved = self.tokens_saved / self.usable if self.usable > 0 else 0
        
        return round((1 - usage + saved) * 100, 1)

    # ==================== v0.4: MÉTODOS AUXILIARES ====================
    
    @property
    def usage_percent(self) -> float:
        """Porcentaje de uso del usable."""
        if self.usable == 0:
            return 0.0
        return round((self.token_count / self.usable) * 100, 1)
    
    @property
    def is_overflowing(self) -> bool:
        """¿Está en overflow?"""
        return self.token_count > self.threshold
    
    def clear(self):
        """Limpia todos los mensajes."""
        self.messages = []
        self.token_count = 0
    
    async def update_context_length(self, new_length: int):
        """Actualiza la longitud del contexto y recalcula los límites.
        Evita la pérdida de métricas y el historial de mensajes.
        """
        self.model_context_length = new_length
        self.usable = max(
            new_length - self.reserved_system - self.reserved_completion,
            0
        )
        self.threshold = int(self.usable * 0.85)
        self.threshold = max(self.threshold, int(self.usable * 0.5))
        
        # Re-evaluate current overflow if applicable
        if self.token_count > self.threshold:
            await self._compact()
    
    def set_messages(self, messages: List[Dict]):
        """Reemplaza mensajes (ej: para cargar desde sesión)."""
        self.messages = messages
        self.token_count = self.compactor._messages_tokens(self.messages)
    
    # ==================== v0.4: PERSISTENCE ====================
    
    def preserve(self, label: str = "session") -> str:
        """Persiste a filesystem."""
        return self.compactor.preserve_to_disk(self.messages, label)
    
    def record_result(self, task: str, success: bool, info_used: List[str]):
        """Registra resultado para failure-driven learning."""
        self.compactor.record_result(task, success, info_used)
    
    def optimize_guidelines(self, llm=None) -> str:
        """Optimiza guidelines basándose en failures."""
        return self.compactor.optimize_guidelines(llm)
    
    # ==================== v0.5: MATRIZ DE INTEGRIDAD ====================
    
    def get_load_level(self) -> str:
        """Retorna nivel de carga actual.
        
        Uses usable capacity (not threshold) as denominator so that
        'overflow' is correctly returned when token_count >= usable.
        """
        if self.usable <= 0:
            return "overflow"
        usage = self.token_count / self.usable
        
        if usage < self.integrity_matrix["safe"]:
            return "safe"
        elif usage < self.integrity_matrix["warning"]:
            return "warning"
        elif usage < self.integrity_matrix["critical"]:
            return "critical"
        else:
            return "overflow"
    
    def get_integrity_action(self) -> str:
        """Retorna acción según nivel de carga."""
        level = self.get_load_level()
        
        actions = {
            "safe": "_NORMAL",
            "warning": "_DEDUPLICATE_JACCARD",
            "critical": "_TRIGGER_CONSOLIDATE",
            "overflow": "_EMERGENCY_TRUNCATE"
        }
        
        return actions.get(level, "_UNKNOWN")
    
    # ==================== v0.5: DEDUPLICACIÓN ====================
    
    def deduplicate(self) -> int:
        """
        Deduplicación Adaptativa v1.3:
        Ajusta la agresividad según el nivel de carga.
        """
        level = self.get_load_level()
        
        # Threshold adaptativo: más carga = más agresivo (menor threshold Jaccard)
        # Nota: 0.7 es estándar, 0.4 es muy agresivo
        thresholds = {
            "safe": 1.0,      # No deduplicar
            "warning": 0.7,   # Estándar
            "critical": 0.5,  # Agresivo
            "overflow": 0.3   # Extremo
        }
        
        current_threshold = thresholds.get(level, 0.7)
        if current_threshold >= 1.0: return 0
        
        before = len(self.messages)
        self.messages = self.compactor.deduplicate_messages(self.messages, current_threshold)
        after = len(self.messages)
        
        # Update token count
        self.token_count = self.compactor._messages_tokens(self.messages)
        
        return before - after
    
    def emergency_truncate(self) -> List[Dict]:
        """Truncamiento de emergencia basado en token budget (Sprint 3).
        
        En lugar de mantener un número fijo de mensajes (10), calcula un budget
        de tokens para la ventana reciente y descarta mensajes antiguos hasta
        cumplirlo. También aplica deduplicación semántica si es posible.
        """
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        non_system = [m for m in self.messages if m.get("role") != "system"]
        
        if not non_system:
            self.messages = system_msgs
            self.token_count = self.compactor._messages_tokens(self.messages)
            return self.messages
        
        # Calcular budget para la ventana reciente: 40% del usable
        recent_budget = int(self.usable * 0.40)
        recent_budget = max(recent_budget, 500)  # Mínimo 500 tokens
        
        # Primero: deduplicación semántica agresiva
        if len(non_system) > 10:
            non_system = self.compactor.deduplicate_messages(non_system, threshold=0.6)
        
        # Segundo: token-based window de atrás hacia adelante
        # (mantener el final de la conversación)
        recent = []
        tokens_used = 0
        
        for msg in reversed(non_system):
            msg_tokens = self.compactor.count_tokens(msg.get("content", ""))
            if tokens_used + msg_tokens <= recent_budget:
                recent.insert(0, msg)
                tokens_used += msg_tokens
            else:
                # Último mensaje parcial si hay espacio
                remaining = recent_budget - tokens_used
                if remaining > 20:
                    content = msg.get("content", "")
                    
                    # Truncamiento preciso: iteramos hasta que count_tokens cumpla el budget
                    truncated = content
                    while len(truncated) > 20 and self.compactor.count_tokens(truncated) > remaining:
                        # Reducimos proporcionalmente basándonos en la diferencia de tokens
                        current_tokens = self.compactor.count_tokens(truncated)
                        ratio = remaining / current_tokens if current_tokens > 0 else 0.9
                        truncated = truncated[:max(int(len(truncated) * ratio), 20)]
                    
                    truncated = truncated + "\n[TRUNCATED]"
                    recent.insert(0, {"role": msg.get("role", "assistant"), "content": truncated})
                break
        
        self.messages = system_msgs + recent
        self.token_count = self.compactor._messages_tokens(self.messages)
        
        return self.messages