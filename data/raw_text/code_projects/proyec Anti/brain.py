import httpx
import asyncio
import re
import logging
import json
import time
import random

from src.logger import AppLogger
from src.exceptions import BrainConnectionError, BrainContextError, AntiError as BrainError

logger = logging.getLogger(__name__)
app_logger = AppLogger(__name__)


# --- Tool Registry (unified) ---
# MCP_TOOLS removed — plugin_manager.tools is the single source of truth.
# Use is_mcp_tool(tool_name, known_tools) with the tool set from plugin_manager.


def is_mcp_tool(tool_name: str, known_tools: set | None = None) -> bool:
    """
    Check if a tool name is registered.
    Returns True if the tool should be invoked via MCP protocol.
    Uses known_tools (from plugin_manager.tools.keys()) as the source of truth.
    """
    if not tool_name:
        return False
    if known_tools is not None:
        return tool_name in known_tools
    # Fallback: if no tool set provided, allow (let orchestrator validate)
    return True


def get_tool_category(tool_name: str, known_tools: set | None = None) -> str:
    """Get the category of a tool for routing decisions."""
    # Categories are now determined by the tool itself, not a hardcoded registry
    return "tool"


class Brain:
    def __init__(self, base_url="http://127.0.0.1:1234/v1"):
        self.base_url = base_url
        self.model = "local-model"
        self._session = None
        self._session_lock = asyncio.Lock()
        self.max_retries = 3
        self.timeout = 120
        
        # Basic model state
        self.context_max = 32000
        self.last_prompt_tokens = 0

    async def close(self):
        """Closes the HTTP session to release resources."""
        if self._session:
            await self._session.aclose()
            self._session = None

    async def get_session(self) -> httpx.AsyncClient:
        """Obtiene o crea una sesión HTTP asíncrona reusable."""
        async with self._session_lock:
            if self._session is None:
                self._session = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.timeout, connect=10),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
                )
            return self._session

    async def chat(self, messages, temperature=0.7):
        """Envía un mensaje al LLM y retorna la respuesta (async native)."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                session = await self.get_session()
                response = await session.post(url, json=payload)
                response.raise_for_status()
                end_time = time.time()
                
                data = response.json()
                choices = data.get('choices', [])
                if not choices or not isinstance(choices[0], dict):
                    raise BrainConnectionError(f"Invalid response structure: no choices")
                msg = choices[0].get('message', {})
                content = msg.get('content')
                if content is None:
                    raise BrainConnectionError(f"Invalid response structure: no content in message")
                
                # v1.4 Sentinel Fix: Robust usage parsing
                usage_raw = data.get('usage', {})
                try:
                    prompt_tokens = max(int(usage_raw.get('prompt_tokens', 0)), 0)
                    completion_tokens = max(int(usage_raw.get('completion_tokens', 0)), 0)
                    total_tokens = max(int(usage_raw.get('total_tokens', 0)), 0)
                except (ValueError, TypeError):
                    app_logger.warning("Sentinel Warning: Data corruption in 'usage'. Using Regex Fallback.")
                    prompt_tokens = self.count_tokens(str(messages))
                    completion_tokens = self.count_tokens(content)
                    total_tokens = prompt_tokens + completion_tokens
                
                usage = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'duration': end_time - start_time,
                    'tps': (completion_tokens / (end_time - start_time)) if (end_time - start_time) > 0 else 0
                }
                
                self.last_prompt_tokens = prompt_tokens
                
                return content, usage
            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(random.uniform(0, 2 * (2 ** attempt)))
                else:
                    raise BrainConnectionError(f"Failed to connect to LLM server after {self.max_retries} attempts: {e}")

    async def stream_chat(self, messages, temperature=0.7):
        """Streaming implementation — yields (chunk, usage) tuples."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }
        start_time = time.time()
        full_content = []
        usage_raw = {}
        
        session = await self.get_session()
        try:
            async with session.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.strip():
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            if "usage" in chunk:
                                usage_raw = chunk["usage"]
                                continue
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    full_content.append(content)
                                    yield content, {}
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise BrainConnectionError(f"Streaming connection failed: {e}")
        
        end_time = time.time()
        
        prompt_tokens = max(int(usage_raw.get('prompt_tokens', 0)), 0)
        completion_tokens = max(int(usage_raw.get('completion_tokens', 0)), 0)
        total_tokens = max(int(usage_raw.get('total_tokens', 0)), 0)
        
        usage = {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'duration': end_time - start_time,
            'tps': (completion_tokens / (end_time - start_time)) if (end_time - start_time) > 0 else 0
        }
        self.last_prompt_tokens = prompt_tokens
        yield "", usage

    async def get_context_info(self):
        """Retorna información detallada sobre el contexto del modelo (v0.4 CORREGIDO)."""
        try:
            session = await self.get_session()
            res = await session.get(f"{self.base_url}/models", timeout=5)
            data = res.json()
            if 'data' in data and len(data['data']) > 0:
                model_data = data['data'][0]
                self.context_max = model_data.get('context_length', self.context_max)
                self.model = model_data.get('id', self.model)
        except Exception as e:
            app_logger.warning(f"[Brain] Error fetching model info: {e}")
        
        return {
            "max": self.context_max,
            "model": self.model
        }
    
    
    async def check_connection(self):
        """Verifica conexión con el servidor (v0.6 Compatibility)."""
        try:
            session = await self.get_session()
            res = await session.get(f"{self.base_url}/models", timeout=5)
            return res.status_code == 200
        except Exception as e:
            app_logger.warning(f"[Brain] Connection check failed: {e}")
            return False
    
    # ==================== v0.4: Token Counting ====================


    def _get_tiktoken_encoding(self):
        """
        Obtiene el encoding correcto para el modelo activo.
        tiktoken mapea automáticamente por prefijo de modelo (gpt-4, gpt-3.5, etc.).
        Para modelos locales (LM Studio, Ollama), usa cl100k_base como aproximación.
        Retorna None si tiktoken no está instalado.
        """
        try:
            import tiktoken
            # Try to get exact encoding for the model name
            try:
                return tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Unknown model (local/custom) — cl100k_base is the best universal approximation
                return tiktoken.get_encoding("cl100k_base")
        except ImportError:
            return None

    def count_tokens(self, text: str) -> int:
        """
        Cuenta tokens con precisión real usando tiktoken (si está disponible).
        Fallback automático al método regex si tiktoken no está instalado.
        """
        if not text:
            return 0
        enc = self._get_tiktoken_encoding()
        if enc is not None:
            return len(enc.encode(text))
        # Fallback: regex approximation
        return self._count_tokens_regex(text)

    def _count_tokens_regex(self, text: str) -> int:
        """Estimación regex — solo se usa cuando tiktoken no está disponible."""
        words = re.findall(r"\b[\w']+\b", text)
        punct = re.findall(r"[^\w\s]", text)
        return len(words) + len(punct) + text.count('\n')

    def count_tokens_estimate(self, text: str) -> int:
        """Estimación rápida - divide por 4."""
        return len(text) // 4

    # ==================== v0.4: U-Shape ====================
    
    def ushape_order(self, chunks: list) -> list:
        """
        U-shape ordering para máxima atención.
        [A, B, C, D, E] → [A, C, E, D, B]
        """
        if len(chunks) <= 3:
            return chunks
        
        result = []
        front = []
        back = []
        
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                front.append(chunk)
            else:
                back.append(chunk)
        
        back.reverse()
        result.extend(front)
        result.extend(back)
        
        return result

    # ==================== v0.5: DYNAMIC CONTEXT SYNC ====================
    
    async def sync_model_context(self) -> dict:
        """
        Detecta el context_length real del modelo dinámicamente.
        Combina API + fallback + cache.
        """
        try:
            session = await self.get_session()
            res = await session.get(f"{self.base_url}/models", timeout=5)
            data = res.json()
            
            if 'data' in data and len(data['data']) > 0:
                model_data = data['data'][0]
                
                new_model = model_data.get('id', self.model)
                new_context = model_data.get('context_length', self.context_max)
                old_context = self.context_max
                
                changed = False
                if new_model != self.model:
                    app_logger.info(f"Model changed: {self.model} → {new_model}")
                    self.model = new_model
                    changed = True
                
                if new_context != old_context:
                    app_logger.info(f"Context changed: {old_context} → {new_context}")
                    self.context_max = new_context
                    self._update_threshold()
                    changed = True
                
                return {"changed": changed, "old_context": old_context, "new_context": self.context_max, "model": self.model}
                    
        except Exception as e:
            app_logger.warning(f"Context sync failed: {e}")
        
        return {"changed": False}
    
    def _update_threshold(self):
        """Stub for backward compatibility if needed, but logic is now in ContextManager."""
        pass

    # ==================== v1.0: Active Coordination ====================
    
    def prepare_messages(self, system_prompt, history, max_chunks=5):
        """
        Prepares messages for the LLM, applying U-shape ordering to conversation history
        if it's fragmented into chunks for context management.
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        if not history:
            return messages
            
        messages.extend(history)
            
        return messages

    def _extract_tool_call(self, match):
        """
        Extracts tool name and arguments from a regex match using balanced brace
        matching to handle nested JSON arguments, ignoring braces inside strings.
        Returns (tool_name, tool_args_dict) or (None, None).
        """
        tool_name = match.group(1)
        raw_json = match.group(2).strip()
        if not raw_json:
            return None, None

        brace_depth = 0
        in_string = False
        escaped = False
        found_start = False

        for i, ch in enumerate(raw_json):
            if escaped:
                escaped = False
                continue
            
            if ch == '\\':
                escaped = True
                continue
                
            if ch == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if ch == '{':
                    brace_depth += 1
                    found_start = True
                elif ch == '}':
                    brace_depth -= 1
                    if found_start and brace_depth == 0:
                        json_str = raw_json[:i + 1]
                        try:
                            return tool_name, json.loads(json_str)
                        except json.JSONDecodeError as e:
                            app_logger.warning(f"Tool call JSON decode failed for {tool_name}: {e}")
                            return None, None
                    elif brace_depth < 0:
                        # Guard against negative brace_depth: ignore leading '}'
                        brace_depth = 0
        
        return None, None

    def process_response(self, response_text, known_tools=None):
        """
        Parse response for tool calls.
        Returns a tuple (is_tool_call, calls, final_text), where calls is a list of (tool_name, tool_args).
        known_tools: set of valid tool names from plugin_manager.tools.keys()
        """
        pattern = r'<tool_call name="([^"]+)"\s*>(.*?)</tool_call>'
        valid_calls = []
        
        for match in re.finditer(pattern, response_text, re.DOTALL):
            tool_name, tool_args = self._extract_tool_call(match)
            if tool_name and tool_args:
                if known_tools and tool_name not in known_tools:
                    app_logger.warning(f"Tool {tool_name} called but not registered in plugin_manager")
                valid_calls.append((tool_name, tool_args))

        if valid_calls:
            return True, valid_calls, response_text

        return False, [], response_text

