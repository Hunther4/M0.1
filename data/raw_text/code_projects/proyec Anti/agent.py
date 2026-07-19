"""
AntiAgent — Core orchestrator (v1.6 Quantum).

Slimmed-down version: delegates to renderer, prompt_builder,
tool_orchestrator, and command_handler modules.
"""

import os
import json
import re
import asyncio
import logging
import subprocess
import time
import uuid
import signal

from rich.console import Console
from rich.panel import Panel

from src.logger import AppLogger, Colors, set_request_id
from src.brain import Brain
from src.exceptions import BrainConnectionError
from src.memory import MemoryManager
from src.context_manager import ContextManager
from src.scorer import PRMScorer
from src.evolver import SkillEvolver
from src.consolidator import MemoryConsolidator
from src import metrics

from src.renderer import render_markdown, display_banner
from src.prompt_builder import build_agent_prompt
from src.tool_orchestrator import run_tool_loop
from src import command_handler

logger = logging.getLogger(__name__)
app_logger = AppLogger(__name__)


class AntiAgent:
    DEFAULT_LM_URL = "http://127.0.0.1:1234/v1"

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.console = Console()

        # Local-First config
        self.local_config_path = os.path.join(self.base_dir, "config.local.json")
        self.default_config_path = os.path.join(self.base_dir, "config.json")
        self.config = self._load_config()

        # Provider initialization
        from src.providers import create_provider
        provider_type = self.config.get("provider", "auto")
        if provider_type == "auto":
            try:
                import httpx
                detected_url = None
                with httpx.Client(timeout=2) as client:
                    for port in [1234, 11434, 8000, 8001]:
                        try:
                            endpoint = "/api/tags" if port == 11434 else "/v1/models"
                            url = f"http://127.0.0.1:{port}"
                            r = client.get(f"{url}{endpoint}")
                            if r.status_code == 200:
                                detected_url = url if port == 11434 else f"{url}/v1"
                                break
                        except Exception:
                            logger.debug("Omitiendo puerto fallido")
                            continue
                if detected_url:
                    if ":11434" in detected_url:
                        self.brain = create_provider("ollama", base_url=detected_url,
                            model=self.config.get("model"), timeout=self.config.get("timeout", 120))
                    else:
                        self.brain = create_provider("lmstudio", base_url=detected_url,
                            model=self.config.get("model"), timeout=self.config.get("timeout", 120))
                    logger.info(f"Proveedor auto-detectado: {type(self.brain).__name__}")
                else:
                    raise Exception("No provider found")
            except Exception as e:
                logger.warning(f"Auto-deteccion fallo: {e}. Usando LM Studio por defecto.")
                self.brain = create_provider(
                    "lmstudio",
                    base_url=self.config.get("lm_studio_url", self.DEFAULT_LM_URL),
                    model=self.config.get("model")
                )
        else:
            url_config = self.config.get(f"{provider_type}_url",
                          self.config.get("lm_studio_url", self.DEFAULT_LM_URL))
            api_key = self.config.get(f"{provider_type}_api_key")
            self.brain = create_provider(
                provider_type,
                base_url=url_config,
                model=self.config.get("model"),
                api_key=api_key
            )

        workspace_path = os.path.join(self.base_dir, "workspace")
        if not os.path.exists(workspace_path):
            os.makedirs(workspace_path, exist_ok=True)

        self.memory = MemoryManager(
            memory_path=os.path.join(self.base_dir, "memory"),
            workspace_path=workspace_path
        )

        self.context_mgr = ContextManager(model_context_length=32000)
        self.is_running = True
        self.task_counter = 0
        self.history = []
        self.reasoner_mode = False

        url = getattr(self.brain, 'base_url', self.config.get("lm_studio_url", self.DEFAULT_LM_URL))
        self.scorer = PRMScorer(prm_url=url, prm_model=self.brain.model)
        self.evolver = SkillEvolver(base_url=url, model="local-model")

        from src.plugin_manager import PluginManager
        self.plugin_manager = PluginManager(plugins_dir=os.path.join(self.base_dir, "src/plugins"))
        self.consolidator = MemoryConsolidator(self.memory, self.evolver)
        self.last_maintenance_count = 0

    def _load_config(self):
        from src.config_validator import validate_config
        for path in (self.local_config_path, self.default_config_path):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    return validate_config(config)
        raise FileNotFoundError(
            "Configuration file not found. Please copy config.json.example to "
            "config.local.json and fill in your keys."
        )

    async def close(self):
        if hasattr(self, 'brain') and self.brain:
            if hasattr(self.brain, 'close'):
                await self.brain.close()
        if hasattr(self, 'memory') and self.memory:
            self.memory.close()
            if hasattr(self.memory, 'archive') and self.memory.archive:
                try:
                    await self.memory.archive.close()
                except Exception as e:
                    logger.warning(f"Error closing archive: {e}")
        app_logger.info("AntiAgent resources closed successfully.")

    # --- Delegated: Rendering ---

    def _emit_tui_message(self, msg_type: str, content: str, value: str = None):
        """Emits a JSON-line message to the Go TUI bridge over stdout."""
        msg = {"type": msg_type, "content": content}
        if value:
            msg["value"] = value
        print(json.dumps(msg), flush=True)

    async def _generate_plan(self, user_text: str):
        """Determines if the request is complex and generates a TaskPlan."""
        prompt = (
            f"Analyze the following user request and determine if it is complex "
            f"(requires multiple steps, tool calls, or deep research).\n\n"
            f"Request: {user_text}\n\n"
            f"If it is complex, respond ONLY with a JSON object matching this structure:\n"
            f"{{ \"goal\": \"overall goal\", \"steps\": [ {{ \"id\": \"1\", \"name\": \"step description\" }}, ... ] }}\n"
            f"If it is NOT complex, respond with 'SIMPLE'."
        )
        
        try:
            response, _ = await self.brain.chat([{"role": "user", "content": prompt}])
            response = response.strip()
            
            if response == "SIMPLE":
                return None
            
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
                plan_id = uuid.uuid4().hex[:8]
                return {
                    "id": plan_id,
                    "goal": plan_data.get("goal", user_text),
                    "steps": [
                        {"id": str(i+1), "name": s.get("name", ""), "status": "pending"}
                        for i, s in enumerate(plan_data.get("steps", []))
                    ]
                }
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
        return None

    def render_markdown(self, text: str) -> str:
        return render_markdown(text)

    # --- Delegated: Commands ---

    async def handle_command(self, cmd, image_data=None, rid=None):
        return await command_handler.handle_command(cmd, self, image_data=image_data, rid=rid)

    # --- Core Processing Pipeline ---

    async def _process(self, user_msg, image_data=None, _depth=0, rid=None, _total_refinements=0):
        # Guard: prevent stack overflow from recursive refinement
        if _depth > 3:
            logger.warning("[Agent] Max refinement depth reached, returning as-is")
            return {"response": str(user_msg), "steps": [], "sources": {}, "usage": {}, "score": 0.0}

        user_text = user_msg if isinstance(user_msg, str) else str(user_msg)

        # Use provided correlation ID or generate one
        if rid is None:
            rid = uuid.uuid4().hex[:12]
        set_request_id(rid)

        # 1. Build System Prompt
        prompt_result = build_agent_prompt(
            user_text=user_text,
            config=self.config,
            memory=self.memory,
            plugin_manager=self.plugin_manager,
            base_dir=self.base_dir,
        )
        system_prompt = prompt_result["prompt"]
        locked_tools = prompt_result["locked_tools"]

        # 2. Build conversation thread
        messages = [{"role": "system", "content": system_prompt}]
        for msg in self.history:
            if isinstance(msg["content"], list):
                text = next((item["text"] for item in msg["content"] if item["type"] == "text"), "Imagen previa")
                messages.append({"role": msg["role"], "content": text})
            else:
                messages.append(msg)

        if image_data:
            print(f"{Colors.YELLOW}[i] Imagen recibida para analisis.{Colors.END}")
            user_content = [
                {"type": "text", "text": user_text if user_text else "Analiza esta imagen."},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        else:
            user_content = user_text

        messages.append({"role": "user", "content": user_content})

        # 3. Initial Chat Inference
        start_timestamp = time.time()
        streaming_enabled = self.config.get("streaming", False)

        try:
            if streaming_enabled and hasattr(self.brain, 'stream_chat'):
                full_response_chunks = []
                usage = {}
                # Use stream_chat generator
                async for chunk, chunk_usage in asyncio.wait_for(self.brain.stream_chat(messages), timeout=120):
                    if chunk_usage:
                        usage = chunk_usage
                    if chunk:
                        full_response_chunks.append(chunk)
                        self._emit_tui_message("chunk", chunk)
                
                response = "".join(full_response_chunks)
                metrics.record_ttft(start_timestamp)
                
                if usage:
                    completion_tokens = usage.get('completion_tokens', 0)
                    duration = usage.get('duration', 0)
                    metrics.record_token_generation(completion_tokens, duration)
                    self.brain.record_usage(usage)
                    self.context_mgr.token_count = usage.get("prompt_tokens", 0)
            else:
                response, usage = await asyncio.wait_for(self.brain.chat(messages), timeout=120)
                metrics.record_ttft(start_timestamp)
                completion_tokens = usage.get('completion_tokens', 0)
                duration = usage.get('duration') if usage.get('duration') is not None else usage.get('time', 0)
                metrics.record_token_generation(completion_tokens, duration)
                self.brain.record_usage(usage)
                self.context_mgr.token_count = usage.get("prompt_tokens", 0)
        except BrainConnectionError as e:
            app_logger.error(f"Brain connection error: {e}")
            return {
                "response": f"No pude procesar tu solicitud. Error de conexion: {e}",
                "steps": [], "sources": {},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration": 0, "tps": 0},
                "score": 0.0
            }
        except Exception as e:
            app_logger.exception(f"Chat inference failed")
            return {
                "response": f"Error en inferencia: {e}",
                "steps": [], "sources": {},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration": 0, "tps": 0},
                "score": 0.0
            }

        if isinstance(response, (list, tuple)):
            response_str = response[0] if len(response) > 0 else ""
        else:
            response_str = str(response)

        response = response_str.replace("<thought>", "").replace("</thought>", "").strip()
        
        # 3.5 Plan Generation for complex requests
        plan = await self._generate_plan(user_text)
        if plan:
            self._emit_tui_message("plan_create", json.dumps(plan))

        # 4. ReAct Tool Loop (delegated)
        final_response, execution_steps, extracted_sources, final_usage = await run_tool_loop(
            messages=messages,
            initial_response=response,
            user_msg=user_text,
            brain=self.brain,
            plugin_manager=self.plugin_manager,
            context_mgr=self.context_mgr,
            metrics=metrics,
            locked_tools=locked_tools,
            plan=plan,
            emit_callback=self._emit_tui_message,
        )


        # 5. Evaluation & Refinement
        tool_step = len(execution_steps)
        final_response, score, is_success, votes = await self._evaluate_response(final_response, user_text, tool_step, _depth=_depth, _total_refinements=_total_refinements)

        # 6. Update History & Stats
        await self._update_history(user_msg, final_response, is_success, score, votes)

        # Auto-maintenance
        self.task_counter += 1
        if self.task_counter >= 10:
            await self._reflect()
            # Evict old engrams (decay-based)
            try:
                evicted = await self.memory.decay_old_engrams(max_fallos=3)
                if evicted > 0:
                    logger.info(f"[Memory] Auto-evicted {evicted} stale engrams")
            except Exception as e:
                logger.warning(f"[Memory] Decay failed: {e}")
            self.task_counter = 0

        await self._check_integrity(final_usage.get("prompt_tokens", 0) if final_usage else 0)

        return {
            "response": final_response,
            "steps": execution_steps,
            "sources": extracted_sources,
            "usage": final_usage if final_usage else usage,
            "score": score
        }

    async def _update_history(self, user_msg, final_response, is_success, score=None, votes=None):
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": final_response})
        if len(self.history) > 20:
            # Trim to even number to preserve user/assistant turn pairs
            trim = len(self.history) - 20
            if trim % 2 != 0:
                trim += 1
            self.history = self.history[trim:]
        await self.memory.log_experience(user_msg, final_response, is_success, score, votes)

    async def _evaluate_response(self, response, user_text, tool_step, _depth=0, _total_refinements=0):
        """Evaluates response quality using PRM Scorer and optionally refines."""
        try:
            result = await self.scorer.evaluate(
                response=response, instruction=user_text, turn_num=tool_step,
            )
            score = 0.0
            if result and isinstance(result, dict):
                val = result.get("score")
                if isinstance(val, (int, float)):
                    score = float(val)
            votes = result.get("votes", []) if result and isinstance(result, dict) else []

            # If the scorer returned None/unparseable score, the model can't judge quality.
            # Don't waste time refining — return as-is.
            if result is None or (isinstance(result, dict) and result.get("score") is None):
                app_logger.info("[Agent] PRM scorer returned None — skipping refinement")
                return response, 0.0, False, []

            max_refinements = 3
            refinement_step = 0
            best_response = response
            best_score = score

            while score < 0.5 and self.config.get("enable_prm_scorer", True) and refinement_step < max_refinements and _total_refinements < 6:
                refinement_step += 1
                print(f"{Colors.YELLOW}[*] Calidad insuficiente ({score:.2f} < 0.5). Refinamiento ({refinement_step}/{max_refinements})...{Colors.END}")

                try:
                    refined_response = await self._process(
                        f"REFINEMENT REQUEST: The previous response was rated {score:.2f}/1.0. "
                        f"Instruction: {user_text}\n"
                        f"Previous Response: {response}\n\n"
                        f"Please provide a corrected, high-quality version. If you lack specific data, "
                        f"USE YOUR TOOLS (SEARCH, WEB_READ) now to find it. "
                        f"Do NOT explain your errors, just deliver the final result.",
                        _depth=_depth + 1,
                        _total_refinements=_total_refinements + 1,
                    )
                    if isinstance(refined_response, dict):
                        candidate = refined_response.get("response", response)
                    else:
                        candidate = str(refined_response)
                    # Reject meta-prompt echoes — keep best real response
                    if not candidate.startswith("REFINEMENT REQUEST:"):
                        response = candidate
                except Exception as e:
                    app_logger.warning(f"Refinement process failed: {e}")
                    break

                result = await self.scorer.evaluate(
                    response=response, instruction=user_text, turn_num=tool_step + refinement_step,
                )
                score = 0.0
                if result and isinstance(result, dict):
                    val = result.get("score")
                    if isinstance(val, (int, float)):
                        score = float(val)
                votes = result.get("votes", []) if result and isinstance(result, dict) else []

                if score > best_score:
                    best_score = score
                    best_response = response

            is_success = best_score >= 0.5
            return best_response, best_score, is_success, votes

        except Exception as e:
            app_logger.warning(f"PRM evaluation failed: {e}")
            return response, 0.0, False, []

    # --- Evolution & Maintenance ---

    async def _reflect(self):
        logger.info("Iniciando evolucion autonoma profunda (Dual)...")
        logs = self.memory.get_recent_logs(50)

        logger.info("Fase 1: Extrayendo conocimiento factual (Engrams)...")
        try:
            new_engrams = await self.evolver.extract_engrams(logs)
            for e in new_engrams:
                self.memory.save_engram(e.get("topic", "tema-desconocido"), e.get("content", ""))
                logger.info(f"Engram memorizado: {e.get('topic')}")
        except Exception as e:
            app_logger.exception("Error in Engram extraction")
            logger.error(f"Error en extraccion de Engrams: {e}")

        logger.info(f"Fase 2: Analizando {len(logs)} experiencias para destilar mejores practicas...")
        try:
            new_skills = await self.evolver.evolve(logs, self.memory.skills.skills)
        except Exception as e:
            app_logger.exception("Error in Skill Evolver")
            logger.error(f"Error en Evolver (Skills): {e}")
            return "Error en evolucion de habilidades."

        if not new_skills:
            logger.info("El sistema considera que las reglas actuales son optimas.")
            return "Evolucion completada sin nuevas reglas de comportamiento."

        for skill in new_skills:
            self.memory.skills.add_skill(
                name=skill.get("name"), description=skill.get("description"),
                content=skill.get("content"), category=skill.get("category", "forced-evolution")
            )
            logger.info(f"Evolucion aplicada: {skill.get('name')}")

        return f"Evolucion Dual completada. Nuevos Engrams memorizados y {len(new_skills)} nuevas directivas anadidas."

    async def _compact_memory(self):
        logger.info("Compactando memoria...")
        from prompts.templates import COMPACT_PROMPT
        patterns = self.memory.load_patterns()
        if not patterns.strip():
            logger.info("Memoria vacia, nada que compactar.")
            return
        prompt = COMPACT_PROMPT.format(patterns=patterns[:4000])
        compacted, _ = await self.brain.chat([{"role": "user", "content": prompt}])
        self.memory.save_pattern(compacted)
        logger.info("Memoria compactada.")

    async def _check_integrity(self, current_prompt_tokens=0):
        if current_prompt_tokens > 0:
            self.context_mgr.token_count = current_prompt_tokens

        await self.brain.sync_model_context()
        model_context = getattr(self.brain, "context_max", self.context_mgr.model_context_length)

        if self.context_mgr.model_context_length != model_context:
            await self.context_mgr.update_context_length(model_context)

        self.scorer.prm_model = self.brain.model

        usage_percent = self.context_mgr.usage_percent
        level = self.context_mgr.get_load_level()

        if level == "warning":
            removed = self.context_mgr.deduplicate()
            if removed > 0:
                logger.info(f"Anti-Deduplication: {removed} mensajes redundantes eliminados.")
        elif level in ("critical", "overflow"):
            logger.info(f"Anti-Alert ({level}): {usage_percent}%. Limpieza Sentinel...")
            self.context_mgr.deduplicate()
            await self.consolidator.run_maintenance()
            try:
                await self._compact_memory()
            except Exception as e:
                app_logger.warning(f"Memory compaction failed during integrity check: {e}")
            return

        engrams_count = self.memory.count_engrams()
        skills_count = len(self.memory.skills.skills)
        total = engrams_count + skills_count
        thresholds = [20] + list(range(50, 550, 50))

        current_threshold = 0
        for t in thresholds:
            if total >= t:
                current_threshold = t
            else:
                break

        if current_threshold > self.last_maintenance_count:
            logger.info(f"Anti-Memory Threshold ({total}). Consolidando...")
            await self.consolidator.run_maintenance()
            self.last_maintenance_count = current_threshold

    async def _renew_system(self):
        print(f"{Colors.BLUE}[*] Iniciando ciclo de renovacion...{Colors.END}")
        try:
            # Kill previous server process by tracked PID instead of pkill -f
            if hasattr(self, 'server_proc') and self.server_proc is not None:
                try:
                    os.kill(self.server_proc.pid, 15)  # SIGTERM
                    await asyncio.to_thread(self.server_proc.wait, timeout=3)
                except (ProcessLookupError, OSError):
                    pass  # Already dead
                except Exception:
                    logger.debug("Fallo SIGTERM, intentando forzar muerte")
                    try:
                        self.server_proc.kill()
                    except Exception:
                        logger.debug("Fallo forzar muerte")
            self.server_proc = None

            python_exe = "python3"
            venv_python = os.path.join(self.base_dir, "venv/bin/python3")
            if os.path.exists(venv_python):
                python_exe = venv_python

            server_script = os.path.join(self.base_dir, "server.py")
            env = os.environ.copy()
            env["ANTI_MANAGED"] = "1"
            proc = await asyncio.to_thread(
                subprocess.Popen,
                [python_exe, server_script],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.PIPE, cwd=self.base_dir,
                env=env
            )
            # Pipe the shared secret from server module to child process stdin
            try:
                import sys as _sys
                _server_mod = _sys.modules.get('server')
                _secret = getattr(_server_mod, 'SHARED_SECRET', None) if _server_mod else None
                if not _secret or not isinstance(_secret, bytes) or len(_secret) != 32:
                    raise RuntimeError(
                        "SHARED_SECRET is missing or invalid. Ensure server module "
                        "is loaded and provides a 32-byte secret."
                    )
                await asyncio.to_thread(proc.stdin.write, _secret)
                await asyncio.to_thread(proc.stdin.flush)
                proc.stdin.close()
            except Exception as e:
                logger.warning(f"Error escribiendo SHARED_SECRET a stdin: {e}")
            self.server_proc = proc
            logger.info("Nuevo servidor iniciado con el codigo actualizado.")

            await asyncio.sleep(1)
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    health_res = await client.get("http://127.0.0.1:8000/health", timeout=2)
                    if health_res.status_code == 200:
                        return "Sistema renovado. El dashboard y el servidor ahora corren con la ultima version."
                    else:
                        return f"Servidor iniciado pero salud no confirmada (Status: {health_res.status_code})."
            except Exception as e:
                return f"Servidor iniciado pero no se pudo contactar el endpoint de salud: {e}"
        except Exception as e:
            return f"Error al renovar: {e}"

    # --- CLI Entry Point ---

    def run(self):
        provider_name = type(self.brain).__name__.lower()
        is_local = "lmstudio" in provider_name or "ollama" in provider_name
        display_banner(self.console, is_local)

        def handle_exit(sig, frame):
            self.is_running = False
            os.kill(os.getpid(), signal.SIGINT)

        signal.signal(signal.SIGTERM, handle_exit)
        signal.signal(signal.SIGINT, handle_exit)

        try:
            asyncio.run(self._async_run(is_local))
        except KeyboardInterrupt:
            pass
        finally:
            try:
                asyncio.run(self.close())
            except Exception as e:
                logger.error(f"Error during final close: {e}")

    async def _async_run(self, is_local: bool):
        # Share the event loop with MemoryManager to prevent async leak
        self.memory._event_loop = asyncio.get_running_loop()

        try:
            if not await self.brain.check_connection():
                self.console.print(f"[bold yellow][!] Advertencia: No se pudo conectar con el proveedor seleccionado.[/]")
                self.console.print(f"[bold yellow]    Asegurate de que el servidor local o tu API key esten configurados.[/]\n")
        except Exception as e:
            self.console.print(f"[bold red][!] Error critico verificando conexion: {e}[/]")
        await self._async_input_loop(is_local)

    async def _async_input_loop(self, is_local: bool):
        prompt_text = "Anti@Local" if is_local else "Anti@Cloud"
        prompt_color = "green" if is_local else "blue"

        self.console.print("\n[bold magenta]Bienvenido al nucleo de Anti-Agent. Escribe [bold cyan]'help'[ /bold cyan] para ver comandos.[/bold magenta]")

        while self.is_running:
            try:
                user_input = await asyncio.to_thread(self.console.input, f"[{prompt_color} bold]>>> {prompt_text}[/]")
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit"]:
                    self.is_running = False
                    self.console.print(f"\n[bold blue][*] Apagando sistemas... Hasta pronto![/]")
                    break

                with self.console.status(f"[bold yellow]Procesando...[/]", spinner="dots"):
                    result = await self.handle_command(user_input)

                if result:
                    if isinstance(result, dict) and "response" in result:
                        formatted_response = self.render_markdown(result['response'])
                        self.console.print(Panel(formatted_response, title="[bold cyan]Anti[/]", border_style="blue"))
                    else:
                        formatted_result = self.render_markdown(str(result))
                        self.console.print(Panel(formatted_result, title="[bold cyan]Anti[/]", border_style="blue"))

            except KeyboardInterrupt:
                self.console.print(f"\n[bold yellow][!] Interrumpido por el usuario.[/]")
                self.is_running = False
                break
            except EOFError:
                self.console.print(f"\n[bold yellow][!] EOF recibido. Saliendo...[/]")
                self.is_running = False
                break
            except Exception as e:
                app_logger.exception(f"Error en CLI loop")
                self.console.print(f"\n[bold red][!] Error: {e}[/]")
