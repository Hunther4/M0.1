"""
ToolOrchestrator — ReAct tool loop with anti-loop protection, smart chaining,
                    FailureDoc integration, and intelligent retry.
"""

import re
import json
import asyncio
import time
from src.logger import AppLogger, Colors
from src.failure_doc import FailureRegistry
from src.retry_strategy import run_with_retry
from src.chaining import ChainEngine, default_rules
from src.rate_limiter import RateLimiter

app_logger = AppLogger(__name__)

# Global chain engine with default rules (URL depth-read, etc.)
_chain_engine = ChainEngine()
for rule in default_rules():
    _chain_engine.add_rule(rule)

# Global rate limiter for tool execution
_rate_limiter = RateLimiter()


async def run_tool_loop(
    messages,
    initial_response,
    user_msg,
    brain,
    plugin_manager,
    context_mgr,
    metrics,
    locked_tools=None,
    plan=None,
    emit_callback=None,
):
    """
    Advanced ReAct Orchestrator.
    Coordinates tool execution, handles automatic chaining, and prevents infinite loops.
    locked_tools: optional set of tool names to block (enforced by agent loop).
    Returns: (response, execution_steps, extracted_sources, usage)
    """
    messages = list(messages)  # Work on a copy to avoid mutating caller's list
    MAX_TOOL_STEPS = 10
    tool_step = 0
    execution_steps = []
    extracted_sources = {}
    response = initial_response
    usage = None

    # FailureDoc — registro persistente de fallos
    failure_registry = FailureRegistry.get_instance()

    # Anti-Loop Registry: (tool_name, args_hash) -> count
    call_registry = {}
    # Session ID para correlación de fallos
    import uuid
    session_id = uuid.uuid4().hex[:12]

    while tool_step < MAX_TOOL_STEPS:
        tool_triggered = False
        tool_context = None
        current_step = {"step": tool_step + 1, "tool": None, "query": None, "result_summary": None}

        # 1. Parse response for tool calls (use plugin_manager as source of truth)
        known_tools = set(plugin_manager.tools.keys()) if plugin_manager else None
        is_tool, valid_calls, clean_response = brain.process_response(response, known_tools=known_tools)

        if not is_tool or not valid_calls:
            break  # Final response reached

        tool_name = valid_calls[0][0]
        tool_args = valid_calls[0][1]

        # 2. Anti-Loop Check (with FailureDoc)
        args_str = json.dumps(tool_args) if isinstance(tool_args, dict) else str(tool_args)
        current_step = {"step": tool_step + 1, "tool": tool_name, "query": args_str, "result_summary": None}
        
        # --- Plan Tracking ---
        if plan and emit_callback:
            step_id = str(tool_step + 1)
            # If the plan has this step, mark it running.
            # Otherwise, we might be in an unplanned tool call (which is fine).
            emit_callback("plan_step_start", step_id)
        
        call_key = (tool_name, args_str)
        call_registry[call_key] = call_registry.get(call_key, 0) + 1

        if call_registry[call_key] > 2:
            failure_registry.record_from_loop(tool_name, args_str, tool_step, session_id)
            tool_context = (
                f"[SYSTEM ERROR] Loop detected: Tool {tool_name} called too many times with same args. "
                "Stop and provide a final answer based on available info."
            )
            tool_triggered = True
        elif tool_name not in plugin_manager.tools:
            tool_context = f"[SYSTEM ERROR] Tool {tool_name} not found. Available: {list(plugin_manager.tools.keys())}"
            tool_triggered = True
        elif not await _rate_limiter.allowed(tool_name):
            remaining = await _rate_limiter.remaining(tool_name)
            tool_context = (
                f"[SYSTEM ERROR] Tool {tool_name} rate limited. "
                f"Calls remaining in current window: {remaining}. "
                "Stop and provide a final answer or use a different approach."
            )
            tool_triggered = True
        elif locked_tools and tool_name.upper() in locked_tools:
            tool_context = (
                f"[SYSTEM ERROR] Tool {tool_name} is BLOCKED in locked-to-document mode. "
                f"Blocked tools: {sorted(locked_tools)}. "
                "Answer using only the provided documents."
            )
            tool_triggered = True
        else:
            # 3. Execution with Retry Strategy
            app_logger.info(f"[*] Step {tool_step+1}: Executing {tool_name}({args_str[:50]}...)")

            result, success, attempts = await run_with_retry(
                tool_name=tool_name,
                args_raw=args_str,
                execute_fn=plugin_manager.execute_tool,
                step_number=tool_step,
                session_id=session_id,
                failure_registry=failure_registry,
            )

            if success:
                try:
                    json.loads(result)
                    metrics.record_parse_success(True)
                except Exception:
                    metrics.record_parse_success(False)

                # 4. Smart Chaining (Dependency Resolution)
                if isinstance(result, str):
                    result = await _chain_engine.execute(
                        tool_name=tool_name,
                        result=result,
                        execute_tool=plugin_manager.execute_tool,
                        extracted_sources=extracted_sources,
                    )

                tool_triggered = True
                current_step.update({"tool": tool_name, "query": args_str, "result_summary": str(result)[:200]})
                
                if plan and emit_callback:
                    emit_callback("plan_step_done", str(tool_step + 1))
            else:
                # Tool failed after all retries
                failure_history = failure_registry.get_failure_context(tool_name)
                tool_context = (
                    f"[ERROR] Tool {tool_name} failed after {attempts} attempt(s).\n"
                    f"Last error: {result[:300]}\n"
                )
                if failure_history:
                    tool_context += f"\n{failure_history}\n"
                tool_context += (
                    "Suggest a different approach or provide the best available answer. "
                    "Do NOT retry the same failing tool."
                )
                
                if plan and emit_callback:
                    emit_callback("plan_step_fail", str(tool_step + 1), value=result[:200])
                
                tool_triggered = True

        if not tool_triggered:
            break

        # 5. Construct Directing Context
        if tool_context:
            final_context = tool_context
        else:
            # Inyectar contexto de fallos previos si los hay para esta tool
            failure_history = failure_registry.get_failure_context(tool_name)
            directive = "Analyze this result and decide: do you need more tools or can you answer the user?"
            final_context = f"[RESULT {tool_name}]\n{result}\n\n{directive}"
            if failure_history:
                final_context += f"\n\n[failure history]\n{failure_history}\n[/failure history]"

        execution_steps.append(current_step)

        # 6. Feed back to LLM
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "content": final_context})

        try:
            response, usage = await asyncio.wait_for(brain.chat(messages), timeout=120)
            brain.record_usage(usage)
            context_mgr.token_count = usage.get("prompt_tokens", 0)
            response = response.replace("<thought>", "").replace("</thought>", "").strip()
        except Exception as e:
            app_logger.exception(f"Inference failed in tool loop")
            response += f"\n\n[Critical Error: {e}]"
            break

        tool_step += 1

    return response, execution_steps, extracted_sources, usage
