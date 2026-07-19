import os
import logging
import asyncio
import requests
from google import genai
from google.genai import types

import config
from core.shared import _get_api_key, _load_system_prompt
from core.stt import LocalSTT
from core.triage import TriageManager
from core.tts import KokoroTTS

logger = logging.getLogger("core.orchestrator")

class HybridOrchestrator:
    def __init__(self, ui=None, execute_tool_cb=None):
        self.ui = ui
        self.execute_tool_cb = execute_tool_cb
        self.stt = LocalSTT()
        self.triage = TriageManager(timeout=config.TRIAGE_TIMEOUT)
        self.tts = KokoroTTS()
        self._input_lock = asyncio.Lock()
        
    async def process_voice_input(self, audio_data: bytes) -> str:
        """
        Execute the hybrid voice pipeline:
        Audio -> LocalSTT -> TriageManager -> LLM (Local / Gemini) -> KokoroTTS
        """
        async with self._input_lock:
            if not audio_data:
                return ""

            if self.ui:
                self.ui.set_state("THINKING")

            # 1. Audio -> LocalSTT (transcribe local or fallback to cloud)
            logger.info("Orchestrator: Transcribing audio input...")
            transcribed_text = await asyncio.to_thread(self.stt.transcribe, audio_data)
            transcribed_text = transcribed_text.strip()

            if not transcribed_text:
                logger.info("Orchestrator: Speech transcription is empty.")
                if self.ui:
                    if getattr(self.ui, "muted", False):
                        self.ui.set_state("THINKING")
                    else:
                        self.ui.set_state("LISTENING")
                return ""

            if self.ui:
                # Write transcription to the user log
                from i18n import _
                try:
                    self.ui.write_log(f"[you] {_('you_label')}: {transcribed_text}")
                except Exception:
                    self.ui.write_log(f"[you] You: {transcribed_text}")

            # 2. TriageManager (decide LOCAL vs CLOUD)
            logger.info(f"Orchestrator: Categorizing query: '{transcribed_text}'")
            route = await asyncio.to_thread(self.triage.route_query, transcribed_text)
            logger.info(f"Orchestrator: Intent categorized as: {route}")

            response_text = ""
            used_fallback = False

            # 3. LLM Execution & Local/Cloud Fallbacks
            if route == "[LOCAL]":
                if self.ui:
                    self.ui.write_log(f"[sys] Brain: Local")
                try:
                    # Attempt Local LLM
                    response_text = await self._call_local_llm(transcribed_text)
                    if not response_text:
                        logger.warning("Orchestrator: Local LLM returned empty response. Seamlessly falling back to Gemini.")
                        used_fallback = True
                except Exception as e:
                    logger.error(f"Orchestrator: Local LLM failed ({e}). Seamlessly falling back to Gemini.")
                    used_fallback = True

            if route == "[CLOUD]" or used_fallback:
                if self.ui:
                    lbl = "[sys] Brain: Cloud (Fallback)" if used_fallback else "[sys] Brain: Cloud"
                    self.ui.write_log(lbl)
                try:
                    # Call Gemini
                    response_text = await self._call_cloud_llm(transcribed_text)
                except Exception as e:
                    logger.error(f"Orchestrator: Cloud Gemini execution failed: {e}")
                    from i18n import _
                    response_text = _("cloud_execution_error", error=str(e))

            # 4. Kokoro TTS (synthesize via REST API)
            if response_text:
                if self.ui:
                    # Write assistant response to UI log
                    from i18n import _
                    try:
                        self.ui.write_log(f"[ai] J.A.R.V.I.S: {response_text}")
                    except Exception:
                        self.ui.write_log(f"[ai] J.A.R.V.I.S: {response_text}")
                    self.ui.set_state("SPEAKING")

                logger.info(f"Orchestrator: Synthesizing reply text via Kokoro: '{response_text}'")
                # Speak is non-blocking (enqueues text)
                await asyncio.to_thread(self.tts.speak, response_text)

                # Start background monitoring of the TTS to restore LISTENING state when done
                asyncio.create_task(self._monitor_tts_speaking())
            else:
                if self.ui:
                    if getattr(self.ui, "muted", False):
                        self.ui.set_state("THINKING")
                    else:
                        self.ui.set_state("LISTENING")

        return response_text

    async def _monitor_tts_speaking(self):
        """
        Monitors the state of the KokoroTTS playback and updates the UI state.
        """
        if not self.ui:
            return
            
        # Wait a moment for play loop to register is_speaking=True
        await asyncio.sleep(0.3)
        
        while getattr(self.tts, "is_speaking", False) or not self.tts.queue.empty():
            await asyncio.sleep(0.1)
            
        # Once finished speaking, restore LISTENING state
        if getattr(self.ui, "muted", False):
            self.ui.set_state("THINKING")
        else:
            self.ui.set_state("LISTENING")

    async def process_text_input(self, text: str) -> str:
        """
        Execute the hybrid text pipeline:
        Text -> TriageManager -> LLM (Local / Gemini) -> KokoroTTS
        
        Same flow as process_voice_input minus STT transcription.
        """
        if not text.strip():
            return ""

        async with self._input_lock:
            if self.ui:
                self.ui.set_state("THINKING")

            # 1. TriageManager (decide LOCAL vs CLOUD)
            logger.info(f"Orchestrator: Processing text input: '{text}'")
            route = await asyncio.to_thread(self.triage.route_query, text)
            logger.info(f"Orchestrator: Intent categorized as: {route}")

            response_text = ""
            used_fallback = False

            # 2. LLM Execution & Local/Cloud Fallbacks
            if route == "[LOCAL]":
                if self.ui:
                    self.ui.write_log(f"[sys] Brain: Local")
                try:
                    response_text = await self._call_local_llm(text)
                    if not response_text:
                        logger.warning("Orchestrator: Local LLM returned empty response. Seamlessly falling back to Gemini.")
                        used_fallback = True
                except Exception as e:
                    logger.error(f"Orchestrator: Local LLM failed ({e}). Seamlessly falling back to Gemini.")
                    used_fallback = True

            if route == "[CLOUD]" or used_fallback:
                if self.ui:
                    lbl = "[sys] Brain: Cloud (Fallback)" if used_fallback else "[sys] Brain: Cloud"
                    self.ui.write_log(lbl)
                try:
                    response_text = await self._call_cloud_llm(text)
                except Exception as e:
                    logger.error(f"Orchestrator: Cloud Gemini execution failed: {e}")
                    from i18n import _
                    response_text = _("cloud_execution_error", error=str(e))

            # 3. Kokoro TTS (synthesize via REST API)
            if response_text:
                if self.ui:
                    from i18n import _
                    try:
                        self.ui.write_log(f"[ai] J.A.R.V.I.S: {response_text}")
                    except Exception:
                        self.ui.write_log(f"[ai] J.A.R.V.I.S: {response_text}")
                    self.ui.set_state("SPEAKING")

                logger.info(f"Orchestrator: Synthesizing reply text via Kokoro: '{response_text}'")
                await asyncio.to_thread(self.tts.speak, response_text)
                asyncio.create_task(self._monitor_tts_speaking())
            else:
                if self.ui:
                    if getattr(self.ui, "muted", False):
                        self.ui.set_state("THINKING")
                    else:
                        self.ui.set_state("LISTENING")

        return response_text

    async def _call_local_llm(self, query: str) -> str:
        from core.shared import _load_system_prompt
        sys_prompt = _load_system_prompt()
        
        payload = {
            "model": config.LOCAL_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        loop = asyncio.get_event_loop()
        def post():
            return requests.post("http://localhost:1234/v1/chat/completions", json=payload, timeout=config.LOCAL_LLM_TIMEOUT)
            
        response = await loop.run_in_executor(None, post)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        else:
            raise RuntimeError(f"LM Studio returned status code {response.status_code}")

    async def _call_cloud_llm(self, query: str) -> str:
        from core.shared import _get_api_key, _load_system_prompt
        from main import TOOL_DECLARATIONS
        from i18n import _
        api_key = _get_api_key()
        sys_prompt = _load_system_prompt()
        
        client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"}
        )
        
        gc_config = types.GenerateContentConfig(
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            system_instruction=sys_prompt
        )
        
        MAX_TOOL_ROUNDS = 5
        history = [types.Content(role="user", parts=[types.Part.from_text(text=query)])]
        response = None
        
        for _ in range(MAX_TOOL_ROUNDS):
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=config.GEMINI_MODEL,
                contents=history,
                config=gc_config
            )
            
            # Safety filter guard: empty candidates → graceful fallback
            if not response.candidates:
                return _("safety_filter_blocked")
            
            if not response.function_calls or not self.execute_tool_cb:
                break
            
            tool_parts = []
            for fc in response.function_calls:
                logger.info(f"Orchestrator: Executing cloud tool '{fc.name}'...")
                tool_res = await self.execute_tool_cb(fc)
                
                part = types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response=tool_res.response if hasattr(tool_res, 'response') else tool_res
                    )
                )
                tool_parts.append(part)
            
            history.append(response.candidates[0].content)
            history.append(types.Content(role="tool", parts=tool_parts))
        
        return (response.text or "") if response else ""
