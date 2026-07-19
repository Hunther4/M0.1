"""
Text-to-Speech implementation using Kokoro-FastAPI.

Replaces the legacy Piper-based TTS with a REST client that sends text
to a Kokoro-FastAPI server and plays back the returned PCM audio
via sounddevice.
"""

import logging
import queue
import threading
from typing import Optional

import numpy as np
import requests
import sounddevice as sd

import config

logger = logging.getLogger("core.tts")

KOKORO_SAMPLE_RATE = 24000  # Kokoro-FastAPI PCM output sample rate


class KokoroTTS:
    """REST-based TTS client that queues text and plays audio via sounddevice.

    Uses the OpenAI-compatible /v1/audio/speech endpoint exposed by
    Kokoro-FastAPI.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        voice: Optional[str] = None,
    ):
        if base_url is None:
            base_url = f"http://{config.KOKORO_HOST}:{config.KOKORO_PORT}"
        if voice is None:
            voice = config.KOKORO_VOICE

        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.tts_url = f"{self.base_url}/v1/audio/speech"

        # Threading and queueing (same contract as the old LocalTTS)
        self.queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        self.is_speaking = False

        self.start()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background worker thread."""
        with self.lock:
            if self.running:
                return
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._worker_loop, daemon=True
            )
            self.worker_thread.start()

    def stop_worker(self) -> None:
        """Signal the worker thread to stop and wait for it."""
        with self.lock:
            self.running = False
            self.queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=2)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """Consume the text queue and synthesise/play each item."""
        while self.running:
            try:
                item = self.queue.get()
                if item is None:
                    break
                if not self.running:
                    self.queue.task_done()
                    continue
                try:
                    self._synthesize_and_play(item)
                finally:
                    self.queue.task_done()
            except Exception:
                logger.exception("Error in TTS worker loop")

    # ------------------------------------------------------------------
    # Synthesis + playback
    # ------------------------------------------------------------------

    def _synthesize_and_play(self, text: str) -> None:
        """Send *text* to Kokoro-FastAPI and play the returned PCM audio."""
        if not text.strip():
            return

        # Determine voice dynamically based on text language
        import re
        voice = self.voice
        
        if any(char in text for char in ["¿", "¡", "á", "é", "í", "ó", "ú", "ñ"]):
            voice = "em_alex"  # Default high-quality Spanish voice
        else:
            text_lower = text.lower()
            spanish_words = r"\b(que|hola|como|cómo|este|para|con|todo|bien|gracias|ayuda|programacion|programación)\b"
            if re.search(spanish_words, text_lower):
                voice = "em_alex"
            else:
                english_words = r"\b(the|and|of|to|is|you|that|it|was|for|on|are|with|hello|goodbye|please|would)\b"
                if re.search(english_words, text_lower):
                    voice = "af_heart"  # Default high-quality English voice

        logger.info("[TTS] Synthesising (voice=%s): %s…", voice, text[:60])

        try:
            resp = requests.post(
                self.tts_url,
                json={
                    "model": "kokoro",
                    "input": text,
                    "voice": voice,
                    "response_format": "pcm",
                    "speed": 1.0,
                },
                timeout=30,
            )
        except requests.ConnectionError:
            logger.error(
                "[TTS] Kokoro-FastAPI connection refused at %s. "
                "Is the server running?",
                self.base_url,
            )
            return
        except requests.Timeout:
            logger.error("[TTS] Kokoro-FastAPI request timed out after 30 s.")
            return
        except requests.RequestException as exc:
            logger.error("[TTS] Kokoro-FastAPI request failed: %s", exc)
            return

        if resp.status_code != 200:
            logger.error(
                "[TTS] Kokoro-FastAPI returned HTTP %d: %s",
                resp.status_code,
                resp.text[:200],
            )
            return

        audio_bytes = resp.content
        if not audio_bytes:
            logger.error("[TTS] Kokoro returned empty audio data!")
            return

        # Ensure even number of bytes for 16-bit PCM conversion
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]

        logger.info("[TTS] Received %d bytes of PCM audio. Playing…", len(audio_bytes))

        # Convert PCM s16le → float32
        audio_data = (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        with self.lock:
            self.is_speaking = True

        try:
            self._play_audio(audio_data)
        finally:
            with self.lock:
                self.is_speaking = False

    def _play_audio(self, audio_data: np.ndarray) -> None:
        """Play a float32 waveform through the configured speaker device."""
        speaker_idx = _speaker_device()

        try:
            stream = sd.OutputStream(
                samplerate=KOKORO_SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=speaker_idx,
            )
            stream.start()
        except Exception:
            logger.warning(
                "[TTS] Device %s failed, falling back to default.",
                speaker_idx,
            )
            try:
                stream = sd.OutputStream(
                    samplerate=KOKORO_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                    device=None,
                )
                stream.start()
            except Exception as exc:
                logger.error("[TTS] Cannot open audio output: %s", exc)
                return

        block_size = 4096
        try:
            for i in range(0, len(audio_data), block_size):
                with self.lock:
                    if not self.running or not self.is_speaking:
                        break
                stream.write(audio_data[i : i + block_size])
        finally:
            stream.stop()
            stream.close()

    # ------------------------------------------------------------------
    # Public API (same contract as the old LocalTTS)
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Enqueue *text* for synthesis and playback."""
        if not text or not text.strip():
            return
        self.queue.put(text)

    def stop(self) -> None:
        """Clear the text queue and halt any active playback immediately."""
        with self.lock:
            self.is_speaking = False

        sd.stop()

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break



# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _speaker_device():
    """Return a valid speaker device index, or None for system default.

    If the configured index is out of range or the device no longer exists,
    log a warning, remove it from config, and return None so we fall back
    to the system default rather than failing on every utterance.
    """
    try:
        idx = config.get_config().get("speaker_device_index", None)
        if idx is None:
            return None

        devices = sd.query_devices()
        if idx >= len(devices):
            logger.warning(
                "[TTS] speaker_device_index %d is out of range (%d devices). "
                "Clearing config entry and using system default.",
                idx, len(devices),
            )
            config.get_config().pop("speaker_device_index", None)
            return None

        info = devices[idx]
        if info["max_output_channels"] < 1:
            logger.warning(
                "[TTS] Device %d ('%s') has no output channels. "
                "Clearing config entry and using system default.",
                idx, info["name"],
            )
            config.get_config().pop("speaker_device_index", None)
            return None

        return idx
    except Exception as exc:
        logger.warning("[TTS] Could not validate speaker device: %s", exc)
        return None
