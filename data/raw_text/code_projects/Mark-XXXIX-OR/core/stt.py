import logging
import numpy as np
from faster_whisper import WhisperModel
import config

logger = logging.getLogger("core.stt")


class LocalSTT:
    def __init__(self):
        # Initialize WhisperModel from config
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")

    def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe 16kHz mono 16-bit PCM audio data.
        Falls back to cloud STT (Gemini) on local failure.
        
        Args:
            audio_data (bytes): Raw PCM audio bytes.
            
        Returns:
            str: Transcribed text.
        """
        if not audio_data:
            return ""

        try:
            # Convert 16-bit PCM bytes to float32 numpy array normalized to [-1.0, 1.0]
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe audio array
            # Whisper model expects 16kHz audio, which matches the input description.
            segments, _ = self.model.transcribe(audio_np, beam_size=5)

            # Combine segments into a single string
            transcription = " ".join(segment.text for segment in segments).strip()
            return transcription
        except Exception as e:
            logger.error(f"LocalSTT: transcription failed: {e}")
            return self._cloud_fallback(audio_data)

    def _cloud_fallback(self, audio_data: bytes) -> str:
        """Convert PCM→WAV in-memory and send to Gemini for transcription."""
        import io
        import wave
        from google import genai
        from google.genai import types

        try:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            api_key = config.get_config()["gemini_api_key"]
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(
                        data=buf.getvalue(),
                        mime_type="audio/wav"
                    ),
                    "Transcribe this audio. Return ONLY the text."
                ]
            )
            result = resp.text.strip()
            logger.info(f"LocalSTT: cloud fallback transcription successful")
            return result
        except Exception as e:
            logger.error(f"LocalSTT: cloud fallback also failed: {e}")
            return ""
