"""
Unit tests for KokoroTTS and KokoroLifecycleManager.

Covers:
  - KokoroTTS.speak() with empty/whitespace text rejection
  - KokoroTTS.stop() — sets is_speaking false, clears queue
  - KokoroTTS.stop_worker() — stops thread, clears queue
  - REST payload verification (correct JSON sent to /v1/audio/speech)
  - KokoroLifecycleManager.is_alive() with no server
  - KokoroLifecycleManager.start() with non-existent command
  - KokoroLifecycleManager.stop() with no process managed
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from core.kokoro_manager import KokoroLifecycleManager
from core.tts import KokoroTTS


# =============================================================================
# KokoroTTS — Unit Tests
# =============================================================================


@pytest.fixture
def tts():
    """Create a KokoroTTS instance with sounddevice mocked.

    The worker thread starts during init but idles on the empty queue.
    Each test gets a fresh instance; the fixture tears down the worker.
    """
    with patch("core.tts.sd") as mock_sd:
        mock_sd.OutputStream.return_value = MagicMock()
        instance = KokoroTTS(base_url="http://test:9999", voice="ef_dora")
        yield instance
        instance.stop_worker()


class TestKokoroTTS:
    """Tests for the REST-based KokoroTTS client."""

    # -- speak() text validation -------------------------------------------

    def test_speak_empty_string_rejected(self, tts):
        """Empty string should NOT be enqueued."""
        size_before = tts.queue.qsize()
        tts.speak("")
        assert tts.queue.qsize() == size_before

    def test_speak_whitespace_rejected(self, tts):
        """Whitespace-only text should NOT be enqueued."""
        size_before = tts.queue.qsize()
        tts.speak("   \t\n  ")
        assert tts.queue.qsize() == size_before

    def test_speak_valid_text_enqueued(self, tts):
        """Non-empty text SHOULD be enqueued."""
        tts.speak("Hello world")
        assert tts.queue.qsize() >= 1
        item = tts.queue.get_nowait()
        assert item == "Hello world"

    # -- stop() ------------------------------------------------------------

    def test_stop_sets_is_speaking_false(self, tts):
        """stop() must clear the is_speaking flag."""
        tts.is_speaking = True
        tts.stop()
        assert tts.is_speaking is False

    def test_stop_clears_queue(self, tts):
        """stop() must drain all pending items from the queue."""
        tts.queue.put("msg1")
        tts.queue.put("msg2")
        tts.stop()
        assert tts.queue.empty()

    # -- stop_worker() -----------------------------------------------------

    def test_stop_worker_clears_running_flag(self, tts):
        """stop_worker() must set the running flag to False."""
        assert tts.running is True
        tts.stop_worker()
        assert tts.running is False

    def test_stop_worker_thread_terminates(self, tts):
        """Worker thread should not be alive after stop_worker()."""
        tts.stop_worker()
        assert tts.worker_thread is not None
        assert not tts.worker_thread.is_alive()

    def test_stop_worker_puts_none_sentinel(self, tts):
        """stop_worker() must place a None sentinel into the queue.

        We spy on queue.put to verify the None is enqueued.
        """
        with patch.object(tts.queue, "put", wraps=tts.queue.put) as spy_put:
            tts.stop_worker()
            spy_put.assert_any_call(None)

    # -- REST payload verification -----------------------------------------

    def test_correct_payload_sent_to_kokoro(self):
        """Verify the exact JSON payload sent to /v1/audio/speech."""
        with patch("core.tts.requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.content = b"\x00\x00" * 100

            with patch("core.tts.sd") as mock_sd:
                mock_sd.OutputStream.return_value = MagicMock()
                tts = KokoroTTS(base_url="http://test:9999", voice="ef_dora")

            tts.speak("xyz")
            # Wait for the worker to process the item before stopping
            tts.queue.join()
            tts.stop_worker()

            mock_post.assert_called_once_with(
                "http://test:9999/v1/audio/speech",
                json={
                    "model": "kokoro",
                    "input": "xyz",
                    "voice": "ef_dora",
                    "response_format": "pcm",
                    "speed": 1.0,
                },
                timeout=30,
            )


# =============================================================================
# KokoroLifecycleManager — Unit Tests
# =============================================================================


class TestKokoroLifecycleManager:
    """Tests for the Kokoro-FastAPI subprocess lifecycle manager."""

    def test_is_alive_when_no_server(self):
        """is_alive() returns False when the voices endpoint is unreachable."""
        with patch("core.kokoro_manager.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Connection refused")

            mgr = KokoroLifecycleManager(
                host="127.0.0.1", port=19999, launch_cmd=None
            )
            assert mgr.is_alive() is False

    def test_start_with_nonexistent_command(self):
        """start() with a bogus command must return False, not crash."""
        with patch.object(
            KokoroLifecycleManager, "is_alive", return_value=False
        ):
            mgr = KokoroLifecycleManager(
                host="127.0.0.1",
                port=19999,
                launch_cmd=["this-command-does-not-exist-hopefully"],
            )
            result = mgr.start()
            assert result is False

    def test_stop_when_no_process_managed(self):
        """stop() is a no-op when _process is None (external service)."""
        mgr = KokoroLifecycleManager(
            host="127.0.0.1", port=19999, launch_cmd=None
        )
        mgr.stop()  # must not raise
        assert mgr._process is None

    def test_stop_when_process_already_exited(self):
        """stop() handles a finished process without error."""
        mgr = KokoroLifecycleManager(
            host="127.0.0.1", port=19999, launch_cmd=["cmd", "/c", "echo"]
        )
        with patch("core.kokoro_manager.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0  # already exited
            mock_popen.return_value = mock_proc

            mgr._process = mock_proc
            mgr.stop()  # must not raise
            assert mgr._process is None
