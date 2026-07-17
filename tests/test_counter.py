import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, mock_open

from src.tools.counter import make_safe, main


# ── make_safe tests ──────────────────────────────────────────────

class TestMakeSafe:
    def test_plain_ascii(self):
        assert make_safe("hello") == "hello"

    def test_newline(self):
        assert make_safe("\n") == "\\n\n"

    def test_tab(self):
        assert make_safe("\t") == "\\t"

    def test_carriage_return(self):
        assert make_safe("\r") == "\\r"

    def test_control_char_below_32(self):
        # BEL (0x07)
        assert make_safe("\x07") == "\\x07"
        # SOH (0x01)
        assert make_safe("\x01") == "\\x01"

    def test_del_char_127(self):
        assert make_safe("\x7f") == "\\x7f"

    def test_mixed_content(self):
        result = make_safe("a\tb\nc\x01")
        # \n becomes literal backslash-n + actual newline
        assert result == "a\\tb\\n\nc\\x01"

    def test_empty_string(self):
        assert make_safe("") == ""

    def test_unicode_passes_through(self):
        assert make_safe("café") == "café"
        assert make_safe("🚀") == "🚀"

    def test_multiple_control_chars(self):
        result = make_safe("\x00\x01\x02")
        assert result == "\\x00\\x01\\x02"

    def test_preserves_printable_chars(self):
        text = "ABCxyz012 !@#"
        assert make_safe(text) == text


# ── CLI argument parsing tests ───────────────────────────────────

def _make_tokenizer_mock(token_ids=None, vocab=None, special_tokens=None):
    """Build a mock Tokenizer with controllable return values."""
    tok = MagicMock()
    tok.encode.return_value = token_ids if token_ids is not None else []
    tok.vocab = vocab if vocab is not None else {}
    tok.special_tokens = special_tokens if special_tokens is not None else {"<|end|>": 256}
    return tok


class TestMainArgumentParsing:
    """Verify CLI argument parsing and early exits."""

    def test_missing_tokenizer_file(self, capsys):
        """--tokenizer points to non-existent file → exit(1)."""
        with pytest.raises(SystemExit, match="1"):
            main(["--tokenizer", "/nonexistent/tokenizer.json", "--text", "hello"])
        captured = capsys.readouterr()
        assert "Error: Tokenizer file not found" in captured.err

    @patch("src.tools.counter.Tokenizer")
    def test_tokenizer_load_error(self, MockTokenizer, capsys, tmp_path):
        """Tokenizer.load() raises → exit(1)."""
        tok = MagicMock()
        tok.load.side_effect = RuntimeError("bad file")
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        tokenizer_path = tokenizer_path  # just for clarity
        # Create the file so exists() passes
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        with pytest.raises(SystemExit, match="1"):
            main(["--tokenizer", tokenizer_path, "--text", "hello"])
        captured = capsys.readouterr()
        assert "Error loading tokenizer" in captured.err

    def test_text_and_file_mutually_exclusive(self, capsys):
        """Passing both --text and --file → argparse error."""
        with pytest.raises(SystemExit):
            main(["--tokenizer", "x.json", "--text", "a", "--file", "b.txt"])

    def test_no_text_or_file(self, capsys):
        """Neither --text nor --file → argparse error."""
        with pytest.raises(SystemExit):
            main(["--tokenizer", "x.json"])

    @patch("src.tools.counter.Tokenizer")
    def test_file_not_found(self, MockTokenizer, capsys, tmp_path):
        """--file points to missing file → exit(1)."""
        tok = MagicMock()
        tok.load.return_value = None
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        with pytest.raises(SystemExit, match="1"):
            main(["--tokenizer", tokenizer_path, "--file", "/nonexistent/file.txt"])
        captured = capsys.readouterr()
        assert "Error: File not found" in captured.err


# ── Token counting & output tests ────────────────────────────────

class TestMainOutput:
    """Test the happy-path output of the main function."""

    @patch("src.tools.counter.Tokenizer")
    def test_basic_text_output(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[72, 101, 108])
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "abc"])
        captured = capsys.readouterr()

        assert "Token Visualization" in captured.out
        assert "Total Tokens:       3" in captured.out
        assert "Total Bytes:        3" in captured.out
        assert "Token IDs:          [72, 101, 108]" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_compression_ratio(self, MockTokenizer, capsys, tmp_path):
        """Verify compression ratio = bytes / tokens."""
        tok = _make_tokenizer_mock(token_ids=[1, 2, 3, 4])
        tok.vocab = {
            1: b"h",
            2: b"e",
            3: b"l",
            4: b"lo",
        }
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        text = "hello"  # 5 bytes, 4 tokens → 1.25x
        main(["--tokenizer", tokenizer_path, "--text", text])
        captured = capsys.readouterr()
        assert "1.25x" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_vocab_coverage(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(
            token_ids=[10, 20, 10],
            vocab={10: b"a", 20: b"b", 30: b"c", 40: b"d"},
        )
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "aba"])
        captured = capsys.readouterr()
        # unique=2, vocab=4 → 50%
        assert "50.00%" in captured.out
        assert "2 unique tokens" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_empty_text(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[])
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", ""])
        captured = capsys.readouterr()
        assert "Total Tokens:       0" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_empty_text_zero_division(self, MockTokenizer, capsys, tmp_path):
        """Compression ratio for 0 tokens should be 0.0, not ZeroDivisionError."""
        tok = _make_tokenizer_mock(token_ids=[])
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", ""])
        captured = capsys.readouterr()
        assert "Compression Ratio:  0.00x" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_zero_vocab_coverage(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[999], vocab={})
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "x"])
        captured = capsys.readouterr()
        assert "0.00%" in captured.out


# ── ANSI color output tests ──────────────────────────────────────

class TestAnsiOutput:
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    @patch("src.tools.counter.Tokenizer")
    def test_alternating_colors(self, MockTokenizer, capsys, tmp_path):
        """Tokens should alternate cyan/yellow."""
        tok = _make_tokenizer_mock(token_ids=[1, 2, 3])
        tok.vocab = {1: b"a", 2: b"b", 3: b"c"}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "abc"])
        captured = capsys.readouterr()
        # Token 0 → cyan, token 1 → yellow, token 2 → cyan
        assert f"{self.CYAN}a{self.RESET}" in captured.out
        assert f"{self.YELLOW}b{self.RESET}" in captured.out
        assert f"{self.CYAN}c{self.RESET}" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_special_token_shows_name(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(
            token_ids=[256],
            special_tokens={"<|end|>": 256},
        )
        tok.vocab = {256: b"<|end|>"}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "hi"])
        captured = capsys.readouterr()
        assert "<|end|>" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_unknown_token_id(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[9999])
        tok.vocab = {}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "x"])
        captured = capsys.readouterr()
        assert "[UNK:9999]" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_hex_fallback_for_invalid_utf8(self, MockTokenizer, capsys, tmp_path):
        """When vocab bytes aren't valid UTF-8, show hex repr."""
        tok = _make_tokenizer_mock(token_ids=[100])
        tok.vocab = {100: b"\xff\xfe"}  # invalid UTF-8
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "x"])
        captured = capsys.readouterr()
        assert "\\xff\\xfe" in captured.out


# ── File reading tests ───────────────────────────────────────────

class TestMainFileInput:
    @patch("src.tools.counter.Tokenizer")
    def test_read_from_file(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[1, 2])
        tok.vocab = {1: b"h", 2: b"i"}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        text_path = str(tmp_path / "input.txt")

        with open(tokenizer_path, "w") as f:
            f.write("{}")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("hi")

        main(["--tokenizer", tokenizer_path, "--file", text_path])
        captured = capsys.readouterr()
        assert "Total Tokens:       2" in captured.out

    @patch("src.tools.counter.Tokenizer")
    def test_utf8_file(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[10, 20])
        tok.vocab = {10: "é".encode(), 20: "ñ".encode()}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        text_path = str(tmp_path / "input.txt")

        with open(tokenizer_path, "w") as f:
            f.write("{}")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("éñ")

        main(["--tokenizer", tokenizer_path, "--file", text_path])
        captured = capsys.readouterr()
        assert "Total Tokens:       2" in captured.out


# ── Edge cases for make_safe ─────────────────────────────────────

class TestMakeSafeEdgeCases:
    def test_all_ascii_printable(self):
        text = "".join(chr(i) for i in range(32, 127))
        assert make_safe(text) == text

    def test_all_control_chars(self):
        for i in range(0, 32):
            result = make_safe(chr(i))
            if chr(i) == "\n":
                assert result == "\\n\n"
            elif chr(i) == "\t":
                assert result == "\\t"
            elif chr(i) == "\r":
                assert result == "\\r"
            else:
                assert result == f"\\x{i:02x}"

    def test_del_char(self):
        assert make_safe("\x7f") == "\\x7f"

    def test_high_codepoints(self):
        # Should pass through unchanged
        assert make_safe("\u4e16") == "\u4e16"  # CJK
        assert make_safe("\U0001f600") == "\U0001f600"  # emoji


# ── Tokenizer encode exception handling ──────────────────────────

class TestMainErrorHandling:
    @patch("src.tools.counter.Tokenizer")
    def test_encode_exception(self, MockTokenizer, capsys, tmp_path):
        tok = _make_tokenizer_mock()
        tok.encode.side_effect = RuntimeError("tokenization failed")
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        with pytest.raises(SystemExit, match="1"):
            main(["--tokenizer", tokenizer_path, "--text", "hello"])
        captured = capsys.readouterr()
        assert "Error tokenizing text" in captured.err


# ── Windows ANSI compatibility ───────────────────────────────────

class TestWindowsAnsi:
    @patch("os.system")
    @patch("os.name", "nt")
    @patch("src.tools.counter.Tokenizer")
    def test_windows_enables_ansi(self, MockTokenizer, mock_system, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[65])
        tok.vocab = {65: b"A"}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "A"])
        mock_system.assert_called_once_with("")

    @patch("os.system")
    @patch("os.name", "posix")
    @patch("src.tools.counter.Tokenizer")
    def test_non_windows_no_os_system(self, MockTokenizer, mock_system, capsys, tmp_path):
        tok = _make_tokenizer_mock(token_ids=[65])
        tok.vocab = {65: b"A"}
        MockTokenizer.return_value = tok

        tokenizer_path = str(tmp_path / "tok.json")
        with open(tokenizer_path, "w") as f:
            f.write("{}")

        main(["--tokenizer", tokenizer_path, "--text", "A"])
        mock_system.assert_not_called()
