import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from src.dataset.prep import download_tiny_shakespeare, ingest_local_file, main

def test_download_tiny_shakespeare():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-length': '12'}
        mock_response.iter_content.return_value = [b"hello ", b"world!"]
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "shakespeare.txt")
            download_tiny_shakespeare(out_file)

            assert os.path.exists(out_file)
            with open(out_file, "rb") as f:
                content = f.read()
            assert content == b"hello world!"

def test_ingest_local_file_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "input.txt")
        out_file = os.path.join(tmpdir, "output.txt")
        
        with open(in_file, "w", encoding="utf-8") as f:
            f.write("Some standard UTF-8 text with emojis: 🚀")

        ingest_local_file(in_file, out_file)

        assert os.path.exists(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == "Some standard UTF-8 text with emojis: 🚀"

def test_ingest_local_file_invalid():
    with tempfile.TemporaryDirectory() as tmpdir:
        in_file = os.path.join(tmpdir, "input.bin")
        out_file = os.path.join(tmpdir, "output.txt")

        with open(in_file, "wb") as f:
            f.write(b"\xff\xfe\xfd\xfc")

        with pytest.raises(ValueError, match="Input file is not valid UTF-8"):
            ingest_local_file(in_file, out_file)

def test_prep_cli_download():
    with patch("src.dataset.prep.download_tiny_shakespeare") as mock_download:
        main(["--download", "--output", "data/dummy.txt"])
        mock_download.assert_called_once_with("data/dummy.txt")

def test_prep_cli_ingest():
    with patch("src.dataset.prep.ingest_local_file") as mock_ingest:
        main(["--input", "in.txt", "--output", "out.txt"])
        mock_ingest.assert_called_once_with("in.txt", "out.txt")

def test_prep_cli_missing_output():
    with pytest.raises(SystemExit):
        main(["--input", "in.txt"])
