"""Integration tests for evaluation suite."""
import json
import tempfile
from pathlib import Path

import pytest
import torch
from unittest.mock import MagicMock, patch

from src.eval.utils import save_results, setup_logging, get_timestamp
from src.eval.evaluate import load_checkpoint


class TestEvaluateIntegration:
    """Integration tests for evaluate.py workflow."""
    
    def test_save_results_creates_file(self):
        """Test that save_results creates a JSON file."""
        results = {
            "metrics": {"perplexity": 15.5},
            "qa": {}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp dir so artifacts/evals is created there
            original_cwd = Path.cwd()
            try:
                Path(tmpdir).joinpath("artifacts/evals").mkdir(parents=True, exist_ok=True)
                
                # Save results
                output_path = save_results(results, "test_checkpoint.pt")
                
                assert output_path.exists()
                
                # Verify content
                with open(output_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                
                assert "timestamp" in loaded
                assert loaded["checkpoint"] == "test_checkpoint.pt"
                assert loaded["metrics"]["perplexity"] == 15.5
            finally:
                pass  # Don't change cwd back since temp dir will be deleted
    
    def test_save_results_timestamp_format(self):
        """Test that timestamp is in correct ISO format."""
        results = {"metrics": {}}
        
        output_path = save_results(results, "test.pt")
        
        with open(output_path, encoding="utf-8") as f:
            loaded = json.load(f)
        
        # Should be ISO format ending with Z
        assert loaded["timestamp"].endswith("Z")
    
    def test_evaluate_cli_help(self):
        """Test that evaluate.py --help works."""
        import subprocess
        result = subprocess.run(
            ["python", "src/eval/evaluate.py", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--checkpoint" in result.stdout
        assert "--coherence" in result.stdout
        assert "--niah" in result.stdout
    
    def test_evaluate_cli_missing_checkpoint(self):
        """Test that evaluate.py fails gracefully without checkpoint."""
        import subprocess
        result = subprocess.run(
            ["python", "src/eval/evaluate.py"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode != 0
        assert "error: the following arguments are required: --checkpoint" in result.stderr


class TestGenerateReportIntegration:
    """Integration tests for generate_report.py."""
    
    def test_generate_report_help(self):
        """Test that generate_report.py --help works."""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/generate_report.py", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--eval" in result.stdout
    
    def test_generate_report_with_dummy_json(self):
        """Test report generation with dummy evaluation data."""
        import subprocess
        
        # Create temp eval JSON
        eval_data = {
            "timestamp": "2026-07-18T10:00:00Z",
            "checkpoint": "test_checkpoint.pt",
            "metrics": {"perplexity": 12.34},
            "qa": {
                "coherence": {"average_coherence": 15.2, "interval": 128},
                "niah": {"needle": "42", "accuracy": 0.95, "context_length": 512}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(eval_data, f)
            temp_json = f.name
        
        try:
            result = subprocess.run(
                ["python", "scripts/generate_report.py", "--eval", temp_json],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Evaluation Report" in result.stdout
            assert "12.34" in result.stdout
            assert "Coherence Test" in result.stdout
            assert "NIAH" in result.stdout
        finally:
            Path(temp_json).unlink()


class TestCompareIntegration:
    """Integration tests for compare.py."""
    
    def test_compare_help(self):
        """Test that compare.py --help works."""
        import subprocess
        result = subprocess.run(
            ["python", "scripts/compare.py", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "--json1" in result.stdout
        assert "--json2" in result.stdout
    
    def test_compare_two_evals(self):
        """Test comparing two evaluation files."""
        import subprocess
        
        eval1 = {
            "timestamp": "2026-07-18T10:00:00Z",
            "checkpoint": "ckpt1.pt",
            "metrics": {"perplexity": 15.0}
        }
        eval2 = {
            "timestamp": "2026-07-18T11:00:00Z",
            "checkpoint": "ckpt2.pt",
            "metrics": {"perplexity": 12.0}
        }
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(eval1, f)
            temp1 = f.name
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(eval2, f)
            temp2 = f.name
        
        try:
            result = subprocess.run(
                ["python", "scripts/compare.py", "--json1", temp1, "--json2", temp2],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0
            assert "Checkpoint Comparison" in result.stdout
            assert "ckpt1.pt" in result.stdout
            assert "ckpt2.pt" in result.stdout
            assert "15" in result.stdout  # perplexity from ckpt1
            assert "12" in result.stdout  # perplexity from ckpt2
        finally:
            Path(temp1).unlink()
            Path(temp2).unlink()