"""
Benchmark: ROCm vs CPU performance for M0.1 components.

Tests actual throughput (tokens/s) for the Fase 1 modules.
"""
import time
import pytest
import torch
from src.transformer.config import M01Config
from src.transformer.embeddings import TokenEmbedding
from src.transformer.rope import RotaryPositionalEmbedding
from src.transformer.attention import CausalSelfAttention
from src.transformer.feedforward import FeedForward

DEVICES = []
if torch.cuda.is_available():
    DEVICES.append(('rocm', 'cuda'))  # ROCm presents as CUDA in PyTorch
DEVICES.append(('cpu', 'cpu'))

MESSAGES = [
    "M0.1 is an educational transformer built from scratch",
    "No black boxes, no magic, just code",
    "Each component is designed to be readable and extensible",
    "Love the process, trust the fundamentals",
]


def run_and_measure(fn, *args, **kwargs):
    """Run a function and return (result, elapsed_time_seconds)."""
    # warmup
    for _ in range(3):
        fn(*args, **kwargs)

    # timed runs
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.perf_counter() - start
    return result, elapsed


class TestBenchmark:

    def setup_method(self):
        self.config = M01Config()

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_embeddings_forward(self, name, device):
        """Benchmark embedding lookup + projection."""
        dev = torch.device(device)
        emb = TokenEmbedding(self.config).to(dev)
        tokens = torch.randint(0, self.config.vocab_size, (4, 512)).to(dev)

        def run():
            embeddings = emb(tokens)  # (4, 512, 640) forward
            logits = emb.output_head(embeddings)  # tied projection
            return logits

        _, elapsed = run_and_measure(run)
        tokens_per_sec = 4 * 512 / elapsed
        print(f"  [{name}] Embeddings throughput: {tokens_per_sec:.0f} tok/s")
        assert tokens_per_sec > 0
        assert elapsed < 30, "Suspiciously slow"

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_attention_forward(self, name, device):
        """Benchmark self-attention forward pass."""
        dev = torch.device(device)
        config = M01Config(n_layers=1)
        attn = CausalSelfAttention(config).to(dev)
        x = torch.randn(2, 256, config.d_model).to(dev)

        def run():
            return attn(x)

        _, elapsed = run_and_measure(run)
        tokens_per_sec = 2 * 256 / elapsed
        print(f"  [{name}] Self-attention throughput: {tokens_per_sec:.0f} tok/s")
        assert tokens_per_sec > 0

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_full_forward_pass(self, name, device):
        """Benchmark a full forward pass: embed -> attention -> FFN."""
        dev = torch.device(device)
        config = M01Config(n_layers=2)
        emb = TokenEmbedding(config).to(dev)
        attn = CausalSelfAttention(config).to(dev)
        ff = FeedForward(config).to(dev)

        tokens = torch.randint(0, config.vocab_size, (2, 256)).to(dev)

        def run():
            x = emb(tokens)
            x = attn(x)
            x = ff(x)
            return x

        _, elapsed = run_and_measure(run)
        tokens_per_sec = 2 * 256 / elapsed
        print(f"  [{name}] Full forward (2 layers) throughput: {tokens_per_sec:.0f} tok/s")
        assert tokens_per_sec > 0
        assert elapsed < 30

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_backward_pass(self, name, device):
        """Benchmark a full forward+backward pass."""
        dev = torch.device(device)
        config = M01Config(n_layers=1)
        emb = TokenEmbedding(config).to(dev)
        attn = CausalSelfAttention(config).to(dev)
        ff = FeedForward(config).to(dev)
        loss_fn = torch.nn.MSELoss()

        tokens = torch.randint(0, config.vocab_size, (2, 128)).to(dev)
        targets = torch.randn(2, 128, config.d_model).to(dev)

        def run():
            x = emb(tokens)
            x = attn(x)
            x = ff(x)
            loss = loss_fn(x, targets)
            loss.backward()

        _, elapsed = run_and_measure(run)
        tokens_per_forward = 2 * 128
        total_tokens_equivalent = tokens_per_forward  # 1 forward + 1 backward
        throughput = total_tokens_equivalent / elapsed
        print(f"  [{name}] forward+backward throughput: {throughput:.0f} tok/s")
        assert throughput > 0

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_attention_scale(self, name, device):
        """Test attention at increasing sequence lengths."""
        dev = torch.device(device)
        config = M01Config(n_layers=1)
        attn = CausalSelfAttention(config).to(dev)

        for seq_len in [64, 128, 256, 512]:
            x = torch.randn(1, seq_len, config.d_model).to(dev)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()
            y = attn(x)  # single forward, no warmup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            # O(n^2) is expected for attention, so growth should be quadratic
            if name == 'cpu':
                print(f"  [{name}] seq_len={seq_len}: {elapsed:.4f}s")
            else:
                print(f"  [{name} {device}] seq_len={seq_len}: {elapsed * 1000:.2f} ms")

    @pytest.mark.parametrize("name,device", DEVICES)
    def test_vram_size(self, name, device):
        """Check available VRAM or RAM."""
        if device == 'cpu':
            import os
            # cross-platform approximation: use os
            print(f"  [{name}] RAM (from system): N/A (no cross-platform getter)")
            assert True
        else:
            vram_total = torch.cuda.get_device_properties(0).total_memory
            vram_used = torch.cuda.memory_allocated()
            vram_free = vram_total - vram_used
            print(f"  [{name}] VRAM total: {vram_total / 1024**3:.2f} GB")
            print(f"  [{name}] VRAM used: {vram_used / 1024**3:.4f} GB")
            print(f"  [{name}] VRAM free: {vram_free / 1024**3:.2f} GB")
            assert vram_total > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
