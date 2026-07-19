import torch
import pytest
from src.inference.generate import generate
from src.tokenizer.bpe import Tokenizer
from src.model.lm import TransformerLM
from src.transformer.config import M01Config
from src.inference.profiling import profile_inference

def test_inference_profiling_output():
    # Setup
    config = M01Config(n_layers=2, n_heads=4, d_model=32, vocab_size=258, context_length=128)
    model = TransformerLM(config)
    tokenizer = Tokenizer()
    tokenizer.train("Hello", vocab_size=258)
    prompt = "Hello"
    
    # Execution
    tps, mem = profile_inference(model, tokenizer, prompt, max_gen_len=10)
    
    # Assertions
    assert isinstance(tps, float)
    assert isinstance(mem, float)
    assert tps > 0

def test_inference_profiling_output_longer():
    # Setup
    config = M01Config(n_layers=2, n_heads=4, d_model=32, vocab_size=258, context_length=128)
    model = TransformerLM(config)
    tokenizer = Tokenizer()
    tokenizer.train("Hello world", vocab_size=258)
    prompt = "Hello world"
    
    # Execution
    tps, mem = profile_inference(model, tokenizer, prompt, max_gen_len=20)
    
    # Assertions
    assert isinstance(tps, float)
    assert isinstance(mem, float)
    assert tps > 0
