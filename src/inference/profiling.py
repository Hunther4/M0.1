import torch
import time
from src.inference.generate import generate
from src.tokenizer.bpe import Tokenizer
from src.model.lm import TransformerLM

def profile_inference(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_gen_len: int = 10) -> tuple[float, float]:
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
    else:
        start_mem = 0
        
    generate(model, tokenizer, prompt, max_gen_len=max_gen_len)
    
    end_time = time.perf_counter()
    if torch.cuda.is_available():
        end_mem = torch.cuda.max_memory_allocated()
    else:
        end_mem = 0
        
    tps = max_gen_len / (end_time - start_time)
    mem_diff = float(end_mem - start_mem)
    
    return tps, mem_diff
