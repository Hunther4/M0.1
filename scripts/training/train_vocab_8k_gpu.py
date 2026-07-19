import os
import sys
import json
import time
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate
from src.training.checkpoint import config_to_dict, save_checkpoint
from src.training.loop import train
from src.training.eval import evaluate_val_loss
from src.training.datasets import JsonlDataset
from src.training.setup import setup_device, setup_stdout
from torch.utils.data import DataLoader
from torch.optim import AdamW

def evaluate_val_ppl(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in val_loader:
            if steps >= 30:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
    model.train()
    avg_loss = total_loss / max(steps, 1)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()

def check_json_validity(text):
    tag = "<|tool_call|>"
    if tag not in text:
        return False
    try:
        start_idx = text.find(tag) + len(tag)
        json_str = text[start_idx:].strip()
        end_idx = json_str.rfind("}")
        if end_idx != -1:
            json_str = json_str[:end_idx+1]
        json.loads(json_str)
        return True
    except Exception:
        return False

def evaluate_tool_accuracy(text, expected_tool):
    tag = "<|tool_call|>"
    if tag not in text:
        return False
    try:
        start_idx = text.find(tag) + len(tag)
        json_str = text[start_idx:].strip()
        end_idx = json_str.rfind("}")
        if end_idx != -1:
            json_str = json_str[:end_idx+1]
        data = json.loads(json_str)
        return data.get("name") == expected_tool
    except Exception:
        return False

def run_needle_test(model, tokenizer, device):
    """Embed a secret fact (needle) deep in a context (haystack) and check retrieval."""
    needle = "El código secreto de acceso es X-9060."
    # Build a small haystack
    haystack_sentences = [
        "El proyecto M0.1 es una arquitectura experimental.",
        "Los programadores están entrenando el modelo en GPU.",
        "La base de datos contiene múltiples shards de conversaciones.",
        "Se utilizan expertos compartidos para conocimiento general.",
        "La atención híbrida utiliza CSA y HCA para el contexto largo."
    ]
    # Place needle in the middle
    full_text = "\n".join(haystack_sentences[:3]) + "\n" + needle + "\n" + "\n".join(haystack_sentences[3:])
    prompt = f"<|system|>\nLee el siguiente texto:\n{full_text}\n<|user|>\n¿Cuál es el código secreto de acceso?\n<|assistant|>\n"
    
    generated = generate(model, tokenizer, prompt, max_gen_len=20, temperature=0.3, device=device)
    # Check if "X-9060" is in the answer
    return "X-9060" in generated, generated

def run_story_test(model, tokenizer, device):
    """Prompt model to write a short story and check orthographic coherence."""
    prompt = "<|user|>\nEscribe una mini historia de 3 oraciones en español sobre un robot.\n<|assistant|>\n"
    generated = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.5, device=device)
    
    # Calculate ratio of valid alphabetic chars
    words = generated.split()
    if not words:
        return False, generated, 0.0
        
    valid_words = 0
    # A word is relatively valid if it has alphabetic chars or common punctuation
    for w in words:
        clean = w.strip(".,;:!?\"'")
        if clean.isalpha() and len(clean) > 1:
            valid_words += 1
            
    ratio = valid_words / len(words)
    # If more than 70% of generated tokens form recognizable alphabetic words
    return ratio >= 0.70, generated, ratio

def main():
    setup_stdout()
    
    print("=" * 60)
    print("       M0.1-Lite: 8K Vocab Tokenizer + 5800 Step GPU Training")
    print("=" * 60)
    
    device = setup_device()
    
    # 1. Train Tokenizer to 8192 vocabulary (Shakespeare + Synthetic Spanish Conversations)
    print("\nTraining 8K Tokenizer on combined corpus...")
    tokenizer = Tokenizer()
    
    # Combine texts
    text_corpus = ""
    # Add Shakespeare
    if os.path.exists("data/tinyshakespeare.txt"):
        text_corpus += open("data/tinyshakespeare.txt", encoding="utf-8").read()[:500000]
    # Add Shard 0000
    shard_path = "D:/Proyectos/M0.2/data/corpus/synthetic/shard_0000.jsonl"
    if os.path.exists(shard_path):
        with open(shard_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 500:
                    break
                data = json.loads(line)
                text_corpus += f"\n{data['system']}\n{data['conversation']}"
                
    tokenizer.train(text_corpus, 8192, show_progress=False)
    tokenizer.save("data/tokenizer_8k.json")
    vocab_size = len(tokenizer.vocab)
    print(f"BPE Tokenizer trained and saved. Vocab size: {vocab_size}")
    
    # 2. Config & Initialize model
    config = M01Config(
        vocab_size=vocab_size,
        context_length=256,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=4,
        num_experts=4,
        num_shared_experts=2,
        moe_top_k=2,
        use_hybrid_attention=True,
        local_window_size=16
    )
    model = TransformerLM(config).to(device)
    print(f"Model parameterized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")
    
    # 3. Load Datasets
    shards_dir = "D:/Proyectos/M0.2/data/corpus/synthetic"
    train_shards = [os.path.join(shards_dir, f"shard_{i:04d}.jsonl") for i in range(5)]
    train_dataset = JsonlDataset(tokenizer, train_shards, seq_len=config.context_length, max_lines_per_shard=800)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    val_shard = [os.path.join(shards_dir, "shard_0010.jsonl")]
    val_dataset = JsonlDataset(tokenizer, val_shard, seq_len=config.context_length, max_lines_per_shard=300)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training (5800 steps)
    model.train()
    steps = 5800
    
    print(f"\nTraining on GPU for {steps} steps...")
    
    train_result = train(model, train_loader, optimizer, criterion, steps, device, log_interval=500, val_loader=val_loader)
    
    print(f"\nGPU Training completed in {train_result['elapsed']:.2f} seconds!")
    
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path_final = "checkpoints/vocab_8k_final.pt"
    save_checkpoint(model, config, checkpoint_path_final)
    print(f"8K vocab checkpoint saved to {checkpoint_path_final}\n")
    
    # 5. Run 5-test Evaluation suite
    print("=" * 60)
    print("         M0.1-Lite: Expanded 5-Test Validation Suite")
    print("=" * 60)
    model.eval()
    
    # Test 1: PPL
    val_loss, val_ppl = evaluate_val_ppl(model, val_loader, device, criterion)
    print(f"Test 1: Validation Perplexity (PPL): {val_ppl:.4f} (Loss: {val_loss:.4f})")
    
    # Prompts for JSON & Tool test
    eval_prompts = [
        ("<|user|>\n¿Qué hay en tests/test_main.py?\n<|tool_call|>", "read_file"),
        ("<|user|>\nNecesito escribir cmd/app/main.go con un hello world\n<|tool_call|>", "write_file"),
        ("<|user|>\nLista el contenido del directorio src/\n<|tool_call|>", "list_dir"),
        ("<|user|>\nEjecuta pytest en el terminal\n<|tool_call|>", "run_tests"),
        ("<|user|>\nBusca informacion sobre el clima de hoy en la web\n<|tool_call|>", "web_search")
    ]
    
    json_passes = 0
    tool_passes = 0
    for prompt, expected_tool in eval_prompts:
        generated = generate(model, tokenizer, prompt, max_gen_len=40, temperature=0.3, device=device)
        if check_json_validity(generated):
            json_passes += 1
            if evaluate_tool_accuracy(generated, expected_tool):
                tool_passes += 1
                
    json_rate = (json_passes / len(eval_prompts)) * 100
    tool_rate = (tool_passes / len(eval_prompts)) * 100
    
    # Test 2: JSON Syntax
    print(f"Test 2: JSON Syntax Validity (Pass@1): {json_rate:.1f}% ({json_passes}/{len(eval_prompts)})")
    
    # Test 3: Tool routing
    print(f"Test 3: Tool Routing Gating Accuracy: {tool_rate:.1f}% ({tool_passes}/{len(eval_prompts)})")
    
    # Test 4: NIAH
    niah_success, niah_out = run_needle_test(model, tokenizer, device)
    print(f"Test 4: Needle in a Haystack (NIAH) Retrieval: {'PASS' if niah_success else 'FAIL'}")
    print(f"  -> Generated: {niah_out.replace(chr(10), ' ')}")
    
    # Test 5: Mini story orthography
    story_success, story_out, word_ratio = run_story_test(model, tokenizer, device)
    print(f"Test 5: Story Spelling/Orthography Rate: {'PASS' if story_success else 'FAIL'} ({word_ratio*100:.1f}% real words)")
    print(f"  -> Generated: {story_out.replace(chr(10), ' ')}")
    
    print("\n" + "=" * 60)
    print("All evaluations complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
