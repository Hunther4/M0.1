import os
import sys
import json
import torch
import torch.nn as nn
from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer
from src.inference.generate import generate

def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")

def check_json_validity(text):
    """Check if the generated text after <|tool_call|> contains a valid JSON."""
    tag = "<|tool_call|>"
    if tag not in text:
        return False
    try:
        # Extract text after tag
        start_idx = text.find(tag) + len(tag)
        json_str = text[start_idx:].strip()
        # Find closing bracket of JSON
        end_idx = json_str.rfind("}")
        if end_idx != -1:
            json_str = json_str[:end_idx+1]
        json.loads(json_str)
        return True
    except Exception:
        return False

def evaluate_tool_accuracy(text, expected_tool):
    """Check if the correct tool name was predicted in the tool call."""
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

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    print("=" * 60)
    print("         M0.1-Lite: Validation & Capability Suite")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluations on: {device}\n")
    
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer.json")
    
    checkpoint_path = "checkpoints/phase3_final.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
        
    print(f"Loaded checkpoint: {checkpoint_path}")
    ckpt_config = checkpoint["config"]
    config = M01Config(
        vocab_size=ckpt_config["vocab_size"],
        context_length=ckpt_config["context_length"],
        d_model=ckpt_config["d_model"],
        n_heads=ckpt_config["n_heads"],
        d_ff=ckpt_config["d_ff"],
        n_layers=ckpt_config["n_layers"],
        num_experts=ckpt_config["num_experts"],
        num_shared_experts=ckpt_config["num_shared_experts"],
        moe_top_k=ckpt_config["moe_top_k"],
        use_hybrid_attention=ckpt_config["use_hybrid_attention"],
        local_window_size=ckpt_config["local_window_size"]
    )
    model = TransformerLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # -------------------------------------------------------------
    # METRICA 1: Perplejidad (PPL) en Validación
    # -------------------------------------------------------------
    print("Evaluating Metric 1: Perplexity (PPL) on unseen validation data...")
    val_shard_path = "D:/Proyectos/M0.2/data/corpus/synthetic/shard_0010.jsonl"
    
    total_loss = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    
    if os.path.exists(val_shard_path):
        with open(val_shard_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 50: # Check 50 lines to keep it fast
                    break
                try:
                    data = json.loads(line)
                    text = f"{data['system']}\n{data['conversation']}"
                    tokens = tokenizer.encode(text)
                    if len(tokens) < 10:
                        continue
                    x = torch.tensor([tokens[:-1]], device=device)
                    y = torch.tensor([tokens[1:]], device=device)
                    with torch.no_grad():
                        logits = model(x)
                        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                        total_loss += loss.item()
                        count += 1
                except Exception:
                    continue
        avg_loss = total_loss / max(count, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        print(f"  -> Validation Loss: {avg_loss:.4f}")
        print(f"  -> Validation Perplexity (PPL): {ppl:.4f}\n")
    else:
        print("  -> Skip PPL (Validation file not found).\n")
        
    # -------------------------------------------------------------
    # METRICA 2: Sintaxis JSON (Pass@1 Rate)
    # -------------------------------------------------------------
    print("Evaluating Metric 2: JSON Syntax Validity (Pass@1)...")
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
        generated = generate(model, tokenizer, prompt, max_gen_len=50, temperature=0.3, device=device)
        # Check if valid JSON
        is_valid_json = check_json_validity(generated)
        if is_valid_json:
            json_passes += 1
            # Check if correct tool name was predicted
            is_correct_tool = evaluate_tool_accuracy(generated, expected_tool)
            if is_correct_tool:
                tool_passes += 1
                
    json_pass_rate = (json_passes / len(eval_prompts)) * 100
    tool_accuracy = (tool_passes / len(eval_prompts)) * 100
    
    print(f"  -> JSON Syntax Pass@1: {json_pass_rate:.1f}% ({json_passes}/{len(eval_prompts)})")
    
    # -------------------------------------------------------------
    # METRICA 3: Tool Routing Accuracy (Exact Tool Name Match)
    # -------------------------------------------------------------
    print("\nEvaluating Metric 3: Tool Name Gating Accuracy...")
    print(f"  -> Tool Call Alignment Accuracy: {tool_accuracy:.1f}% ({tool_passes}/{len(eval_prompts)})")
    
    print("\n" + "=" * 60)
    print("Evaluation execution finished!")
    print("=" * 60)

if __name__ == "__main__":
    main()
