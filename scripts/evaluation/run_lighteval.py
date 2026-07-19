import os
import sys
import json
import torch
import torch.nn.functional as F

# Insert current directory to import local src packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.transformer.config import M01Config
from src.model.lm import TransformerLM
from src.tokenizer.bpe import Tokenizer

# LightEval API Imports
from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached

class LightevalM01Adapter(LightevalModel):
    """
    Adapter class to bridge M0.1 MoE model with Hugging Face LightEval evaluation suite.
    """
    def __init__(self, config, env_config=None):
        super().__init__(config, env_config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load local BPE tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.load("data/tokenizers/tokenizer_final_8k.json")
        
        # Load 55M parameters model weights
        ckpt_path = "checkpoints/m01_hardened_final.pt"
        if not os.path.exists(ckpt_path):
            ckpt_path = "checkpoints/final_combined_8k.pt"
            
        print(f"[LightEval Adapter] Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        ckpt_config = checkpoint["config"]
        
        # Configure model
        self.model_config = M01Config(
            vocab_size=ckpt_config["vocab_size"],
            context_length=ckpt_config["context_length"],
            d_model=ckpt_config["d_model"],
            n_heads=ckpt_config["n_heads"],
            d_ff=ckpt_config["d_ff"],
            d_ff_shared=ckpt_config.get("d_ff_shared"),
            d_ff_routed=ckpt_config.get("d_ff_routed"),
            n_layers=ckpt_config["n_layers"],
            num_experts=ckpt_config["num_experts"],
            num_shared_experts=ckpt_config["num_shared_experts"],
            moe_top_k=ckpt_config["moe_top_k"],
            use_hybrid_attention=ckpt_config["use_hybrid_attention"],
            local_window_size=ckpt_config["local_window_size"]
        )
        
        self.model = TransformerLM(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        """
        Runs generative tasks (like HumanEval, MBPP, TruthfulQA generative).
        """
        responses = []
        for doc in docs:
            prompt = doc.query
            # Basic greedy generation
            tokens = self.tokenizer.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            
            generated_tokens = []
            max_len = 100 # Default ceiling for evaluation speed
            
            with torch.no_grad():
                for _ in range(max_len):
                    if x.shape[1] > self.model_config.context_length:
                        x = x[:, -self.model_config.context_length:]
                    
                    logits = self.model(x)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    
                    if next_token == self.tokenizer.vocab.get("<|endoftext|>"):
                        break
                        
                    generated_tokens.append(next_token)
                    x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=1)
                    
            generated_text = self.tokenizer.decode(generated_tokens)
            responses.append(ModelResponse(generated_text, logprob=0.0))
        return responses

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """
        Runs classification tasks (like MMLU, ARC, GPQA, TruthfulQA multiple choice).
        Computes conditional log-likelihood of target completion given query context.
        """
        responses = []
        for doc in docs:
            query = doc.query
            choices = doc.choices
            
            query_tokens = self.tokenizer.encode(query)
            choice_responses = []
            
            for choice in choices:
                choice_tokens = self.tokenizer.encode(choice)
                full_tokens = query_tokens + choice_tokens
                
                x = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=self.device)
                targets = torch.tensor([full_tokens[1:]], dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    logits = self.model(x) # (1, seq_len, vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Gather the log-probabilities of the target tokens
                    target_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                    
                    # Conditionally select only the portion corresponding to the choice tokens
                    choice_len = len(choice_tokens)
                    choice_log_probs = target_log_probs[:, -choice_len:]
                    sum_log_prob = choice_log_probs.sum().item()
                    
                choice_responses.append(sum_log_prob)
                
            # Select the option with the highest log-likelihood as predicted text
            best_idx = int(torch.argmax(torch.tensor(choice_responses)).item())
            responses.append(ModelResponse(choices[best_idx], logprob=choice_responses[best_idx]))
        return responses

def run_benchmarks():
    print("=" * 60)
    print("         Hugging Face LightEval - M0.1 Custom Runner")
    print("=" * 60)
    
    # Check environment variables, set dummy token if not set to prevent huggingface blocks
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "dummy_token_to_bypass_init")
    
    # Instantiation of custom model adapter
    model = LightevalM01Adapter(config="m01-55m")
    
    # We define standard benchmark prompts for immediate verification
    test_docs = [
        Doc(query="¿Cuál es la capital de Australia?", choices=["Sídney", "Canberra", "Melbourne", "Brisbane"]),
        Doc(query="El fuego necesita oxígeno para arder. Si quitamos el aire,", choices=["el fuego continuará", "el fuego se apagará"]),
    ]
    
    print("\n[LightEval Verification] Evaluating conditional loglikelihood of Multiple Choice Questions:")
    results = model.loglikelihood(test_docs)
    for doc, resp in zip(test_docs, results):
        print(f"Pregunta: {doc.query}")
        print(f"Opciones: {doc.choices}")
        print(f"Predicción elegida por M0.1: {resp.result_text} (LogLikelihood: {resp.logprob:.4f})\n")

if __name__ == "__main__":
    run_benchmarks()
