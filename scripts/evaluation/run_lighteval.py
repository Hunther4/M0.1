import os
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import BatchEncoding

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


class WindowsSafeSampleCache(SampleCache):
    """SampleCache that sanitizes invalid Windows filename chars in task names (|, :)."""

    def get_cache_path(self, task_id) -> Path:
        safe_task_name = task_id.task_name.replace("|", "_").replace(":", "_")
        return self.cache_dir / safe_task_name / task_id.task_hash / f"{task_id.sampling_method.name}.parquet"


class HFTokenizerWrapper:
    """Minimal HuggingFace-compatible tokenizer wrapper around M0.1's custom BPE tokenizer.

    lighteval's LightevalModel ABC requires `tokenizer` to behave like a
    PreTrainedTokenizerBase (encode/decode/eos_token_id/batch_decode/__call__).
    Our custom Tokenizer returns plain lists of ints and lacks eos_token_id, so we
    wrap it here.
    """

    def __init__(self, custom: "Tokenizer"):
        self._t = custom
        self.eos_token_id = custom.special_tokens.get("<|endoftext|>", 256)

    def encode(self, text, add_special_tokens=None, **kwargs):
        return self._t.encode(text)

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._t.decode(ids)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return [self._t.decode(seq) for seq in sequences]

    def __call__(self, texts, padding=None, add_special_tokens=None, return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self._t.encode(t) for t in texts]
        if return_tensors == "pt":
            return BatchEncoding({"input_ids": [torch.tensor(x, dtype=torch.long) for x in ids]})
        return BatchEncoding({"input_ids": ids})

    @property
    def vocab(self):
        return {i: self._t.decode([i]) for i in range(len(self._t.vocab))}


class LightevalM01Adapter(LightevalModel):
    """Adapter bridging the M0.1 MoE model with the Hugging Face LightEval suite.

    Subclasses lighteval 0.13.0's LightevalModel ABC (which has NO __init__ and
    requires the tokenizer/add_special_tokens/max_length properties plus the
    greedy_until/loglikelihood/loglikelihood_rolling methods).
    """

    def __init__(self, checkpoint_path=None, config=None, env_config=None):
        # LightevalModel is an ABC with no __init__ -> do NOT call super().__init__.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Local BPE tokenizer (custom interface)
        self._custom_tok = Tokenizer()
        self._custom_tok.load("data/tokenizers/tokenizer_final_8k.json")
        self.eos_token_id = self._custom_tok.special_tokens.get("<|endoftext|>", 256)

        # Load model weights
        ckpt_path = checkpoint_path
        if ckpt_path is None:
            ckpt_path = "checkpoints/m01_hardened_final.pt"
            if not os.path.exists(ckpt_path):
                ckpt_path = "checkpoints/final_combined_8k.pt"

        print(f"[LightEval Adapter] Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        ckpt_config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

        # Infer FFN routed/shared expert dimensions directly from actual parameter shapes
        d_ff_routed_inferred = None
        for k, v in state_dict.items():
            if "blocks.0.ff.experts.0.gate_proj.weight" in k:
                d_ff_routed_inferred = v.shape[0]
                break

        d_ff_shared_inferred = None
        for k, v in state_dict.items():
            if "blocks.0.ff.shared_experts.0.gate_proj.weight" in k:
                d_ff_shared_inferred = v.shape[0]
                break

        self.model_config = M01Config(
            vocab_size=ckpt_config["vocab_size"],
            context_length=ckpt_config["context_length"],
            d_model=ckpt_config["d_model"],
            n_heads=ckpt_config["n_heads"],
            d_ff=ckpt_config["d_ff"],
            d_ff_shared=ckpt_config.get("d_ff_shared") or d_ff_shared_inferred,
            d_ff_routed=ckpt_config.get("d_ff_routed") or d_ff_routed_inferred,
            n_layers=ckpt_config["n_layers"],
            num_experts=ckpt_config["num_experts"],
            num_shared_experts=ckpt_config["num_shared_experts"],
            moe_top_k=ckpt_config["moe_top_k"],
            use_hybrid_attention=ckpt_config["use_hybrid_attention"],
            local_window_size=ckpt_config["local_window_size"],
        )

        self.model = TransformerLM(self.model_config).to(self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    # --- ABC properties ------------------------------------------------------
    @property
    def tokenizer(self):
        return HFTokenizerWrapper(self._custom_tok)

    @property
    def add_special_tokens(self) -> bool:
        # Our tokenizer already handles special tokens; don't let lighteval add more.
        return False

    @property
    def max_length(self) -> int:
        return self.model_config.context_length

    # --- Generation (HumanEval, etc.) ---------------------------------------
    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        for doc in docs:
            prompt = doc.query
            max_len = doc.generation_size if doc.generation_size else self.model_config.context_length
            stop_seqs = doc.stop_sequences or []

            tokens = self._custom_tok.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=self.device)
            generated = []

            with torch.no_grad():
                for _ in range(max_len):
                    if x.shape[1] > self.model_config.context_length:
                        x = x[:, -self.model_config.context_length:]
                    logits = self.model(x)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    if next_token == self.eos_token_id:
                        break
                    generated.append(next_token)
                    x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long, device=self.device)], dim=1)
                    if stop_seqs:
                        text = self._custom_tok.decode(generated)
                        if any(ss in text for ss in stop_seqs):
                            break

            text = self._custom_tok.decode(generated)
            responses.append(ModelResponse(text=[text], output_tokens=[generated], logprobs=[]))
        return responses

    # --- Multiple-choice log-likelihood (MMLU, ARC, ...) --------------------
    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        for doc in docs:
            query_tokens = self._custom_tok.encode(doc.query)
            per_choice_lp = []
            argmax_eq_gold = []
            output_tokens = []

            with torch.no_grad():
                for choice in doc.choices:
                    choice_tokens = self._custom_tok.encode(choice)
                    full = query_tokens + choice_tokens
                    x = torch.tensor([full[:-1]], dtype=torch.long, device=self.device)
                    targets = torch.tensor([full[1:]], dtype=torch.long, device=self.device)

                    logits = self.model(x)  # (1, len-1, vocab)
                    log_probs = F.log_softmax(logits, dim=-1)
                    target_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

                    # Log-prob of the continuation (choice) tokens only
                    choice_lp = target_lp[:, -len(choice_tokens):].sum().item()
                    per_choice_lp.append(choice_lp)

                    # argmax at the position predicting the first choice token
                    pred_pos = max(len(query_tokens) - 1, 0)
                    first_pred = logits[:, pred_pos, :].argmax(dim=-1).item()
                    argmax_eq_gold.append(first_pred == choice_tokens[0])
                    output_tokens.append([choice_tokens[0]])

            responses.append(
                ModelResponse(
                    logprobs=per_choice_lp,
                    argmax_logits_eq_gold=argmax_eq_gold,
                    input_tokens=query_tokens,
                    output_tokens=output_tokens,
                )
            )
        return responses

    # --- Rolling log-likelihood (perplexity-style tasks) --------------------
    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        responses = []
        for doc in docs:
            tokens = self._custom_tok.encode(doc.query)
            if len(tokens) < 2:
                responses.append(ModelResponse(logprobs=[0.0], input_tokens=tokens, output_tokens=[[t] for t in tokens]))
                continue
            x = torch.tensor([tokens[:-1]], dtype=torch.long, device=self.device)
            targets = torch.tensor([tokens[1:]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                logits = self.model(x)
                log_probs = F.log_softmax(logits, dim=-1)
                target_lp = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            total = target_lp.sum().item()
            responses.append(
                ModelResponse(logprobs=[total], input_tokens=tokens, output_tokens=[[t] for t in tokens])
            )
        return responses


# --- Real LightEval runner -------------------------------------------------

CHECKPOINT_PRIMARY = "checkpoints/m01_hardened_final.pt"
CHECKPOINT_FALLBACK = "checkpoints/final_combined_8k.pt"
TASKS = "mmlu,arc,gsm8k"


def _resolve_checkpoint_path() -> str | None:
    if os.path.exists(CHECKPOINT_PRIMARY):
        return CHECKPOINT_PRIMARY
    if os.path.exists(CHECKPOINT_FALLBACK):
        return CHECKPOINT_FALLBACK
    return None


def run_real_eval(tasks: str = TASKS, max_samples: int | None = None, checkpoint_path: str | None = None) -> None:
    print("=" * 60)
    print("   Hugging Face LightEval - M0.1 Real Benchmark Runner")
    print("=" * 60)

    if checkpoint_path is not None:
        ckpt = checkpoint_path
    else:
        ckpt = _resolve_checkpoint_path()

    if ckpt is None or not os.path.exists(ckpt):
        print(
            "[LightEval] ERROR: No model checkpoint found.\n"
            f"[LightEval] Expected one of:\n"
            f"             - {CHECKPOINT_PRIMARY}\n"
            f"             - {CHECKPOINT_FALLBACK}\n"
            "[LightEval] Place a checkpoint and re-run. Aborting."
        )
        sys.exit(1)
    print(f"[LightEval] Using checkpoint: {ckpt}")

    try:
        from lighteval.logging.evaluation_tracker import EvaluationTracker
        from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
    except ImportError as e:
        print(f"[LightEval] ERROR: Could not import lighteval ({e}). Install with: pip install lighteval")
        sys.exit(1)

    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "dummy_token_to_bypass_init")

    model = LightevalM01Adapter(checkpoint_path=ckpt, config="m01-55m")

    model_config = lighteval_model_config(model)
    model.config = model_config
    model._cache = WindowsSafeSampleCache(model_config)

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=max_samples,
    )

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model=model,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    results = pipeline.get_results()
    print("\n=== RESULTS ===")
    if results:
        for k, v in results.items():
            print(f"  {k}: {v}")
    else:
        print("  (empty dict)")
    print("===============")


def lighteval_model_config(model: LightevalM01Adapter):
    from lighteval.models.abstract_model import ModelConfig

    return ModelConfig(model_name="m01-55m", cache_dir="./results/lighteval_cache")


def run_benchmarks():
    """Backwards-compatible entry point -> delegates to the real runner."""
    run_real_eval()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LightEval benchmarks on M0.1.")
    parser.add_argument("--tasks", type=str, default=TASKS, help="Comma-separated task list.")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap samples per task (smoke test).")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint.")
    args = parser.parse_args()
    run_real_eval(tasks=args.tasks, max_samples=args.max_samples, checkpoint_path=args.checkpoint)
