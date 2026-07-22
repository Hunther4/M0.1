import math
import importlib.util
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from src.model.lm import TransformerLM
from src.transformer.config import M01Config


def _load_function(script_name, function_name):
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "model_manipulation"
        / script_name
    )
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, function_name)


expand_model_depth = _load_function("expand_model.py", "expand_model_depth")
merge_checkpoints = _load_function("merge_checkpoints.py", "merge_checkpoints")


def _save_checkpoint(path, config, state, *, aliases=False):
    torch.save(
        {
            "model_config" if aliases else "config": config,
            "model_state" if aliases else "model_state_dict": state,
        },
        path,
    )


def test_merge_validates_configs_keys_and_shapes(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "merged.pt"
    _save_checkpoint(first, {"n_layers": 1}, {"weight": torch.ones(2)})
    _save_checkpoint(second, {"n_layers": 2}, {"weight": torch.ones(2)})
    with pytest.raises(ValueError, match="configs differ"):
        merge_checkpoints(str(first), str(second), str(output))

    _save_checkpoint(second, {"n_layers": 1}, {"other": torch.ones(2)})
    with pytest.raises(ValueError, match="keys differ"):
        merge_checkpoints(str(first), str(second), str(output))

    _save_checkpoint(second, {"n_layers": 1}, {"weight": torch.ones(3)})
    with pytest.raises(ValueError, match="Shape mismatch"):
        merge_checkpoints(str(first), str(second), str(output))


def test_merge_rejects_moe_by_default_and_can_copy_coherent_set(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "merged.pt"
    config = {"n_layers": 1, "num_experts": 2}
    state1 = {
        "blocks.0.attn.W_o.weight": torch.tensor([2.0]),
        "blocks.0.ff.experts.0.down_proj.weight": torch.tensor([10.0]),
        "blocks.0.ff.gate.weight": torch.tensor([20.0]),
    }
    state2 = {key: value + 2 for key, value in state1.items()}
    _save_checkpoint(first, config, state1)
    _save_checkpoint(second, config, state2)

    with pytest.raises(ValueError, match="expert alignment"):
        merge_checkpoints(str(first), str(second), str(output))

    merge_checkpoints(
        str(first), str(second), str(output), alpha=0.25, moe_policy="copy-second"
    )
    merged = torch.load(output, weights_only=True)
    assert torch.equal(
        merged["model_state_dict"]["blocks.0.ff.experts.0.down_proj.weight"],
        state2["blocks.0.ff.experts.0.down_proj.weight"],
    )
    assert torch.equal(
        merged["model_state_dict"]["blocks.0.ff.gate.weight"],
        state2["blocks.0.ff.gate.weight"],
    )
    assert torch.allclose(
        merged["model_state_dict"]["blocks.0.attn.W_o.weight"],
        torch.tensor([3.5]),
    )


def test_expand_preserves_aliases_boundaries_and_scales_residual_outputs(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    config = {"n_layers": 3, "num_dense_layers": 1, "num_experts": 2}
    state = {"embedding.weight": torch.tensor([99.0])}
    for index in range(3):
        kind = "dense" if index == 0 else "experts.0"
        state[f"blocks.{index}.attn.W_o.weight"] = torch.tensor([index + 1.0])
        state[f"blocks.{index}.ff.{kind}.down_proj.weight"] = torch.tensor(
            [10.0 + index]
        )
        state[f"blocks.{index}.norm1.gamma"] = torch.tensor([20.0 + index])
    _save_checkpoint(source, config, state, aliases=True)

    expand_model_depth(str(source), str(output), 5)
    expanded = torch.load(output, weights_only=True)
    assert "model_state" in expanded and "model_config" in expanded
    assert expanded["model_config"]["n_layers"] == 5
    assert expanded["model_manipulation"]["layer_mapping"] == [0, 1, 1, 2, 2]

    scale = math.sqrt(3 / 5)
    expanded_state = expanded["model_state"]
    assert torch.equal(expanded_state["embedding.weight"], state["embedding.weight"])
    assert torch.allclose(
        expanded_state["blocks.0.attn.W_o.weight"], torch.tensor([1.0 * scale])
    )
    assert torch.allclose(
        expanded_state["blocks.1.ff.experts.0.down_proj.weight"],
        torch.tensor([11.0 * scale]),
    )
    assert torch.equal(expanded_state["blocks.4.norm1.gamma"], torch.tensor([22.0]))
    assert not any(
        key.startswith("blocks.0.ff.experts") for key in expanded_state
    )


def test_expand_rejects_incomplete_layer_state(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "expanded.pt"
    _save_checkpoint(
        source,
        {"n_layers": 2, "num_dense_layers": 2},
        {"blocks.0.attn.W_o.weight": torch.ones(1)},
    )
    with pytest.raises(ValueError, match="no parameters"):
        expand_model_depth(str(source), str(output), 3)


def test_expanded_real_model_state_loads_strictly(tmp_path):
    source = tmp_path / "model.pt"
    output = tmp_path / "expanded.pt"
    config = M01Config(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        num_dense_layers=1,
        num_experts=2,
        num_shared_experts=1,
        moe_top_k=1,
        use_mla=False,
        use_hybrid_attention=False,
    )
    model = TransformerLM(config)
    serialized_config = {field.name: getattr(config, field.name) for field in fields(config)}
    _save_checkpoint(source, serialized_config, model.state_dict())

    expand_model_depth(str(source), str(output), 3)
    expanded = torch.load(output, weights_only=True)
    expanded_model = TransformerLM(M01Config(**expanded["config"]))
    expanded_model.load_state_dict(expanded["model_state_dict"], strict=True)
