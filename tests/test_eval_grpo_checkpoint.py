"""eval_grpo_checkpoint.py's checkpoint spec keeps the base model's
per-model fixes (same contract as rollout_fn -- see test_rollout.py)."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_grpo_checkpoint.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("eval_grpo_checkpoint_t", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_grpo_checkpoint_t"] = module
    spec.loader.exec_module(module)
    return module


def test_llama_checkpoint_keeps_llama_fixes_and_art_endpoint() -> None:
    spec = _load().checkpoint_spec(
        model_key="llama8b-grpo-baseline-step20",
        inference_name="wandb-artifact:///e/p/llama8b-grpo-baseline:step20",
        base_url="https://api.training.example/v1",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert spec.single_tool_call_per_turn is True
    assert spec.supports_code_exec is False
    assert spec.params_b == 8
    assert spec.model_id == "wandb-artifact:///e/p/llama8b-grpo-baseline:step20"
    assert spec.api_base_url == "https://api.training.example/v1"
    assert spec.self_hosted is False
    assert spec.api_key_env == "MAS_GRPO_CHECKPOINT_API_KEY"


def test_unknown_base_model_keeps_previous_defaults() -> None:
    spec = _load().checkpoint_spec(
        model_key="k", inference_name="m", base_url=None, base_model="someone/x"
    )
    assert spec.family == "qwen3" and spec.params_b == 14
    assert spec.single_tool_call_per_turn is False
    assert spec.supports_code_exec is True


def test_script_offers_r2_sync() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    assert '"--r2-sync"' in text and "r2.finish([out_dir])" in text
