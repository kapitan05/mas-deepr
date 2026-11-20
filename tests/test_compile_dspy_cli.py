"""scripts/compile_dspy.py -- registry-model mode vs the post-RL
--grpo-model-name checkpoint mode."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compile_dspy.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("compile_dspy_t", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compile_dspy_t"] = module
    spec.loader.exec_module(module)
    return module


def _run_main(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["compile_dspy.py", *argv])
    _load().main()


def test_model_and_grpo_model_name_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--model", "qwen3-8b", "--grpo-model-name", "m"])
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, [])


def test_grpo_model_name_requires_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, ["--grpo-model-name", "m"])


@pytest.mark.asyncio
async def test_compile_against_checkpoint_serves_then_compiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load()
    from mas_deepr.config import models
    from mas_deepr.rl import checkpoint_serving
    from mas_deepr.rl.checkpoint_serving import ServedCheckpoint

    monkeypatch.setattr(models, "MODEL_REGISTRY", dict(models.MODEL_REGISTRY))
    calls: dict[str, Any] = {}

    async def fake_serve(**kwargs: Any) -> ServedCheckpoint:
        calls["serve"] = kwargs
        return ServedCheckpoint(
            model=SimpleNamespace(inference_base_url="http://127.0.0.1:8000/v1"),
            backend=object(),
            step=100,
            inference_name="qwen4b-s4-rubrics@100",
        )

    def fake_compile(*, spec: Any, settings: Any, **kwargs: Any) -> dict[str, Path]:
        calls["compile"] = (spec, kwargs)
        return {"synthesizer": Path("synthesizer.compiled.yaml")}

    monkeypatch.setattr(checkpoint_serving, "serve_checkpoint", fake_serve)
    monkeypatch.setattr(module, "compile_pipeline", fake_compile)
    args = argparse.Namespace(
        grpo_model_name="qwen4b-s4-rubrics",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        project="mas-deepr",
        backend="local",
        step=None,
        max_seq_length=32_768,
        gpu_memory_utilization=0.8,
        load_in_4bit=False,
        lora_rank=16,
        lora_alpha=32,
    )
    written = await module._compile_against_checkpoint(
        args, SimpleNamespace(wandb_project="x"), {"auto": "medium"}
    )
    assert written == {"synthesizer": Path("synthesizer.compiled.yaml")}
    assert calls["serve"]["model_name"] == "qwen4b-s4-rubrics"
    assert calls["serve"]["backend_kind"] == "local"
    assert set(calls["serve"]["model_kwargs"]) == {"_internal_config", "lora_config"}
    spec, kwargs = calls["compile"]
    assert spec.model_id == "qwen4b-s4-rubrics@100"
    assert kwargs == {"auto": "medium"}
    assert "qwen4b-s4-rubrics-step100" in models.MODEL_REGISTRY
