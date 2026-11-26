"""rl/checkpoint_serving.py -- the checkpoint spec and registry entry shared
by eval_grpo_checkpoint.py and compile_dspy.py --grpo-model-name."""

import sys
import types
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mas_deepr.rl.checkpoint_serving import (
    CHECKPOINT_API_KEY_ENV,
    ServedCheckpoint,
    checkpoint_spec,
    register_checkpoint_spec,
)


def test_checkpoint_spec_points_at_served_endpoint() -> None:
    spec = checkpoint_spec(
        model_key="k",
        inference_name="qwen4b-s1-baseline@100",
        base_url="http://127.0.0.1:8000/v1",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
    )
    assert spec.model_id == "qwen4b-s1-baseline@100"
    assert spec.api_base_url == "http://127.0.0.1:8000/v1"
    assert spec.api_key_env == CHECKPOINT_API_KEY_ENV
    assert spec.self_hosted is False
    assert spec.params_b == 4  # from the qwen3-4b-instruct-2507 registry entry


def test_register_checkpoint_spec_adds_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.config import models

    monkeypatch.setattr(models, "MODEL_REGISTRY", dict(models.MODEL_REGISTRY))
    served = ServedCheckpoint(
        model=SimpleNamespace(inference_base_url="http://127.0.0.1:8000/v1"),
        backend=object(),
        step=100,
        inference_name="m@100",
    )
    spec = register_checkpoint_spec(
        served, model_key="m-step100", base_model="Qwen/Qwen3-4B-Instruct-2507"
    )
    assert models.MODEL_REGISTRY["m-step100"] is spec
    assert spec.model_id == "m@100"


@pytest.mark.art
@pytest.mark.asyncio
async def test_serve_checkpoint_rejects_unknown_backend() -> None:
    from mas_deepr.rl.checkpoint_serving import serve_checkpoint

    with pytest.raises(ValueError, match="unknown backend"):
        await serve_checkpoint(
            model_name="m", project="p", base_model="b", backend_kind="tinker"
        )


def _fake_art(
    monkeypatch: pytest.MonkeyPatch, art_path: Path, latest: int
) -> list[tuple[str, int]]:
    """Stand-in for the parts of ``art`` that serve_checkpoint uses; returns
    the log of adapter-lease events ("enter"/"exit", step)."""
    events: list[tuple[str, int]] = []

    class TrainableModel:
        def __init__(self, *, name: str, project: str, **_: Any) -> None:
            self.name, self.project = name, project
            self.inference_api_key = None
            self.inference_base_url = "http://127.0.0.1:8000/v1"

        async def register(self, backend: Any) -> None:
            pass

        async def get_step(self) -> int:
            return latest

        def get_inference_name(self, step: int | None = None) -> str:
            return f"{self.name}@{step}"

    class LocalBackend:
        _path = str(art_path)

        @asynccontextmanager
        async def exact_adapter_lease(
            self, model: Any, step: int
        ) -> AsyncIterator[None]:
            events.append(("enter", step))
            yield
            events.append(("exit", step))

    def get_model_dir(model: Any, art_path: str | None = None) -> str:
        return f"{art_path}/{model.project}/models/{model.name}"

    def get_step_checkpoint_dir(model_output_dir: str, step: int) -> str:
        return f"{model_output_dir}/checkpoints/{step:04d}"

    art = types.ModuleType("art")
    art.TrainableModel = TrainableModel  # type: ignore[attr-defined]
    local = types.ModuleType("art.local")
    local.LocalBackend = LocalBackend  # type: ignore[attr-defined]
    utils = types.ModuleType("art.utils")
    output_dirs = types.ModuleType("art.utils.output_dirs")
    output_dirs.get_model_dir = get_model_dir  # type: ignore[attr-defined]
    output_dirs.get_step_checkpoint_dir = get_step_checkpoint_dir  # type: ignore[attr-defined]
    for name, module in [
        ("art", art),
        ("art.local", local),
        ("art.utils", utils),
        ("art.utils.output_dirs", output_dirs),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    return events


def _checkpoints(art_path: Path, *steps: int) -> None:
    for step in steps:
        (art_path / "p" / "models" / "m" / "checkpoints" / f"{step:04d}").mkdir(
            parents=True
        )


async def _serve(step: int | None) -> Any:
    from mas_deepr.rl.checkpoint_serving import serve_checkpoint

    return await serve_checkpoint(
        model_name="m", project="p", base_model="b", backend_kind="local", step=step
    )


@pytest.mark.asyncio
async def test_older_local_step_is_loaded_into_vllm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """vLLM starts with only the latest checkpoint; step 0 needs a lease,
    released by close()."""
    events = _fake_art(monkeypatch, tmp_path, latest=100)
    _checkpoints(tmp_path, 0, 50, 100)
    served = await _serve(0)
    assert (served.step, served.inference_name) == (0, "m@0")
    assert events == [("enter", 0)]
    await served.close()
    assert events == [("enter", 0), ("exit", 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [None, 100])
async def test_latest_local_step_needs_no_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, step: int | None
) -> None:
    events = _fake_art(monkeypatch, tmp_path, latest=100)
    served = await _serve(step)
    assert (served.step, served.inference_name) == (100, "m@100")
    await served.close()
    assert events == []


@pytest.mark.asyncio
async def test_pruned_local_step_fails_before_any_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events = _fake_art(monkeypatch, tmp_path, latest=100)
    _checkpoints(tmp_path, 0, 25, 100)
    with pytest.raises(ValueError, match=r"step 47 .*steps on disk: \[0, 25, 100\]"):
        await _serve(47)
    assert events == []
