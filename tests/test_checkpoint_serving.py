"""rl/checkpoint_serving.py -- the checkpoint spec and registry entry shared
by eval_grpo_checkpoint.py and compile_dspy.py --grpo-model-name."""

from types import SimpleNamespace

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
