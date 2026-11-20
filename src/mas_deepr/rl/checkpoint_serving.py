"""Serve a GRPO-trained checkpoint so the rest of mas-deepr can call it.

A trained ``art.TrainableModel`` checkpoint is only servable through ART's
own registration (``model.register(backend)`` + ``get_inference_name``),
not a static registry key. Evaluation (``scripts/eval_grpo_checkpoint.py``)
and post-RL prompt optimization (``scripts/compile_dspy.py
--grpo-model-name``) both need that, so the registration and the
``ModelSpec`` built for the served endpoint live here, shared.

``art`` is imported lazily inside the functions: importing it patches
``transformers`` process-globally (see tests/test_rollout.py's docstring),
and the scripts' own unit tests import those scripts without it.
"""

import os
from dataclasses import dataclass
from typing import Any

from mas_deepr.config import ModelSpec

CHECKPOINT_API_KEY_ENV = "MAS_GRPO_CHECKPOINT_API_KEY"


def checkpoint_spec(
    *, model_key: str, inference_name: str, base_url: str | None, base_model: str
) -> ModelSpec:
    """ModelSpec for a trained checkpoint served by ART (W&B or local vLLM).

    Endpoint, name and key come from ART; *behaviour* (tool-call quirks,
    code-exec support, context window, size) from the registry entry for the
    same base model -- e.g. a Llama 3.1 8B checkpoint keeps
    single_tool_call_per_turn and supports_code_exec=False, exactly like its
    pre-training eval (otherwise it 400s on multi-call turns and re-enables
    the garbled run_python). Unknown base models keep the previous defaults
    (family="qwen3", params_b=14). Same lookup as rl/rollout.py::rollout_fn.
    """
    from mas_deepr.config import behaviour_for_model_id

    return ModelSpec(
        **{
            "family": "qwen3",
            "params_b": 14,
            **behaviour_for_model_id(base_model),
            "key": model_key,
            "model_id": inference_name,
            "self_hosted": False,
            "api_base_url": base_url,
            "api_key_env": CHECKPOINT_API_KEY_ENV,
        }
    )


@dataclass
class ServedCheckpoint:
    model: Any  # art.TrainableModel -- typed loosely, art is imported lazily
    backend: Any
    step: int
    inference_name: str


async def serve_checkpoint(
    *,
    model_name: str,
    project: str,
    base_model: str,
    backend_kind: str,
    model_kwargs: dict[str, Any] | None = None,
    step: int | None = None,
) -> ServedCheckpoint:
    """Register ``model_name`` with a backend so its checkpoint is served.

    ``backend_kind`` is ``"local"`` (this machine's GPU; ``model_kwargs``
    must then carry the same ``rl/local_backend.py`` settings the model was
    trained with) or ``"serverless"`` (W&B). ``step`` defaults to the
    latest checkpoint. Also sets the API-key environment variable the
    checkpoint's ``ModelSpec`` reads -- a local vLLM server may have no
    key, but the OpenAI client still needs a non-empty one.
    """
    import art

    model = art.TrainableModel(
        name=model_name,
        # Same convention as train_grpo.py -- required as of openpipe-art
        # 0.5.20, see that file's comment on the same kwarg.
        run_name=model_name,
        project=project,
        base_model=base_model,
        **(model_kwargs or {}),
    )
    backend: Any
    if backend_kind == "local":
        from art.local import LocalBackend

        backend = LocalBackend()
    elif backend_kind == "serverless":
        from art.serverless.backend import ServerlessBackend

        backend = ServerlessBackend()
    else:
        raise ValueError(f"unknown backend {backend_kind!r}")
    await model.register(backend)
    resolved = step if step is not None else await model.get_step()
    if model.inference_api_key:
        os.environ[CHECKPOINT_API_KEY_ENV] = model.inference_api_key
    elif backend_kind == "local":
        os.environ[CHECKPOINT_API_KEY_ENV] = "EMPTY"
    return ServedCheckpoint(
        model=model,
        backend=backend,
        step=resolved,
        inference_name=model.get_inference_name(step=resolved),
    )


def register_checkpoint_spec(
    served: ServedCheckpoint, *, model_key: str, base_model: str
) -> ModelSpec:
    """Build the served checkpoint's ``ModelSpec`` and add it to
    ``MODEL_REGISTRY`` under ``model_key`` (what the eval and DSPy code
    look models up by)."""
    from mas_deepr.config.models import MODEL_REGISTRY

    spec = checkpoint_spec(
        model_key=model_key,
        inference_name=served.inference_name,
        base_url=served.model.inference_base_url,
        base_model=base_model,
    )
    MODEL_REGISTRY[model_key] = spec
    return spec
