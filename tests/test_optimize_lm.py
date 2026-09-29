import dspy
import pytest

from mas_deepr.config import Settings
from mas_deepr.config.models import ModelSpec
from mas_deepr.optimize.lm import build_dspy_lm


def test_build_dspy_lm_self_hosted_uses_slm_base_url() -> None:
    settings = Settings(slm_base_url="http://localhost:9000/v1", slm_api_key="k")
    spec = ModelSpec(
        key="m", model_id="Qwen/Qwen3-8B", family="qwen3", self_hosted=True
    )

    lm = build_dspy_lm(spec, settings)

    assert isinstance(lm, dspy.LM)
    assert lm.model == "openai/Qwen/Qwen3-8B"
    assert lm.kwargs.get("api_base") == "http://localhost:9000/v1"


def test_build_dspy_lm_frontier_no_own_key_falls_back_to_judge_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gpt-5-mini-shaped spec: no api_key_env/api_base_url of its own -> by
    design, uses the judge's settings (see llm/factory.py's identical
    fallback).

    judge_api_key's validation_alias="OPENAI_API_KEY" means a real
    OPENAI_API_KEY in the environment otherwise wins over the explicit
    kwarg below (the same pydantic-settings gotcha documented in
    test_mcp_web_search_fallback.py) -- delenv + _env_file=None isolate it.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        judge_base_url="http://judge.example/v1",
        OPENAI_API_KEY="j",  # judge_api_key's validation_alias, not its own name
    )
    spec = ModelSpec(
        key="gpt-5-mini", model_id="gpt-5-mini", family="frontier", self_hosted=False
    )

    lm = build_dspy_lm(spec, settings)

    # Regression test: litellm needs an explicit provider prefix for a bare
    # model string -- confirmed live, "LLM Provider NOT provided" without it.
    assert lm.model == "openai/gpt-5-mini"
    assert lm.kwargs.get("api_base") == "http://judge.example/v1"
    assert lm.kwargs.get("api_key") == "j"


def test_build_dspy_lm_frontier_with_own_key_uses_its_own_endpoint_not_judges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real, confirmed bug: a spec declaring its own
    api_key_env/api_base_url (e.g. qwen3-14b-instruct-wandb, W&B Inference)
    used to be silently routed to the JUDGE's endpoint/key instead of its
    own -- same bug class already fixed in llm/factory.py::build_chat_client,
    just never mirrored here until compile_dspy.py was first run against a
    non-self-hosted spec."""
    monkeypatch.setenv("WANDB_API_KEY", "wb-real-key")
    settings = Settings(judge_base_url="http://judge.example/v1", judge_api_key="j")
    spec = ModelSpec(
        key="qwen3-14b-instruct-wandb",
        model_id="OpenPipe/Qwen3-14B-Instruct",
        family="frontier",
        self_hosted=False,
        api_base_url="https://api.inference.wandb.ai/v1",
        api_key_env="WANDB_API_KEY",
    )

    lm = build_dspy_lm(spec, settings)

    assert lm.model == "openai/OpenPipe/Qwen3-14B-Instruct"
    assert lm.kwargs.get("api_base") == "https://api.inference.wandb.ai/v1"
    assert lm.kwargs.get("api_key") == "wb-real-key"


def test_build_dspy_lm_missing_declared_key_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both delenv AND _env_file=None are needed to truly isolate this from
    # the real .env, which does have WANDB_API_KEY set -- see
    # test_mcp_web_search_fallback.py's identical caveat.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    spec = ModelSpec(
        key="qwen3-14b-instruct-wandb",
        model_id="OpenPipe/Qwen3-14B-Instruct",
        family="frontier",
        self_hosted=False,
        api_base_url="https://api.inference.wandb.ai/v1",
        api_key_env="WANDB_API_KEY",
    )

    with pytest.raises(ValueError, match="WANDB_API_KEY"):
        build_dspy_lm(spec, settings)
