import dspy

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


def test_build_dspy_lm_frontier_uses_judge_settings() -> None:
    settings = Settings(judge_base_url="http://judge.example/v1", judge_api_key="j")
    spec = ModelSpec(
        key="gpt-5-mini", model_id="gpt-5-mini", family="frontier", self_hosted=False
    )

    lm = build_dspy_lm(spec, settings)

    assert lm.model == "gpt-5-mini"
    assert lm.kwargs.get("api_base") == "http://judge.example/v1"
