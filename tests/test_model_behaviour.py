"""behaviour_for_model_id: per-model behaviour flags follow the weights.

GRPO rollouts build their own ModelSpec (ART's endpoint/checkpoint name),
so the registry is the only place Llama 3.1 8B's fixes live -- this pins
that the lookup actually returns them.
"""

from mas_deepr.config import ModelSpec, behaviour_for_model_id, get_model


def test_llama_base_model_gets_its_registry_fixes() -> None:
    flags = behaviour_for_model_id("meta-llama/Llama-3.1-8B-Instruct")
    assert flags["single_tool_call_per_turn"] is True
    assert flags["supports_code_exec"] is False
    assert flags == {k: getattr(get_model("llama-3.1-8b-wandb"), k) for k in flags}


def test_unknown_base_model_returns_nothing() -> None:
    assert behaviour_for_model_id("someone/unknown-model") == {}


def test_flags_do_not_include_serving_or_pricing_fields() -> None:
    flags = behaviour_for_model_id("meta-llama/Llama-3.1-8B-Instruct")
    for field in ("key", "model_id", "self_hosted", "api_base_url", "api_key_env"):
        assert field not in flags
    # Must be valid ModelSpec kwargs.
    ModelSpec(key="k", model_id="m", **flags)
