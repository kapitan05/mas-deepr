"""Frontier-model endpoint routing in llm/factory.py.

No network: we only construct the client and inspect where it points.
"""

import httpx
import pytest
from agent_framework.openai import OpenAIChatCompletionClient

from mas_deepr.config import Settings, get_model
from mas_deepr.llm import build_chat_client


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("WANDB_API_KEY", "wb-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.delenv("MAS_SLM_BASE_URL", raising=False)
    return Settings()


def test_self_hosted_routes_to_slm_endpoint(settings: Settings) -> None:
    client = build_chat_client(get_model("qwen3-8b"), settings)
    assert str(client.client.base_url).rstrip("/") == settings.slm_base_url.rstrip("/")
    assert client.client.api_key == settings.slm_api_key
    assert client.model == "Qwen/Qwen3-8B"


def test_gpt_41_uses_openai_key_and_default_endpoint(settings: Settings) -> None:
    client = build_chat_client(get_model("gpt-4.1"), settings)
    assert client.client.api_key == "sk-openai"
    assert "api.openai.com" in str(client.client.base_url)
    assert client.model == "gpt-4.1"


def test_gemini_routes_to_google_openai_compat_endpoint(settings: Settings) -> None:
    client = build_chat_client(get_model("gemini-2.5-pro"), settings)
    assert client.client.api_key == "sk-gemini"
    assert "generativelanguage.googleapis.com" in str(client.client.base_url)


def test_deepseek_routes_to_deepseek_endpoint(settings: Settings) -> None:
    client = build_chat_client(get_model("deepseek-chat"), settings)
    assert client.client.api_key == "sk-deepseek"
    assert "api.deepseek.com" in str(client.client.base_url)


def test_judge_frontier_spec_falls_back_to_judge_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # gpt-5-mini has no api_key_env / api_base_url -> uses judge_* fields.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-judge")
    monkeypatch.setenv("MAS_JUDGE_BASE_URL", "https://judge.example/v1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    settings = Settings()
    client = build_chat_client(get_model("gpt-5-mini"), settings)
    assert client.client.api_key == "sk-judge"
    assert "judge.example" in str(client.client.base_url)


def test_missing_frontier_key_fails_loudly_not_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real bug: GEMINI_API_KEY unset used to fall
    back to settings.judge_api_key (OPENAI_API_KEY), silently sending the
    wrong provider's key to Google's endpoint (rejected with "Please pass
    a valid API key"). It must now raise, not substitute a different key.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings()

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        build_chat_client(get_model("gemini-2.5-pro"), settings)


def test_wandb_inference_models_route_to_wandb_endpoint(settings: Settings) -> None:
    for key in ("llama-3.1-8b-wandb", "granite-4.2-8b-wandb", "gpt-oss-20b-wandb"):
        client = build_chat_client(get_model(key), settings)
        assert client.client.api_key == "wb-key"
        assert "api.inference.wandb.ai" in str(client.client.base_url)


def test_default_judge_model_routes_to_wandb_not_a_personal_api_key(
    settings: Settings,
) -> None:
    """Regression guard: the shared judge used to be hardcoded to gpt-5-mini
    (OpenAI), so an exhausted OPENAI_API_KEY zeroed browsecomp/
    research_rubrics for every model under test, not just OpenAI ones.
    Then moved to gemini-2.5-flash-lite (still a personal API key). Default
    judge must now run on WANDB_API_KEY, not any personal provider key."""
    client = build_chat_client(get_model(settings.judge_model), settings)
    assert "api.inference.wandb.ai" in str(client.client.base_url)
    assert client.client.api_key == "wb-key"


def _read_timeout(client: OpenAIChatCompletionClient) -> float:
    timeout = client.client.timeout
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read is not None
    return timeout.read


def test_self_hosted_client_has_timeout_and_retries(settings: Settings) -> None:
    """Regression test: a hung upstream response (confirmed live against
    api.inference.wandb.ai) used to block an entire eval run forever since
    no client-side timeout existed at all."""
    client = build_chat_client(get_model("qwen3-8b"), settings)
    assert _read_timeout(client) == settings.llm_request_timeout_s
    assert client.client.max_retries == settings.llm_max_retries


def test_frontier_client_has_timeout_and_retries(settings: Settings) -> None:
    client = build_chat_client(get_model("gpt-4.1"), settings)
    assert _read_timeout(client) == settings.llm_request_timeout_s
    assert client.client.max_retries == settings.llm_max_retries


def test_custom_timeout_and_retries_are_respected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.setenv("MAS_LLM_REQUEST_TIMEOUT_S", "12.5")
    monkeypatch.setenv("MAS_LLM_MAX_RETRIES", "7")
    settings = Settings()
    client = build_chat_client(get_model("gpt-4.1"), settings)
    assert _read_timeout(client) == 12.5
    assert client.client.max_retries == 7


def test_frontier_specs_carry_pricing_for_the_cost_plot() -> None:
    for key in (
        "gpt-4.1",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "deepseek-chat",
        "llama-3.1-8b-wandb",
        "granite-4.2-8b-wandb",
        "gpt-oss-20b-wandb",
    ):
        spec = get_model(key)
        assert spec.input_price_per_mtok > 0
        assert spec.output_price_per_mtok > 0
        assert not spec.self_hosted
