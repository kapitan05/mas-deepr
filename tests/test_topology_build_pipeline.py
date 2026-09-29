"""build_pipeline's per-spec tool wiring (no network -- just inspects what
gets attached to each role's Agent).
"""

from pathlib import Path

import pytest
from agent_framework.openai import OpenAIChatCompletionClient

from mas_deepr.agents import build_pipeline
from mas_deepr.config import ModelSpec, Settings, get_model
from mas_deepr.rl.trainable_client import TrainableOpenAIChatCompletionClient
from mas_deepr.telemetry import TelemetryTracker


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-x")
    monkeypatch.delenv("MAS_SLM_BASE_URL", raising=False)
    return Settings()


def _tracker(tmp_path: Path) -> TelemetryTracker:
    return TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="t", phase="test")


def test_synthesizer_gets_code_exec_tool_by_default(
    settings: Settings, tmp_path: Path
) -> None:
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=_tracker(tmp_path)
    )
    assert len(pipeline.synthesizer.default_options["tools"]) == 1


def test_synthesizer_drops_code_exec_tool_when_unsupported(
    settings: Settings, tmp_path: Path
) -> None:
    """Regression test for a real bug: llama-3.1-8b-wandb attempted
    run_python and produced malformed pseudo-tool-call text as its "final
    answer" on every FRAMES question (which need it for verification),
    while browsecomp/research_rubrics (no run_python needed) were fine."""
    spec = ModelSpec(
        key="no-code-exec",
        model_id="whatever",
        family="frontier",
        self_hosted=False,
        api_key_env="OPENAI_API_KEY",
        supports_code_exec=False,
    )
    pipeline = build_pipeline(spec=spec, settings=settings, tracker=_tracker(tmp_path))
    assert pipeline.synthesizer.default_options["tools"] == []
    # Browser's tools are unaffected -- this flag is narrower than
    # supports_native_tools, which governs the whole model. Default set:
    # web_search, fetch_page, wikipedia_search, semantic_scholar_search,
    # pubmed_search (Tavily is opt-in, off by default -- see
    # mcp_enabled_tools_set).
    assert len(pipeline.browser.default_options["tools"]) == 5


def test_llama_wandb_registry_entry_has_code_exec_disabled() -> None:
    assert get_model("llama-3.1-8b-wandb").supports_code_exec is False


def test_browser_tools_gated_by_settings(settings: Settings, tmp_path: Path) -> None:
    """mcp_enabled_tools is the single gate for all 6 MCP tools -- confirms
    both directions: dropping a tool, and adding the opt-in Tavily one."""
    settings.mcp_enabled_tools = "web_search,fetch_page"
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=_tracker(tmp_path)
    )
    tool_names = {t.name for t in pipeline.browser.default_options["tools"]}
    assert tool_names == {"web_search", "fetch_page"}

    settings.mcp_enabled_tools = (
        "web_search,fetch_page,wikipedia_search,semantic_scholar_search,"
        "pubmed_search,tavily_search"
    )
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=_tracker(tmp_path)
    )
    tool_names = {t.name for t in pipeline.browser.default_options["tools"]}
    assert tool_names == {
        "web_search",
        "fetch_page",
        "wikipedia_search",
        "semantic_scholar_search",
        "pubmed_search",
        "tavily_search",
    }


def test_trainable_true_uses_the_grpo_chat_client(
    settings: Settings, tmp_path: Path
) -> None:
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"),
        settings=settings,
        tracker=_tracker(tmp_path),
        trainable=True,
    )
    assert isinstance(pipeline.manager.client, TrainableOpenAIChatCompletionClient)
    assert isinstance(pipeline.browser.client, TrainableOpenAIChatCompletionClient)


def test_trainable_false_uses_the_stock_chat_client(
    settings: Settings, tmp_path: Path
) -> None:
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=_tracker(tmp_path)
    )
    assert type(pipeline.manager.client) is OpenAIChatCompletionClient


def test_wikipedia_only_config_for_ban_risk_free_run(
    settings: Settings, tmp_path: Path
) -> None:
    """The specific config the deployment checklist recommends for a
    ban-risk-free run on Wikipedia-answerable benchmarks: drop web_search
    (routes through SearXNG's scraped engines) entirely."""
    settings.mcp_enabled_tools = (
        "wikipedia_search,fetch_page,semantic_scholar_search,pubmed_search"
    )
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=_tracker(tmp_path)
    )
    tool_names = {t.name for t in pipeline.browser.default_options["tools"]}
    assert "web_search" not in tool_names
    assert "wikipedia_search" in tool_names
