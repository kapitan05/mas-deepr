"""``make_retriever``'s returned function is deliberately synchronous (it
bridges into async MCP tool calls via ``asyncio.run`` internally, since
DSPy's MIPROv2 compile loop is sync) -- so these tests call it from plain
sync test functions, not ``@pytest.mark.asyncio`` ones, which would already
have a running event loop and make ``asyncio.run`` raise.
"""

from pathlib import Path
from typing import Any

import pytest

from mas_deepr.config import Settings
from mas_deepr.optimize.modules import make_retriever
from mas_deepr.tools import MCPToolClient


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        **overrides,  # type: ignore[arg-type]
    )


def test_retriever_formats_hits_and_fetches_top_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        if tool_name == "web_search":
            return [
                {"title": "A", "url": "http://a.com", "snippet": "snippet a"},
                {"title": "B", "url": "http://b.com", "snippet": "snippet b"},
            ]
        if tool_name == "fetch_page":
            return f"full page text for {kwargs['url']}"
        raise AssertionError(f"unexpected tool: {tool_name}")

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(tmp_path)
    retrieve = make_retriever(settings=settings, top_k=2, fetch_top=1)

    context = retrieve("some query")

    assert "[1] A (http://a.com)" in context
    assert "snippet a" in context
    assert "full page text for http://a.com" in context
    assert "[2] B (http://b.com)" in context
    assert "snippet b" in context
    assert "full page text for http://b.com" not in context  # fetch_top=1


def test_retriever_handles_no_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        return []

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(tmp_path)
    retrieve = make_retriever(settings=settings)

    assert retrieve("no hits query") == "No search results found."


def test_retriever_handles_failed_search_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``MCPToolClient.call`` never raises -- a failed search comes back as
    an inline ``"[mcp_call_failed: ...]"`` string, which the retriever must
    treat as "no results" rather than iterating over its characters."""

    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        return "[mcp_call_failed: ConnectionError: refused]"

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(tmp_path)
    retrieve = make_retriever(settings=settings)

    assert retrieve("query") == "No search results found."


def test_retriever_prefers_web_search_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        calls.append(tool_name)
        return []

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(
        tmp_path,
        mcp_enabled_tools="web_search,fetch_page,wikipedia_search",
    )
    retrieve = make_retriever(settings=settings)
    retrieve("q")

    assert calls == ["web_search"]


def test_retriever_falls_back_to_wikipedia_search_in_wiki_paper_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real, confirmed bug: this used to hardcode
    "web_search" regardless of mcp_enabled_tools, silently ignoring
    compile_dspy.py's wiki_paper scope (dropping web_search specifically to
    avoid hammering SearXNG at compile-time rollout volume) -- confirmed
    live that every one of 6,728 search calls in one real compile run
    failed as a direct consequence (see the second bug this same commit
    fixes, in test_retriever_reuses_one_event_loop_per_call below)."""
    calls: list[str] = []

    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        calls.append(tool_name)
        return []

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(
        tmp_path,
        mcp_enabled_tools="wikipedia_search,fetch_page,semantic_scholar_search,pubmed_search",
    )
    retrieve = make_retriever(settings=settings)
    retrieve("q")

    assert calls == ["wikipedia_search"]


def test_retriever_raises_clearly_when_no_search_tool_enabled(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, mcp_enabled_tools="fetch_page")

    with pytest.raises(ValueError, match=r"web_search.*wikipedia_search"):
        make_retriever(settings=settings)


def test_retriever_search_and_fetch_share_one_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real, confirmed bug: the old implementation
    called asyncio.run() separately for the search and the fetch, reusing
    one shared MCPToolClient (and therefore its asyncio.Semaphores, bound
    to whichever loop existed when they were constructed) across both --
    confirmed live this raised "RuntimeError: ... is bound to a different
    event loop" on 100% of calls once DSPy's MIPROv2 started evaluating
    examples across multiple threads. Asserts search+fetch actually run to
    completion together (proving one shared loop, not the crash)."""

    async def fake_call(self: Any, tool_name: str, **kwargs: Any) -> Any:
        if tool_name == "web_search":
            return [{"title": "A", "url": "http://a.com", "snippet": "s"}]
        if tool_name == "fetch_page":
            return "fetched"
        raise AssertionError(tool_name)

    monkeypatch.setattr(MCPToolClient, "call", fake_call)

    settings = _settings(tmp_path)
    retrieve = make_retriever(settings=settings, fetch_top=1)

    context = retrieve("q")

    assert "fetched" in context
