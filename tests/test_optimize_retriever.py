"""``make_retriever``'s returned function is deliberately synchronous (it
bridges into async tool coroutines via ``asyncio.run`` internally, since
DSPy's MIPROv2 compile loop is sync) -- so these tests call it from plain
sync test functions, not ``@pytest.mark.asyncio`` ones, which would already
have a running event loop and make ``asyncio.run`` raise.
"""

from pathlib import Path
from typing import Any

import pytest

from mas_deepr.config import Settings
from mas_deepr.optimize import modules as modules_module
from mas_deepr.optimize.modules import make_retriever
from mas_deepr.tools.cache import WebCache


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
    )


def test_retriever_formats_hits_and_fetches_top_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_web_search(query: str, **kwargs: Any) -> list[dict[str, str]]:
        return [
            {"title": "A", "url": "http://a.com", "snippet": "snippet a"},
            {"title": "B", "url": "http://b.com", "snippet": "snippet b"},
        ]

    async def fake_fetch_page(url: str, **kwargs: Any) -> str:
        return f"full page text for {url}"

    monkeypatch.setattr(modules_module, "web_search", fake_web_search)
    monkeypatch.setattr(modules_module, "fetch_page", fake_fetch_page)

    settings = _settings(tmp_path)
    cache = WebCache(tmp_path / "cache.sqlite3")
    retrieve = make_retriever(settings=settings, cache=cache, top_k=2, fetch_top=1)

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
    async def fake_web_search(query: str, **kwargs: Any) -> list[dict[str, str]]:
        return []

    monkeypatch.setattr(modules_module, "web_search", fake_web_search)

    settings = _settings(tmp_path)
    cache = WebCache(tmp_path / "cache.sqlite3")
    retrieve = make_retriever(settings=settings, cache=cache)

    assert retrieve("no hits query") == "No search results found."
