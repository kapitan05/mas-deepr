from pathlib import Path

import pytest

from mas_deepr.tools import fetch as fetch_module
from mas_deepr.tools.cache import WebCache
from mas_deepr.tools.fetch import fetch_page


@pytest.mark.asyncio
async def test_fetch_page_extracts_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = (
        "<html><body><article><p>Hello world, this is the article "
        "body.</p></article></body></html>"
    )
    calls = []

    def fake_http_get(url: str, timeout_s: float) -> str:
        calls.append(url)
        return html

    monkeypatch.setattr(fetch_module, "_http_get", fake_http_get)
    cache = WebCache(tmp_path / "cache.sqlite3")

    text1 = await fetch_page(
        "http://example.com", timeout_s=5.0, max_chars=1000, cache=cache
    )
    text2 = await fetch_page(
        "http://example.com", timeout_s=5.0, max_chars=1000, cache=cache
    )

    assert len(calls) == 1
    assert "Hello world" in text1
    assert text1 == text2


@pytest.mark.asyncio
async def test_fetch_page_truncates_to_max_chars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = "<html><body><article><p>" + ("word " * 500) + "</p></article></body></html>"
    monkeypatch.setattr(fetch_module, "_http_get", lambda url, timeout_s: html)
    cache = WebCache(tmp_path / "cache.sqlite3")

    text = await fetch_page("http://x.com", timeout_s=5.0, max_chars=50, cache=cache)
    assert len(text) <= 50


@pytest.mark.asyncio
async def test_fetch_page_failure_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def raise_error(url: str, timeout_s: float) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(fetch_module, "_http_get", raise_error)
    cache = WebCache(tmp_path / "cache.sqlite3")

    text = await fetch_page("http://x.com", timeout_s=5.0, max_chars=100, cache=cache)
    assert "fetch_failed" in text
