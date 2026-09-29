"""Tests for the primary fetch provider (Crawl4AI). Mocks
``AsyncWebCrawler`` -- these must not require a real Playwright/Chromium
install to run."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from mas_deepr.mcp_backend.providers.crawl4ai_fetch import fetch


class _FakeResult:
    def __init__(
        self, *, success: bool, markdown: str = "", error_message: str = ""
    ) -> None:
        self.success = success
        self.markdown = markdown
        self.error_message = error_message


def _fake_crawler(result: _FakeResult) -> Any:
    crawler = AsyncMock()
    crawler.arun = AsyncMock(return_value=result)
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=False)
    return crawler


@pytest.mark.asyncio
async def test_fetch_extracts_markdown() -> None:
    result = _FakeResult(success=True, markdown="Hello world, article body.")
    with patch(
        "mas_deepr.mcp_backend.providers.crawl4ai_fetch.AsyncWebCrawler",
        return_value=_fake_crawler(result),
    ):
        text = await fetch("http://example.com", timeout_s=5.0, max_chars=1000)

    assert "Hello world" in text


@pytest.mark.asyncio
async def test_fetch_truncates_to_max_chars() -> None:
    result = _FakeResult(success=True, markdown="word " * 500)
    with patch(
        "mas_deepr.mcp_backend.providers.crawl4ai_fetch.AsyncWebCrawler",
        return_value=_fake_crawler(result),
    ):
        text = await fetch("http://x.com", timeout_s=5.0, max_chars=50)

    assert len(text) <= 50


@pytest.mark.asyncio
async def test_fetch_unsuccessful_result_does_not_raise() -> None:
    result = _FakeResult(success=False, error_message="404 not found")
    with patch(
        "mas_deepr.mcp_backend.providers.crawl4ai_fetch.AsyncWebCrawler",
        return_value=_fake_crawler(result),
    ):
        text = await fetch("http://x.com", timeout_s=5.0, max_chars=100)

    assert "fetch_failed" in text


@pytest.mark.asyncio
async def test_fetch_exception_does_not_raise() -> None:
    crawler = AsyncMock()
    crawler.__aenter__ = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "mas_deepr.mcp_backend.providers.crawl4ai_fetch.AsyncWebCrawler",
        return_value=crawler,
    ):
        text = await fetch("http://x.com", timeout_s=5.0, max_chars=100)

    assert "fetch_failed" in text
