"""Primary fetch: Crawl4AI, a free, self-hosted, Playwright-based crawler.

Replaces the plain httpx+trafilatura fetch outright -- same $0 marginal
cost, but handles JS-rendered pages plain httpx silently can't (no JS
execution), and Crawl4AI's pruning filter gives better signal-to-noise on
long pages than trafilatura's extraction alone.

Requires ``crawl4ai-setup`` (installs a Playwright/Chromium binary)
wherever the MCP server process runs -- a deploy-time step, not something
``uv sync`` provides on its own (see docs/running-phase-1.md).
"""

import logging

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

logger = logging.getLogger(__name__)

_BROWSER_CONFIG = BrowserConfig(headless=True, verbose=False)


async def fetch(url: str, *, timeout_s: float, max_chars: int) -> str:
    """Fetch a URL via a headless browser and return cleaned markdown text,
    truncated to ``max_chars``.

    Never raises -- a failed fetch (bad URL, timeout, render error) returns
    an inline ``"[fetch_failed: ...]"`` marker, mirroring the convention
    the pre-MCP ``tools/fetch.py`` already established, so one bad URL
    doesn't abort the agent loop.
    """
    run_config = CrawlerRunConfig(
        # caching is mcp_backend/cache.py's job, not crawl4ai's
        cache_mode=CacheMode.BYPASS,
        page_timeout=int(timeout_s * 1000),
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter()
        ),
    )
    try:
        async with AsyncWebCrawler(config=_BROWSER_CONFIG) as crawler:
            result = await crawler.arun(url=url, config=run_config)
    except Exception as e:
        logger.warning(
            "crawl4ai_fetch failed url=%s error=%s: %s", url, type(e).__name__, e
        )
        return f"[fetch_failed: {type(e).__name__}: {e}]"[:max_chars]

    if not result.success:
        logger.warning(
            "crawl4ai_fetch unsuccessful url=%s error=%s", url, result.error_message
        )
        return f"[fetch_failed: {result.error_message}]"[:max_chars]

    text = str(result.markdown)[:max_chars]
    logger.info("crawl4ai_fetch ok url=%s chars=%d", url, len(text))
    return text
