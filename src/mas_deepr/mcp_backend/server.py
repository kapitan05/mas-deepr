"""mas-deepr's MCP tool-backend server: a small FastMCP app wrapping 5
curated providers (SearXNG, Crawl4AI, Semantic Scholar, PubMed, Tavily),
each cached via ``mcp_backend/cache.py``. Run standalone with
``scripts/run_mcp_server.py``; the Browser agent talks to it through
``tools/mcp_client.py::MCPToolClient``, never directly.

``build_server(settings)`` is an explicit factory over a ``Settings``
object, not module-global ``os.environ.get(...)`` reads at import time --
matches this repo's ``build_pipeline``/``build_chat_client`` convention of
one central, explicit, testable config object.
"""

import functools
import logging
from typing import Annotated

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from mas_deepr.config import Settings
from mas_deepr.mcp_backend.cache import cached
from mas_deepr.mcp_backend.providers import (
    crawl4ai_fetch,
    pubmed,
    searxng,
    semantic_scholar,
    tavily,
    wikipedia,
)
from mas_deepr.mcp_backend.rate_limit import concurrency_limited, rate_limited
from mas_deepr.tools.cache import WebCache

logger = logging.getLogger(__name__)


def build_server(settings: Settings) -> FastMCP:
    """Construct the FastMCP app for one ``Settings`` configuration.

    Composition order matters: rate/concurrency limiting wraps the raw
    provider call, and caching wraps *that* -- so a cache hit returns
    immediately without ever touching the limiter. A cached answer costs
    nothing against the provider's real quota.
    """
    cache = WebCache(settings.cache_db)

    _search = cached("search", cache)(
        rate_limited(
            "searxng", max_rate=settings.searxng_rate_per_min, time_period=60.0
        )(functools.partial(searxng.search, base_url=settings.searxng_base_url))
    )
    _fetch = cached("fetch", cache)(
        concurrency_limited(
            "crawl4ai", max_concurrent=settings.crawl4ai_max_concurrent
        )(crawl4ai_fetch.fetch)
    )
    _s2 = cached("s2", cache)(
        # aiolimiter's AsyncLimiter treats max_rate as a bucket *capacity*,
        # not a bare rate: `async with limiter` acquires amount=1 by
        # default, and acquire() requires 0 <= amount <= max_rate --
        # confirmed live that AsyncLimiter(0.9, 1.0) makes every single
        # acquire() raise "Amount must be a number between 0 and the
        # maximum capacity" (0.9 < 1), a hard 100% failure, not a rate
        # limit. Scaling both numbers by 10 keeps the same average rate
        # (settings.semantic_scholar_rate_per_sec per second) while giving
        # the bucket capacity room for amount=1 acquisitions.
        rate_limited(
            "semantic_scholar",
            max_rate=settings.semantic_scholar_rate_per_sec * 10,
            time_period=10.0,
        )(
            functools.partial(
                semantic_scholar.search,
                api_key=settings.semantic_scholar_api_key or None,
            )
        )
    )
    _pubmed = cached("pubmed", cache)(
        rate_limited("pubmed", max_rate=settings.pubmed_rate_per_sec)(pubmed.search)
    )
    _wikipedia = cached("wikipedia", cache)(
        rate_limited("wikipedia", max_rate=settings.wikipedia_rate_per_sec)(
            wikipedia.search
        )
    )
    _tavily = cached("tavily", cache)(
        rate_limited("tavily", max_rate=settings.tavily_rate_per_min, time_period=60.0)(
            functools.partial(tavily.search, api_key=settings.tavily_api_key)
        )
    )

    mcp = FastMCP("mas-deepr-tools")

    @mcp.tool
    async def web_search(
        query: Annotated[str, "The search query, in natural language."],
        max_results: Annotated[int, "Max hits to return."] = 5,
    ) -> list[dict[str, str]]:
        """Search the web (SearXNG) and return titles, URLs, and snippets.

        Falls back to Tavily (paid) automatically if SearXNG errors or
        returns zero results -- confirmed live 2026-09-17 that merely
        *offering* ``tavily_search`` as a second, separately-named tool
        doesn't work: the Browser has no way to know SearXNG is degraded
        and never reaches for it on its own (0 calls observed across a
        run where SearXNG failed 29/45 times, all Bing/DuckDuckGo
        CAPTCHAs). This fallback is silent/automatic instead of depending
        on LLM tool-choice judgment.

        Gated only on Tavily actually being configured (an API key
        present) -- *not* on this server's own ``mcp_enabled_tools``,
        which is one static value for this whole process and has no idea
        which scope a given caller is running under. The real per-scope
        guarantee lives one layer up, client-side, in
        ``agents/topology.py::build_pipeline``: a wiki_paper-scoped run
        never attaches a ``web_search`` tool to the Browser at all, so
        this function is architecturally unreachable for that scope
        regardless of what this server's own settings say -- if we're
        here, the caller already decided ``web_search`` (and therefore
        this fallback) was in scope.
        """
        try:
            hits = await _search(query, max_results=max_results)
        except Exception as e:
            hits = []
            logger.warning("web_search (SearXNG) failed: %s", e)
        if not hits and settings.tavily_api_key:
            logger.info("web_search: SearXNG returned no results, trying Tavily")
            hits = await _tavily(query, max_results=max_results)
            # TEMPORARY diagnostic for the CI-only (not locally
            # reproducible) empty-fallback failure -- remove once
            # understood. Logged at WARNING so it's captured on failure
            # regardless of configured log level.
            logger.warning(
                "DIAG tavily fallback hits=%r tavily.search id=%r api_key=%r",
                hits,
                id(tavily.search),
                settings.tavily_api_key,
            )
        return hits

    @mcp.tool
    async def fetch_page(
        url: Annotated[str, "The absolute URL to fetch."],
        max_chars: Annotated[int, "Truncate extracted text to this many chars."] = 8000,
        timeout_s: Annotated[float, "Per-page timeout in seconds."] = 20.0,
    ) -> str:
        """Fetch a web page (Crawl4AI) and return cleaned main-text content."""
        return await _fetch(url, timeout_s=timeout_s, max_chars=max_chars)

    @mcp.tool
    async def semantic_scholar_search(
        query: Annotated[str, "Academic search query."],
        max_results: Annotated[int, "Max papers to return."] = 5,
    ) -> list[dict[str, str]]:
        """Search academic papers via Semantic Scholar (free)."""
        return await _s2(query, max_results=max_results)

    @mcp.tool
    async def pubmed_search(
        query: Annotated[str, "Biomedical search query."],
        max_results: Annotated[int, "Max results to return."] = 5,
    ) -> list[dict[str, str]]:
        """Search biomedical literature via PubMed (free, no key required)."""
        return await _pubmed(query, max_results=max_results)

    @mcp.tool
    async def wikipedia_search(
        query: Annotated[str, "Encyclopedic search query."],
        max_results: Annotated[int, "Max articles to return."] = 5,
    ) -> list[dict[str, str]]:
        """Search Wikipedia directly (not via SearXNG) and return titles,
        URLs, and excerpts -- a sanctioned API call, not scraping, so it
        carries none of web_search's upstream-ban risk. Prefer this over
        web_search when the answer is likely to be on Wikipedia."""
        return await _wikipedia(query, max_results=max_results)

    @mcp.tool
    async def tavily_search(
        query: Annotated[str, "The search query, in natural language."],
        max_results: Annotated[int, "Max hits to return."] = 5,
    ) -> list[dict[str, str]]:
        """Search the web via Tavily (paid) -- opt-in fallback, off by
        default; enable via MAS_MCP_ENABLED_TOOLS if SearXNG is
        rate-limited or a benchmark needs a more reliable backend."""
        return await _tavily(query, max_results=max_results)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return mcp
