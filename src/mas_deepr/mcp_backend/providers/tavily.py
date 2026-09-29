"""Tavily web search -- kept as an opt-in fallback provider, not the
default. Moved from the pre-MCP ``tools/search.py`` verbatim (minus the
inline cache get/set, now handled by ``mcp_backend/cache.py``'s decorator
at registration time). Off by default because it's the one paid tool
provider left -- see ``Settings.mcp_enabled_tools``.
"""

import asyncio
import logging
from typing import Any

from tavily import TavilyClient
from tavily.errors import (
    BadRequestError,
    ForbiddenError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Config/auth errors will never succeed on retry -- fail fast on these
# instead of burning ~13s of backoff on a guaranteed-to-fail call.
_NON_RETRYABLE = (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    BadRequestError,
    ForbiddenError,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_not_exception_type(_NON_RETRYABLE),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _tavily_search(
    client: TavilyClient, query: str, max_results: int
) -> dict[str, Any]:
    result: dict[str, Any] = client.search(
        query=query, max_results=max_results, search_depth="basic"
    )
    return result


async def search(query: str, *, max_results: int, api_key: str) -> list[dict[str, str]]:
    """Tavily web search, normalized to ``{"title", "url", "snippet"}``."""
    client = TavilyClient(api_key=api_key)
    raw = await asyncio.to_thread(_tavily_search, client, query, max_results)
    hits = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in raw.get("results", [])[:max_results]
    ]
    logger.info("tavily_search query=%r hits=%d", query, len(hits))
    return hits
