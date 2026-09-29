"""Semantic Scholar Graph API paper search -- free, no key required (an
optional key raises the rate limit but isn't needed to function).
"""

import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _s2_get(query: str, max_results: int, api_key: str | None) -> dict:
    headers = {"x-api-key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            _API_URL,
            params={
                "query": query,
                "limit": max_results,
                "fields": "title,abstract,url,year,authors",
            },
            headers=headers,
        )
        resp.raise_for_status()
        data: dict = resp.json()
        return data


async def search(
    query: str, *, max_results: int, api_key: str | None
) -> list[dict[str, str]]:
    """Semantic Scholar paper search, normalized to ``{"title","url","snippet"}``."""
    data = await _s2_get(query, max_results, api_key)
    hits = [
        {
            "title": p.get("title") or "",
            "url": p.get("url") or "",
            "snippet": (p.get("abstract") or "")[:1000],
        }
        for p in (data.get("data") or [])[:max_results]
    ]
    logger.info("semantic_scholar_search query=%r hits=%d", query, len(hits))
    return hits
