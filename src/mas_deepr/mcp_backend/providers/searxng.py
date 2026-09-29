"""Primary web search: SearXNG, a self-hosted metasearch aggregator.

Genuine $0-marginal-cost live web search -- unlike DR-Tulu's own general
search tool (Serper, paid, same model as Tavily) or its only free "search"
(a static pre-indexed corpus, not live web). SearXNG proxies/scrapes
upstream engines (Google, Bing, ...) rather than calling a licensed API,
so it has no per-query cost but a real availability ceiling: too many
queries from one IP risks upstream rate-limiting/blocking. Retried on
HTTP 429 with backoff; NOT retried on 4xx/5xx that indicate
misconfiguration (e.g. JSON output not enabled server-side -- see
``settings.yml``'s ``search.formats``).

Requires a reachable SearXNG instance (self-hosted, Docker or otherwise --
deployment location is a config detail, see ``Settings.searxng_base_url``,
not decided by this module).
"""

import logging

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError | httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _searxng_get(base_url: str, query: str, max_results: int) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{base_url.rstrip('/')}/search",
            params={"q": query, "format": "json", "pageno": 1},
            headers={"User-Agent": "Mozilla/5.0 (research-agent; +mas-deepr)"},
        )
        resp.raise_for_status()
        data: dict = resp.json()
        return data


async def search(
    query: str, *, max_results: int, base_url: str
) -> list[dict[str, str]]:
    """SearXNG metasearch, normalized to ``{"title", "url", "snippet"}``.

    If the server has JSON output disabled (``search.formats`` in its
    ``settings.yml`` -- disabled by default on many public instances,
    self-hosting with it enabled is the realistic path), this raises a
    ``ValueError`` (not retried -- retrying a config error wastes the
    same backoff every time) rather than silently returning nothing.
    """
    if not base_url:
        raise ValueError(
            "searxng_base_url is not configured -- see Settings.searxng_base_url"
        )
    data = await _searxng_get(base_url, query, max_results)
    if "results" not in data:
        raise ValueError(
            "SearXNG response has no 'results' key -- confirm "
            "search.formats includes 'json' in the instance's settings.yml"
        )
    hits = [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in data["results"][:max_results]
    ]
    logger.info("searxng_search query=%r hits=%d", query, len(hits))
    return hits
