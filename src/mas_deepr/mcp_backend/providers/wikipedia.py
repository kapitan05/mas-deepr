"""Wikipedia search -- direct MediaWiki REST API call, NOT routed through
SearXNG. This is the deliberate fix for a real gap: SearXNG's ``web_search``
tool fans a query out to *every* enabled engine (Google/Bing/DuckDuckGo
alongside Wikipedia), so it can never be "Wikipedia only" -- the safe,
sanctioned-API path and the risky scraped-engine path were bundled into
one tool. This provider isolates the safe path as its own tool, exactly
like Semantic Scholar and PubMed already are.

Free, no key required. Etiquette (not adversarially enforced, but
documented and worth honoring -- mediawiki.org/wiki/API:Etiquette): serial
requests preferred, stay well under the ceiling, identify yourself with a
real User-Agent, back off on 429. See ``Settings.wikipedia_rate_per_sec``
for the server-side rate limiter that actually holds this regardless of
how many concurrent callers (GRPO rollouts, parallel questions) are
queued behind it.
"""

import logging
import re

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_API_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
# Wikimedia's etiquette explicitly asks for a contactable User-Agent
# identifying the client -- not optional politeness, a documented
# requirement they can and do enforce by blocking bare/default UAs.
_USER_AGENT = (
    "mas-deepr-research/0.1 (https://github.com/kapitan05/mas-deepr; research tool)"
)
_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _wikipedia_get(query: str, max_results: int) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            _API_URL,
            params={"q": query, "limit": max_results},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data: dict = resp.json()
        return data


def _strip_html(text: str) -> str:
    """MediaWiki wraps matched terms in the excerpt with <span
    class="searchmatch">...</span> -- strip it, the Browser wants plain
    text, not markup."""
    return re.sub(r"<[^>]+>", "", text)


async def search(query: str, *, max_results: int) -> list[dict[str, str]]:
    """Wikipedia article search, normalized to ``{"title","url","snippet"}``."""
    data = await _wikipedia_get(query, max_results)
    hits = [
        {
            "title": p.get("title", ""),
            "url": f"https://en.wikipedia.org/wiki/{p.get('key', '')}",
            "snippet": _strip_html(p.get("excerpt", "") or p.get("description", "")),
        }
        for p in (data.get("pages") or [])[:max_results]
    ]
    logger.info("wikipedia_search query=%r hits=%d", query, len(hits))
    return hits
