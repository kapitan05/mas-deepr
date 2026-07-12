"""Page-fetch tool: HTTP GET + readability extraction, cached and retried."""

from typing import Annotated

import httpx
import trafilatura
from agent_framework import FunctionTool, tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from mas_deepr.tools.cache import WebCache

_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
def _http_get(url: str, timeout_s: float) -> str:
    resp = httpx.get(
        url,
        timeout=timeout_s,
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (research-agent; +mas-deepr)"},
    )
    resp.raise_for_status()
    return resp.text


async def fetch_page(
    url: str,
    *,
    timeout_s: float,
    max_chars: int,
    cache: WebCache,
) -> str:
    """Fetch a URL and return cleaned main-content text, truncated to ``max_chars``.

    Cached indefinitely by URL. Extraction failures (paywalls, non-HTML
    content, etc.) return an empty-content marker rather than raising, so a
    single bad URL doesn't abort the agent loop.
    """
    cache_key = cache.make_key("fetch", url=url)
    cached = cache.get(cache_key)
    if cached is not None:
        text: str = cached["text"]
        return text

    try:
        html = _http_get(url, timeout_s)
        extracted = trafilatura.extract(html, favor_recall=True) or ""
    except Exception as e:
        extracted = f"[fetch_failed: {type(e).__name__}: {e}]"

    text = extracted[:max_chars]
    cache.set(cache_key, "fetch", {"text": text})
    return text


def make_fetch_page_tool(
    *, timeout_s: float, max_chars: int, cache: WebCache
) -> FunctionTool:
    """Build the MAF-callable fetch tool bound to concrete settings/cache."""

    @tool(
        name="fetch_page",
        description=(
            "Fetch a web page by URL and return its cleaned main text content, "
            "with boilerplate (nav, ads, footers) stripped."
        ),
    )
    async def fetch_page_tool(
        url: Annotated[str, "The absolute URL to fetch."],
    ) -> str:
        return await fetch_page(
            url, timeout_s=timeout_s, max_chars=max_chars, cache=cache
        )

    return fetch_page_tool
