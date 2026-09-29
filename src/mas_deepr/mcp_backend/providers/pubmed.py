"""PubMed search via NCBI E-utilities -- free, no key required (an
optional ``NCBI_API_KEY`` raises the rate limit, not needed to function).

Two-step API: ``esearch`` for matching PMIDs, then ``esummary`` for their
titles/authors/year. Both calls share the same retry policy.
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

_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
_RETRYABLE = (httpx.TransportError, httpx.TimeoutException)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict[str, str | int]
) -> dict:
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data: dict = resp.json()
    return data


async def search(query: str, *, max_results: int) -> list[dict[str, str]]:
    """PubMed search, normalized to ``{"title","url","snippet"}``.

    ``snippet`` is the authors + year (PubMed's esummary doesn't include
    abstracts) -- a real limitation vs. the other providers' full snippet,
    documented rather than papered over; the Browser can ``fetch_page`` the
    returned URL for the abstract if it needs more.
    """
    async with httpx.AsyncClient(timeout=15.0) as client:
        search_data = await _get_json(
            client,
            _ESEARCH_URL,
            {"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
        )
        pmids: list[str] = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            logger.info("pubmed_search query=%r hits=0", query)
            return []

        summary_data = await _get_json(
            client,
            _ESUMMARY_URL,
            {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        )

    result_map = summary_data.get("result", {})
    hits = []
    for pmid in pmids:
        rec = result_map.get(pmid)
        if not rec:
            continue
        authors = ", ".join(a.get("name", "") for a in rec.get("authors", [])[:3])
        year = (rec.get("pubdate") or "").split(" ")[0]
        hits.append(
            {
                "title": rec.get("title", ""),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "snippet": f"{authors} ({year})" if authors else year,
            }
        )
    logger.info("pubmed_search query=%r hits=%d", query, len(hits))
    return hits
