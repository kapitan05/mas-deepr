"""Named MCP tool-scope presets -- which tools the Browser gets, chosen per
benchmark/training-run rather than hand-edited per invocation.

Formalizes docs/mcp-backend-deployment-checklist.md §3's recommendation into
code. ``"wiki_paper"`` drops ``web_search``/``tavily_search`` entirely
(SearXNG's scraped-engine fan-out carries real ban risk at rollout volume;
Tavily costs real money) for benchmarks/training runs answerable from
Wikipedia + academic paper search alone (FRAMES, ResearchQA, HealthBench,
MuSiQue, HotpotQA).

``"general"`` includes ``tavily_search`` as a paid fallback alongside free
SearXNG -- confirmed live (2026-09-17) that SearXNG's scraped engines
(Bing/DuckDuckGo) degrade (CAPTCHA/connection errors) even at smoke-test
volume, exactly as the checklist predicted. Accepting that cost here is
deliberately scoped to *eval* (one benchmark, run once per milestone) --
the original ban-risk/cost argument against Tavily was specifically about
GRPO *training* volume (many rollouts, repeated), a different order of
magnitude. Don't extend this reasoning to a training-time tool scope
without re-deciding it there.
"""

TOOL_SCOPES: dict[str, str] = {
    "general": (
        "web_search,fetch_page,wikipedia_search,semantic_scholar_search,"
        "pubmed_search,tavily_search"
    ),
    "wiki_paper": "wikipedia_search,fetch_page,semantic_scholar_search,pubmed_search",
}


def get_tool_scope(name: str) -> str:
    try:
        return TOOL_SCOPES[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown tool scope {name!r}. Known: {sorted(TOOL_SCOPES)}"
        ) from e
