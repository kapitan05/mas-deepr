"""Server-level tests: tool registration, via an in-process
``fastmcp.Client`` (see ``conftest.py::mcp_test_client`` -- no subprocess,
no network to a real SearXNG/Crawl4AI/Semantic Scholar/PubMed instance).

Caching behavior is tested directly against ``mcp_backend/cache.py`` in
``test_mcp_cache.py`` instead of through the server: ``build_server``
binds each provider function via ``functools.partial`` at server-build
time, so monkeypatching a provider module's function *after* the server
is built (i.e. inside a test using the ``mcp_test_client`` fixture)
wouldn't actually intercept the call -- the cache decorator already holds
a direct reference to the original function.
"""

import pytest
from fastmcp import Client


@pytest.mark.asyncio
async def test_server_registers_all_six_tools(mcp_test_client: Client) -> None:
    tools = await mcp_test_client.list_tools()
    names = {t.name for t in tools}
    assert names == {
        "web_search",
        "fetch_page",
        "wikipedia_search",
        "semantic_scholar_search",
        "pubmed_search",
        "tavily_search",
    }


@pytest.mark.asyncio
async def test_unknown_tool_call_is_not_a_crash(mcp_test_client: Client) -> None:
    result = await mcp_test_client.call_tool(
        "not_a_real_tool", {"query": "x"}, raise_on_error=False
    )
    assert result.is_error
