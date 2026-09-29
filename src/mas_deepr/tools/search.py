"""Web search tool -- proxies to the MCP tool backend's ``web_search``
(SearXNG by default; see ``mcp_backend/providers/searxng.py``).

Provider logic (Tavily, SearXNG, caching, retry) now lives server-side in
``mcp_backend/``; this module is a thin MAF ``@tool`` wrapper around
``MCPToolClient.call(...)`` -- same tool name/schema/return shape as
before the MCP adoption, so nothing above ``build_pipeline`` needs to know
the provider changed underneath it.
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_web_search_tool(
    *, mcp_client: MCPToolClient, tracker: TelemetryTracker, max_results: int
) -> FunctionTool:
    """Build the MAF-callable search tool bound to a concrete MCP client."""

    @tool(
        name="web_search",
        description="Search the web and return titles, URLs, and snippets.",
    )
    async def web_search_tool(
        query: Annotated[str, "The search query, in natural language."],
    ) -> list[dict[str, str]] | str:
        return await record_tool_call(
            tracker,
            tool_name="web_search",
            provider="searxng",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "web_search", query=query, max_results=max_results
            ),
        )

    return web_search_tool
