"""Tavily search tool -- opt-in fallback, proxies to the MCP tool
backend's ``tavily_search`` (paid; see ``mcp_backend/providers/tavily.py``).

Only wired onto the Browser when explicitly enabled via
``Settings.mcp_enabled_tools`` (off by default) -- see
``agents/topology.py::build_pipeline``.
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_tavily_tool(
    *, mcp_client: MCPToolClient, tracker: TelemetryTracker, max_results: int
) -> FunctionTool:
    """Build the MAF-callable Tavily search tool."""

    @tool(
        name="tavily_search",
        description=(
            "Search the web (paid, Tavily) and return titles, URLs, and snippets."
        ),
    )
    async def tavily_search_tool(
        query: Annotated[str, "The search query, in natural language."],
    ) -> list[dict[str, str]] | str:
        return await record_tool_call(
            tracker,
            tool_name="tavily_search",
            provider="tavily",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "tavily_search", query=query, max_results=max_results
            ),
        )

    return tavily_search_tool
