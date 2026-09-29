"""Semantic Scholar paper-search tool -- proxies to the MCP tool backend's
``semantic_scholar_search`` (free, see ``mcp_backend/providers/semantic_scholar.py``).
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_semantic_scholar_tool(
    *, mcp_client: MCPToolClient, tracker: TelemetryTracker, max_results: int
) -> FunctionTool:
    """Build the MAF-callable Semantic Scholar search tool."""

    @tool(
        name="semantic_scholar_search",
        description=(
            "Search academic papers via Semantic Scholar and return "
            "titles, URLs, and abstracts."
        ),
    )
    async def semantic_scholar_search_tool(
        query: Annotated[str, "Academic search query."],
    ) -> list[dict[str, str]] | str:
        return await record_tool_call(
            tracker,
            tool_name="semantic_scholar_search",
            provider="semantic_scholar",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "semantic_scholar_search", query=query, max_results=max_results
            ),
        )

    return semantic_scholar_search_tool
