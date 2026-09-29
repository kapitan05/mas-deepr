"""PubMed search tool -- proxies to the MCP tool backend's
``pubmed_search`` (free, see ``mcp_backend/providers/pubmed.py``).
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_pubmed_tool(
    *, mcp_client: MCPToolClient, tracker: TelemetryTracker, max_results: int
) -> FunctionTool:
    """Build the MAF-callable PubMed search tool."""

    @tool(
        name="pubmed_search",
        description=(
            "Search biomedical literature via PubMed and return titles, "
            "URLs, and authors/year."
        ),
    )
    async def pubmed_search_tool(
        query: Annotated[str, "Biomedical search query."],
    ) -> list[dict[str, str]] | str:
        return await record_tool_call(
            tracker,
            tool_name="pubmed_search",
            provider="pubmed",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "pubmed_search", query=query, max_results=max_results
            ),
        )

    return pubmed_search_tool
