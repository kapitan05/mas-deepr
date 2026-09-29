"""Wikipedia search tool -- proxies to the MCP tool backend's
``wikipedia_search`` (free, direct API call, not routed through SearXNG;
see ``mcp_backend/providers/wikipedia.py``).

Deliberately a separate tool from ``web_search``: SearXNG's web_search
fans a query out to every enabled engine (including scraped ones like
Bing/DuckDuckGo), so it can never be "Wikipedia only" and carries their
ban risk. This tool is the isolated, genuinely-ban-risk-free path for
Wikipedia-answerable questions (FRAMES/MuSiQue/HotpotQA) -- see the
MCP-backend deployment checklist's "RL-scale ban risk" section.
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_wikipedia_tool(
    *, mcp_client: MCPToolClient, tracker: TelemetryTracker, max_results: int
) -> FunctionTool:
    """Build the MAF-callable Wikipedia search tool."""

    @tool(
        name="wikipedia_search",
        description=(
            "Search Wikipedia directly and return titles, URLs, and "
            "excerpts. Prefer this over web_search when the answer is "
            "likely to be on Wikipedia."
        ),
    )
    async def wikipedia_search_tool(
        query: Annotated[str, "Encyclopedic search query."],
    ) -> list[dict[str, str]] | str:
        return await record_tool_call(
            tracker,
            tool_name="wikipedia_search",
            provider="wikipedia",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "wikipedia_search", query=query, max_results=max_results
            ),
        )

    return wikipedia_search_tool
