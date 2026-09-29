"""Page-fetch tool -- proxies to the MCP tool backend's ``fetch_page``
(Crawl4AI by default; see ``mcp_backend/providers/crawl4ai_fetch.py``).

Provider logic (Crawl4AI, caching, retry) now lives server-side in
``mcp_backend/``; this module is a thin MAF ``@tool`` wrapper around
``MCPToolClient.call(...)`` -- same tool name/schema/return contract as
before the MCP adoption.
"""

from typing import Annotated

from agent_framework import FunctionTool, tool

from mas_deepr.telemetry import TelemetryTracker
from mas_deepr.tools.mcp_client import MCPToolClient
from mas_deepr.tools.tool_telemetry import current_question_id, record_tool_call


def make_fetch_page_tool(
    *,
    mcp_client: MCPToolClient,
    tracker: TelemetryTracker,
    timeout_s: float,
    max_chars: int,
) -> FunctionTool:
    """Build the MAF-callable fetch tool bound to a concrete MCP client."""

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
        return await record_tool_call(
            tracker,
            tool_name="fetch_page",
            provider="crawl4ai",
            question_id=current_question_id.get(),
            call=lambda: mcp_client.call(
                "fetch_page", url=url, timeout_s=timeout_s, max_chars=max_chars
            ),
        )

    return fetch_page_tool
