"""mas-deepr's own MCP tool-backend server.

Not a port of DR-Tulu's ``mcp_backend/`` (github.com/rlresearch/dr-tulu) --
that file exposes 15+ providers tied to their own project layout. This is a
small, curated server (5 tools: SearXNG search, Crawl4AI fetch, Semantic
Scholar, PubMed, Tavily as an opt-in fallback) reusing only the *shape* of
their design (FastMCP app, a signature-aware cache decorator, per-provider
concurrency). See docs/dr-tulu-agent-infra.md for the source analysis and
the plan this package implements.
"""

from mas_deepr.mcp_backend.server import build_server

__all__ = ["build_server"]
