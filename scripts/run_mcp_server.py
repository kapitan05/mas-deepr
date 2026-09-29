"""Start mas-deepr's MCP tool-backend server. Run this first, before
run_baseline.py / run_milestone_eval.py / train_grpo.py -- they connect to
it as a client (Settings.mcp_server_url), they don't spawn it themselves.

Usage:
    uv run python scripts/run_mcp_server.py
    uv run python scripts/run_mcp_server.py --transport stdio  # local dev/inspector

Needs a reachable SearXNG instance (Settings.searxng_base_url /
SEARXNG_BASE_URL) and, for fetch_page, ``crawl4ai-setup`` having been run
once on this machine to install its Playwright/Chromium binary.
"""

import argparse
import logging

from mas_deepr.config import get_settings
from mas_deepr.logging_config import configure_logging
from mas_deepr.mcp_backend import build_server

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["http", "stdio"], default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(logging.getLevelName(args.log_level.upper()))
    settings = get_settings()
    settings.ensure_dirs()

    if not settings.searxng_base_url:
        logger.warning(
            "SEARXNG_BASE_URL is not set -- web_search will fail until a "
            "SearXNG instance is configured. Other tools are unaffected."
        )

    mcp = build_server(settings)
    transport = args.transport or settings.mcp_transport

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        host = args.host or settings.mcp_server_host
        port = args.port or settings.mcp_server_port
        logger.info("Starting MCP server on http://%s:%d/mcp", host, port)
        mcp.run(transport="http", host=host, port=port, path="/mcp")


if __name__ == "__main__":
    main()
