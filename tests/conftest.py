"""Shared pytest fixtures."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastmcp import Client

from mas_deepr.config import Settings
from mas_deepr.mcp_backend import build_server


@pytest.fixture
def mcp_settings(tmp_path: Path) -> Settings:
    """A ``Settings`` instance pointed at throwaway tmp_path state, for
    building an MCP server in tests -- no real provider keys needed unless
    a test explicitly exercises a live-API call."""
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
        searxng_base_url="http://searxng.invalid",
    )


@pytest.fixture
async def mcp_test_client(mcp_settings: Settings) -> AsyncIterator[Client]:
    """A ``fastmcp.Client`` bound directly to an in-process server app --
    no subprocess, no port, deterministic. Mirrors
    docs/dr-tulu-agent-infra.md's transport discussion: for tests, an
    in-memory client beats spawning a real HTTP server."""
    app = build_server(mcp_settings)
    async with Client(app) as client:
        yield client
