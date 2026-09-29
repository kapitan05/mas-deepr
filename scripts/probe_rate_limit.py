"""Standalone MCP provider rate-limit probe -- not part of the training
path, used to empirically test a candidate ``semantic_scholar_rate_per_sec``
/``wikipedia_rate_per_sec`` before changing ``config/settings.py``'s
defaults.

The MCP server (``scripts/run_mcp_server.py``) enforces the actual pacing
server-side (``mcp_backend/rate_limit.py``'s per-provider ``AsyncLimiter``s,
read from ``Settings`` at server startup) -- this script can't change that
remotely. The workflow is:

1. Restart the MCP server with the candidate rate, e.g.::

       MAS_SEMANTIC_SCHOLAR_RATE_PER_SEC=1.2 uv run python scripts/run_mcp_server.py

2. Run this probe against it, firing enough concurrent, cache-missing
   calls to actually stress the configured ceiling::

       uv run python scripts/probe_rate_limit.py \
           --tool semantic_scholar_search --num-calls 30

3. Read the real-failure count it prints (``mcp call failed`` results --
   the exact real upstream 429s FastMCP surfaces, confirmed via
   ``fastmcp/server/server.py``'s ``get_http_status_code(e) == 429`` check,
   not a self-inflicted client-side limit). Keep the highest rate that
   stays near-zero failures.

Queries are deliberately varied per call (a topic word + a random suffix)
so ``WebCache`` can't mask the real rate by serving cached hits -- a cache
hit never touches the rate limiter at all (see ``mcp_backend/server.py``'s
composition order: rate limit wraps the raw call, cache wraps *that*).
"""

import argparse
import asyncio
import secrets
import time

from mas_deepr.config import get_settings
from mas_deepr.tools.mcp_client import MCPToolClient

_TOPICS = [
    "protein folding",
    "quantum computing",
    "climate model",
    "neural network",
    "gene therapy",
    "dark matter",
    "vaccine development",
    "machine translation",
    "renewable energy",
    "language model",
]


async def _probe(tool: str, num_calls: int, concurrency: int) -> None:
    settings = get_settings()
    client = MCPToolClient(
        url=settings.mcp_server_url,
        transport=settings.mcp_transport,
        concurrency=concurrency,
        searxng_concurrency=concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    results: list[str] = []

    async def _one(i: int) -> None:
        query = f"{_TOPICS[i % len(_TOPICS)]} {secrets.token_hex(4)}"
        async with semaphore:
            result = await client.call(tool, query=query, max_results=3)
        results.append(str(result))

    async with client:
        start = time.monotonic()
        await asyncio.gather(*(_one(i) for i in range(num_calls)))
        elapsed = time.monotonic() - start

    failures = [r for r in results if r.startswith("[mcp_call_failed:")]
    print(f"tool={tool} num_calls={num_calls} elapsed_s={elapsed:.1f}")
    print(f"successes={num_calls - len(failures)} failures={len(failures)}")
    if failures:
        print("sample failure:", failures[0][:200])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool",
        choices=("semantic_scholar_search", "wikipedia_search", "pubmed_search"),
        required=True,
    )
    parser.add_argument(
        "--num-calls",
        type=int,
        default=30,
        help="Total calls to fire -- enough to actually exceed a per-second "
        "limiter's bucket if the candidate rate is too aggressive.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Client-side concurrency -- the server's own per-provider "
        "AsyncLimiter is what actually paces requests regardless of this "
        "value, but submitting too slowly here would let the limiter's "
        "bucket refill between calls and hide a too-aggressive rate.",
    )
    args = parser.parse_args()
    asyncio.run(_probe(args.tool, args.num_calls, args.concurrency))


if __name__ == "__main__":
    main()
