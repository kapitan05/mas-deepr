"""``mcp_backend/rate_limit.py`` -- proves the actual claim: a rate
limiter bounds calls/second even when a concurrency semaphore alone
wouldn't (e.g. concurrency=1, but each call completes fast enough to
exceed a hard per-second ceiling)."""

import asyncio
import time

import pytest

from mas_deepr.mcp_backend.rate_limit import concurrency_limited, rate_limited


@pytest.mark.asyncio
async def test_rate_limited_paces_fast_sequential_calls() -> None:
    provider = f"test-provider-{id(object())}"  # unique per test run

    @rate_limited(provider, max_rate=2, time_period=1.0)
    async def fast_call() -> None:
        pass  # completes ~instantly -- concurrency alone wouldn't slow this

    start = time.monotonic()
    for _ in range(4):
        await fast_call()
    elapsed = time.monotonic() - start

    # 4 calls at 2/sec must take at least ~1s (first 2 free, next 2 wait)
    assert elapsed >= 0.9


@pytest.mark.asyncio
async def test_concurrency_limited_bounds_simultaneous_calls() -> None:
    provider = f"test-provider-{id(object())}"
    in_flight = 0
    max_seen = 0

    @concurrency_limited(provider, max_concurrent=2)
    async def slow_call() -> None:
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1

    await asyncio.gather(*(slow_call() for _ in range(6)))

    assert max_seen <= 2
