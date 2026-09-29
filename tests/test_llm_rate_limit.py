"""Tests for the proactive TPM limiter (llm/rate_limit.py). No real API
calls -- these test the pacing primitive itself, not any provider."""

import asyncio
import time

import pytest

from mas_deepr.llm.rate_limit import (
    NullTpmLimiter,
    TpmLimiter,
    build_tpm_limiter,
    estimate_tokens,
)


def test_estimate_tokens_scales_with_length_and_adds_headroom() -> None:
    short = estimate_tokens("hi")
    long = estimate_tokens("word " * 1000)
    assert short < long
    # headroom (500) dominates for a tiny prompt.
    assert short == 1 + 500


def test_build_tpm_limiter_none_gives_null_limiter() -> None:
    assert isinstance(build_tpm_limiter(None), NullTpmLimiter)


def test_build_tpm_limiter_int_gives_real_limiter() -> None:
    assert isinstance(build_tpm_limiter(30_000), TpmLimiter)


@pytest.mark.asyncio
async def test_null_limiter_never_blocks() -> None:
    limiter = NullTpmLimiter()
    start = time.perf_counter()
    for _ in range(50):
        await limiter.acquire(10_000)  # would badly exceed any real budget
    assert time.perf_counter() - start < 0.1


@pytest.mark.asyncio
async def test_real_limiter_paces_calls_within_budget() -> None:
    """1200 tokens/60s (20/s leak rate): fill the bucket to 1199, then ask
    for 10 more -- the first acquire goes through immediately, the second
    must wait ~9/20s=0.45s for the bucket to leak enough capacity back.
    Proves acquire() actually paces instead of passing every call through
    (sized to keep the test fast, not to mirror a real TPM number)."""
    limiter = TpmLimiter(1200)
    start = time.perf_counter()
    await limiter.acquire(1199)
    first_done = time.perf_counter() - start
    await limiter.acquire(10)
    second_done = time.perf_counter() - start
    assert first_done < 0.05
    assert second_done > 0.3  # had to wait for the bucket to leak


@pytest.mark.asyncio
async def test_real_limiter_clamps_amount_above_tpm_limit() -> None:
    """A single call estimated above the whole per-minute budget must
    still go through (clamped to the ceiling) rather than raising --
    aiolimiter itself raises ValueError if amount > max_rate."""
    limiter = TpmLimiter(100)
    await asyncio.wait_for(limiter.acquire(10_000), timeout=1.0)
