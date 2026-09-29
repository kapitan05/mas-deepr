"""Proactive token-per-minute pacing for frontier LLM calls, keyed per
``ModelSpec``.

Distinct from ``llm_max_retries``/``_build_async_client`` in ``factory.py``
-- that's *reactive*: it retries after a 429 has already happened.  This is
*proactive*: it reserves an estimated token budget against a bucket before
a call is even sent, so most 429s never happen in the first place.

Confirmed live 2026-09-25: gpt-4.1 hit its OpenAI org's 30,000 TPM ceiling
under ``eval_concurrency=4`` (four questions' worth of role-calls in
flight at once, each with a sizeable context). ``eval_concurrency`` bounds
how many questions run *concurrently*, not how many tokens/minute they
collectively burn -- a semaphore is blind to call size, same
concurrency-vs-rate gap already solved server-side for MCP tools in
``mcp_backend/rate_limit.py``. Here, the 429s cascaded into MAF's own
"Maximum consecutive function call errors reached (3)" circuit breaker,
which cut a Browser tool loop short and forced an answer from a truncated
tool budget -- a failure mode strictly worse than added latency, since
retry-after-429 doesn't undo it.

Same underlying primitive as ``mcp_backend/rate_limit.py``
(``aiolimiter.AsyncLimiter``), applied to a different resource -- tokens
per minute instead of calls per second -- and acquired with a variable
``amount`` per call instead of the default ``1``.
"""

from aiolimiter import AsyncLimiter

# Real token count isn't known until the response comes back. Reserve a
# conservative pre-call estimate instead: ~4 chars/token is the commonly
# used rough ratio for English text, plus fixed headroom for the system
# prompt, tool schemas, and the response itself, none of which this
# function sees. Overestimating just paces slightly slower than strictly
# necessary -- the safe failure direction for a rate limiter. Underestimating
# risks a 429 anyway, which the existing SDK-level retry (llm_max_retries)
# still catches.
_CHARS_PER_TOKEN_ESTIMATE = 4
_ESTIMATE_HEADROOM_TOKENS = 500


def estimate_tokens(prompt: str, *, headroom: int = _ESTIMATE_HEADROOM_TOKENS) -> int:
    """Rough pre-call token estimate for reserving TPM bucket capacity.

    KNOWN LIMITATION: this only sees the incremental ``prompt`` text for
    *this* turn, not the full running conversation MAF's ``Agent`` actually
    sends (system prompt + every prior turn's tool calls/results). Early in
    a Browser tool loop that's a fine approximation; deep into one, the
    real request can be far larger than this estimates -- the accumulated
    history dominates, not the new prompt text. This still meaningfully
    reduces 429 risk (paces the high-volume early/short calls correctly,
    and llm_max_retries still catches whatever it misses), but isn't a
    precise reservation. A tighter version would track each pipeline's own
    last-observed ``input_tokens`` per role (telemetry already has it) and
    use that as a floor -- not implemented here.
    """
    return max(1, len(prompt) // _CHARS_PER_TOKEN_ESTIMATE) + headroom


class TpmLimiter:
    """One per model needing proactive TPM pacing (built once in
    ``build_pipeline``, shared across every concurrent question's calls
    for that model -- same lifetime as ``MCPToolClient``).

    ``acquire()`` cooperatively blocks (other coroutines keep running)
    until enough bucket capacity has leaked away for the estimated token
    count, then reserves it -- no matching ``release()``: a leaky bucket
    drains on its own over time, unlike a semaphore.
    """

    def __init__(self, tpm_limit: int) -> None:
        self._limiter = AsyncLimiter(tpm_limit, time_period=60.0)
        self._tpm_limit = tpm_limit

    async def acquire(self, estimated_tokens: int) -> None:
        # AsyncLimiter.acquire raises ValueError if amount > max_rate --
        # a single call estimated above the whole per-minute budget still
        # has to go through (clamped to the ceiling itself), it just
        # doesn't get any pacing headroom left over for calls behind it.
        amount = min(estimated_tokens, self._tpm_limit)
        await self._limiter.acquire(amount)


class NullTpmLimiter:
    """No-op stand-in for a model with no known TPM ceiling (self-hosted
    SLMs served from our own endpoint, or a frontier model we haven't
    confirmed an account-tier cap for yet -- see ``ModelSpec.tpm_limit``).
    Same ``acquire`` interface as ``TpmLimiter``, costs nothing, so
    call sites never need to branch on which one they got."""

    async def acquire(self, estimated_tokens: int) -> None:
        return None


def build_tpm_limiter(tpm_limit: int | None) -> "TpmLimiter | NullTpmLimiter":
    return TpmLimiter(tpm_limit) if tpm_limit is not None else NullTpmLimiter()
