"""Shared tool-call telemetry glue.

``current_question_id`` is a ``ContextVar``, not a parameter, because
``build_pipeline`` constructs each tool's closure once per model and those
closures are shared across every concurrently-running question in
``evals/runner.py``'s ``asyncio.gather(eval_concurrency)`` -- a tool has
no other way to know which question invoked it. ``agents/topology.py::
run_pipeline`` sets it once per question (each ``asyncio.gather``ed
coroutine gets its own context, so concurrent questions don't collide).
Mirrors DR-Tulu's own ``_llm_tool_client_context``/``using_client`` idiom
(docs/dr-tulu-agent-infra.md ss3.3), applied to telemetry instead of client
wiring.

``record_tool_call`` is the "time it, log it, return it" wrapper every
tool factory in this package uses, so that bookkeeping lives in one place.
"""

import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar

from mas_deepr.telemetry import TelemetryTracker

current_question_id: ContextVar[str | None] = ContextVar(
    "current_question_id", default=None
)


def _is_failure_marker(result: object) -> str | None:
    """``MCPToolClient.call`` never raises -- a failed call is signaled by
    an inline ``"[mcp_call_failed: ...]"`` string instead. Recognize it so
    tool-call telemetry can tell success from failure without every tool
    factory re-implementing this check."""
    if isinstance(result, str) and result.startswith("[mcp_call_failed:"):
        return result
    return None


async def record_tool_call[T](
    tracker: TelemetryTracker,
    *,
    tool_name: str,
    provider: str,
    question_id: str | None,
    call: Callable[[], Awaitable[T]],
) -> T:
    """Run ``call()``, record one ``ToolCallRecord`` for it, return the result."""
    start = time.perf_counter()
    result = await call()
    latency_s = time.perf_counter() - start
    tracker.record_tool_call(
        tool_name=tool_name,
        provider=provider,
        question_id=question_id,
        latency_s=latency_s,
        error=_is_failure_marker(result),
    )
    return result
