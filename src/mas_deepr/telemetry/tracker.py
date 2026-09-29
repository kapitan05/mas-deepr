"""Per-LLM-call telemetry: tokens, latency, cost. JSONL sink, polars reader.

Every agent invocation records one ``LLMCallRecord``. Never log prompt/PII
content here -- only counts and identifiers.
"""

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from mas_deepr.config.models import ModelSpec, cost_usd


class LLMCallRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run_id: str
    phase: str  # baseline | post-dspy | post-grpo | dev
    role: str  # manager | browser | synthesizer | judge
    model_key: str
    question_id: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    memory_strategy: str | None = None  # Phase 3: which memory strategy ran this call
    pass_index: int | None = None  # Phase 3: which memory pass this call belongs to


class ToolCallRecord(BaseModel):
    """One Browser tool call (web_search, fetch_page, ...), mirroring
    ``LLMCallRecord``'s shape but for the MCP tool backend instead of LLM
    calls. Written to a sibling ``tool_calls.jsonl`` next to the LLM sink --
    see ``TelemetryTracker.record_tool_call``/``summarize_tool_calls``.
    """

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run_id: str
    phase: str
    # web_search | fetch_page | semantic_scholar_search | pubmed_search | tavily_search
    tool_name: str
    provider: str  # searxng | crawl4ai | semantic_scholar | pubmed | tavily
    question_id: str | None = None
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


# Per-call estimated cost by provider. Only Tavily is nonzero -- confirm
# against its actual rate card before quoting a number in the thesis, same
# "confirm before quoting" convention as config/models.py's pricing table.
TOOL_COST_PER_CALL: dict[str, float] = {
    "tavily": 0.005,
    "searxng": 0.0,
    "crawl4ai": 0.0,
    "semantic_scholar": 0.0,
    "pubmed": 0.0,
}


class TelemetryTracker:
    """Thread-safe JSONL appender for LLM call records."""

    def __init__(self, sink: Path, run_id: str, phase: str) -> None:
        self.sink = sink
        # Sibling file, same directory as the LLM-call sink -- a separate
        # schema (ToolCallRecord vs. LLMCallRecord), not a separate run.
        self.tool_sink = sink.parent / "tool_calls.jsonl"
        self.run_id = run_id
        self.phase = phase
        self._lock = threading.Lock()
        self._cost_by_qid: dict[str, float] = {}
        self._tool_cost_by_qid: dict[str, float] = {}
        sink.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        role: str,
        spec: ModelSpec,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        question_id: str | None = None,
        error: str | None = None,
        memory_strategy: str | None = None,
        pass_index: int | None = None,
    ) -> LLMCallRecord:
        rec = LLMCallRecord(
            run_id=self.run_id,
            phase=self.phase,
            role=role,
            model_key=spec.key,
            question_id=question_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency_s,
            cost_usd=cost_usd(spec, input_tokens, output_tokens),
            error=error,
            memory_strategy=memory_strategy,
            pass_index=pass_index,
        )
        line = json.dumps(rec.model_dump(), ensure_ascii=False)
        with self._lock, self.sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            if question_id is not None:
                self._cost_by_qid[question_id] = (
                    self._cost_by_qid.get(question_id, 0.0) + rec.cost_usd
                )
        return rec

    def cost_for(self, question_id: str) -> float:
        """Total USD recorded so far for one question's LLM calls."""
        with self._lock:
            return self._cost_by_qid.get(question_id, 0.0)

    def record_tool_call(
        self,
        *,
        tool_name: str,
        provider: str,
        question_id: str | None,
        latency_s: float,
        error: str | None = None,
    ) -> ToolCallRecord:
        rec = ToolCallRecord(
            run_id=self.run_id,
            phase=self.phase,
            tool_name=tool_name,
            provider=provider,
            question_id=question_id,
            latency_s=latency_s,
            cost_usd=TOOL_COST_PER_CALL.get(provider, 0.0),
            error=error,
        )
        line = json.dumps(rec.model_dump(), ensure_ascii=False)
        with self._lock, self.tool_sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            if question_id is not None:
                self._tool_cost_by_qid[question_id] = (
                    self._tool_cost_by_qid.get(question_id, 0.0) + rec.cost_usd
                )
        return rec

    def tool_cost_for(self, question_id: str) -> float:
        """Total USD recorded so far for one question's tool calls."""
        with self._lock:
            return self._tool_cost_by_qid.get(question_id, 0.0)


def usage_from_response(resp: Any) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from a MAF ``AgentResponse``."""
    ud = resp.usage_details
    if ud is None:
        return 0, 0
    return int(ud.get("input_token_count") or 0), int(ud.get("output_token_count") or 0)


class Timer:
    """``with Timer() as t: ...`` then ``t.elapsed_s``."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_s = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_s = time.perf_counter() - self._start


# Explicit schemas for the two JSONL record shapes -- required, not just
# nice-to-have. ``pl.read_ndjson``'s default ``infer_schema_length=100``
# only samples the first 100 rows to guess each column's dtype; ``error``
# is null on nearly every row (it's populated only on failed calls) and a
# sink that's been appended to across many past invocations (a real
# scenario for run_milestone_eval.py's smoke-test loop, which intentionally
# reuses the same path run after run) can easily have >100 all-null rows
# before the first real error string shows up. Polars then commits to
# Null-type for that column and crashes the moment a genuine string
# appears: ``ComputeError: got non-null value for NULL-typed column``.
# Confirmed live 2026-09-25 on exactly this sequence. An explicit schema
# sidesteps inference entirely instead of just widening the sample.
_LLM_CALL_SCHEMA: dict[str, type[pl.DataType]] = {
    "record_id": pl.Utf8,
    "ts": pl.Utf8,
    "run_id": pl.Utf8,
    "phase": pl.Utf8,
    "role": pl.Utf8,
    "model_key": pl.Utf8,
    "question_id": pl.Utf8,
    "input_tokens": pl.Int64,
    "output_tokens": pl.Int64,
    "latency_s": pl.Float64,
    "cost_usd": pl.Float64,
    "error": pl.Utf8,
    "memory_strategy": pl.Utf8,
    "pass_index": pl.Int64,
}

_TOOL_CALL_SCHEMA: dict[str, type[pl.DataType]] = {
    "record_id": pl.Utf8,
    "ts": pl.Utf8,
    "run_id": pl.Utf8,
    "phase": pl.Utf8,
    "tool_name": pl.Utf8,
    "provider": pl.Utf8,
    "question_id": pl.Utf8,
    "latency_s": pl.Float64,
    "cost_usd": pl.Float64,
    "error": pl.Utf8,
}


def read_telemetry(
    sink: Path, schema: dict[str, type[pl.DataType]] = _LLM_CALL_SCHEMA
) -> pl.DataFrame:
    return pl.read_ndjson(sink, schema=schema)


def summarize(sink: Path) -> pl.DataFrame:
    """Aggregate tokens/cost/latency per run, phase, model, role.

    Note ``run_id`` is only as unique as the caller made it -- a sink path
    that's reused across invocations (see run_milestone_eval.py's
    smoke-test loop) will show one row per *historical* run_id, not just
    the invocation that just ran. Give each invocation its own run_id
    suffix if the printed table should reflect only "this run".
    """
    return (
        read_telemetry(sink, _LLM_CALL_SCHEMA)
        .group_by(["run_id", "phase", "model_key", "role"])
        .agg(
            pl.len().alias("calls"),
            pl.sum("input_tokens"),
            pl.sum("output_tokens"),
            pl.sum("cost_usd"),
            pl.mean("latency_s").alias("mean_latency_s"),
        )
        .sort(["run_id", "role"])
    )


def summarize_tool_calls(sink: Path) -> pl.DataFrame:
    """Aggregate tool-call count/cost/latency per run, phase, provider --
    the ``tool_calls.jsonl`` analog of ``summarize()``. A provider row with
    ``cost_usd == 0`` for every call is the "$0 beyond the paid fallback"
    audit line the MCP-adoption plan asks for. Same run_id-uniqueness
    caveat as ``summarize()`` applies."""
    return (
        read_telemetry(sink, _TOOL_CALL_SCHEMA)
        .group_by(["run_id", "phase", "provider", "tool_name"])
        .agg(
            pl.len().alias("calls"),
            pl.sum("cost_usd"),
            pl.mean("latency_s").alias("mean_latency_s"),
            pl.col("error").is_not_null().sum().alias("errors"),
        )
        .sort(["run_id", "provider"])
    )
