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


class TelemetryTracker:
    """Thread-safe JSONL appender for LLM call records."""

    def __init__(self, sink: Path, run_id: str, phase: str) -> None:
        self.sink = sink
        self.run_id = run_id
        self.phase = phase
        self._lock = threading.Lock()
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
        )
        line = json.dumps(rec.model_dump(), ensure_ascii=False)
        with self._lock, self.sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec


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


def read_telemetry(sink: Path) -> pl.DataFrame:
    return pl.read_ndjson(sink)


def summarize(sink: Path) -> pl.DataFrame:
    """Aggregate tokens/cost/latency per run, phase, model, role."""
    return (
        read_telemetry(sink)
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
