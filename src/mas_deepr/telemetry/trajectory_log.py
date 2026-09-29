"""Incremental question/answer trajectory log for benchmark eval runs.

**Deliberate, narrow exception to ``tracker.py``'s "never log prompt/
response text" rule.** That rule stays in force for the general-purpose
per-LLM-call ledger, because that module might one day sit behind
non-benchmark traffic. This module is scoped specifically to eval runs
against public academic benchmarks (FRAMES/BrowseComp/ResearchRubrics) --
no live user data ever flows through it -- and logging the question,
Browser findings, and final answer is exactly what's needed to read a
run's actual research trajectory, not just its score. Don't reuse this
module for anything that isn't a public-benchmark eval run.

Same shape as ``tracker.py::TelemetryTracker`` on purpose: thread-safe
append-only JSONL, one record per event. Written incrementally -- one line
per question, as it finishes -- so a crash partway through a long
benchmark run still leaves every completed question's full trajectory on
disk, unlike ``evals/runner.py::write_results()``, which only writes once
after every question in a benchmark has finished.
"""

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mas_deepr.evals.runner import EvalRecord


class TrajectoryRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    run_id: str
    phase: str
    model_key: str
    question_id: str
    source: str
    question: str
    gold_answer: str | None = None
    sub_questions: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    final_answer: str = ""
    metric: str
    score: float
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None


class TrajectoryLogger:
    """Thread-safe JSONL appender for eval-run trajectories."""

    def __init__(self, sink: Path, *, run_id: str, phase: str, model_key: str) -> None:
        self.sink = sink
        self.run_id = run_id
        self.phase = phase
        self.model_key = model_key
        self._lock = threading.Lock()
        sink.parent.mkdir(parents=True, exist_ok=True)

    def record(self, eval_record: "EvalRecord") -> TrajectoryRecord:
        rec = TrajectoryRecord(
            run_id=self.run_id,
            phase=self.phase,
            model_key=self.model_key,
            question_id=eval_record.question_id,
            source=eval_record.source,
            question=eval_record.question,
            gold_answer=eval_record.gold_answer,
            sub_questions=eval_record.sub_questions,
            findings=eval_record.findings,
            final_answer=eval_record.final_answer,
            metric=eval_record.metric,
            score=eval_record.score,
            latency_s=eval_record.latency_s,
            cost_usd=eval_record.cost_usd,
            error=eval_record.error,
        )
        line = json.dumps(rec.model_dump(), ensure_ascii=False)
        with self._lock, self.sink.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return rec
