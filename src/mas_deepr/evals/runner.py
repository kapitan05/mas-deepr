"""Benchmark runner: drives the pipeline over a question set and grades results.

Grading dispatch: verifiable-answer sources (frames/musique/hotpotqa, keyed
by ``Question.source``) use exact match; ``browsecomp`` (keyed by source
name -- its own grader template, not rubric-based) and any source with
``Question.rubrics`` set (research_rubrics, research_qa, health_bench, ...
-- keyed by *having rubrics*, not by name, so a new rubric-bearing source
needs no new branch here) go through a ``JudgeClient``. Concurrency is
bounded so eval runs don't hammer the inference endpoint or the
web-cache-backed tools.

Each ``EvalRecord`` also carries the full-run wall-clock ``latency_s`` and
the summed ``cost_usd`` for that question, plus the question text, gold
answer, and Browser ``findings`` -- so the accuracy/cost/latency plots and
the trajectory log (``telemetry/trajectory_log.py``) both read off this one
record, no separate join against telemetry or the pipeline needed.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from mas_deepr.agents import ResearchPipeline, run_pipeline
from mas_deepr.data.schema import Question
from mas_deepr.evals.graders import exact_match, extract_final_answer
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.memory import MemoryStrategy
from mas_deepr.telemetry import Timer

if TYPE_CHECKING:
    from mas_deepr.telemetry.trajectory_log import TrajectoryLogger

_VERIFIABLE_SOURCES = {"frames", "musique", "hotpotqa"}


@dataclass
class EvalRecord:
    question_id: str
    source: str
    metric: str
    score: float
    final_answer: str = ""
    sub_questions: list[str] = field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0  # full-run wall-clock for this question
    cost_usd: float = 0.0  # summed LLM-call cost for this question
    question: str = ""  # original prompt text -- public-benchmark content only
    gold_answer: str | None = None  # reference answer, if the source has one
    findings: list[str] = field(default_factory=list)  # Browser's raw per-sub-q answers
    # False => generated but not yet graded (metric/score are placeholders).
    # An infra-error record is written already-graded (metric="error") by
    # generate_benchmark itself -- there's nothing to judge, the answer
    # never happened. See generate_benchmark/grade_benchmark's split.
    graded: bool = False


async def _grade(
    question: Question, final_answer: str, judge: JudgeClient | None
) -> EvalRecord:
    if question.source in _VERIFIABLE_SOURCES:
        assert question.answer is not None
        # Grade the terse "FINAL ANSWER: ..." line (prompts/templates/
        # synthesizer.yaml), not the full explanation -- exact_match needs
        # a short span to compare against a short gold string, and grading
        # the whole prose response against it scored 0 unconditionally
        # (see graders.py::extract_final_answer's docstring for the real
        # baseline run this was caught on). ``final_answer`` on the record
        # stays the full text either way -- that's what trajectory logs/
        # human review want to see.
        candidate = extract_final_answer(final_answer)
        ok = exact_match(candidate, question.answer, question.answer_aliases)
        return EvalRecord(
            question_id=question.question_id,
            source=question.source,
            metric="exact_match",
            score=1.0 if ok else 0.0,
            final_answer=final_answer,
        )

    if judge is None:
        raise ValueError(
            f"source={question.source!r} requires a JudgeClient but none was provided"
        )

    if question.source == "browsecomp":
        assert question.answer is not None
        ok = await judge.grade_browsecomp(
            question=question.prompt,
            correct_answer=question.answer,
            response=final_answer,
            question_id=question.question_id,
        )
        return EvalRecord(
            question_id=question.question_id,
            source=question.source,
            metric="browsecomp_judge",
            score=1.0 if ok else 0.0,
            final_answer=final_answer,
        )

    if question.rubrics is not None:
        # Dispatched by "has rubrics", not by a hardcoded source name --
        # any rubric-bearing source (research_rubrics, research_qa,
        # health_bench, ...) reaches this same judge path for free, no
        # per-source grading branch needed.
        score, _verdicts = await judge.grade_research_rubrics(
            prompt=question.prompt,
            rubrics=question.rubrics,
            response=final_answer,
            question_id=question.question_id,
        )
        return EvalRecord(
            question_id=question.question_id,
            source=question.source,
            metric="rubric_compliance",
            score=score,
            final_answer=final_answer,
        )

    raise ValueError(f"Unknown question source: {question.source!r}")


def _record_to_json(record: EvalRecord) -> str:
    return json.dumps(asdict(record), ensure_ascii=False)


def _load_existing_records(output_path: Path) -> dict[str, EvalRecord]:
    """Records already on disk from a prior (possibly interrupted)
    ``generate_benchmark`` call, keyed by ``question_id`` -- the resumption
    check. Mirrors DR Tulu's own ``generate-dataset`` pattern (read the
    existing output file, collect completed ids, skip them) rather than
    mas-deepr's previous behavior of always starting clean."""
    if not output_path.exists():
        return {}
    existing: dict[str, EvalRecord] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = EvalRecord(**json.loads(line))
            existing[rec.question_id] = rec
    return existing


async def generate_benchmark(
    pipeline: ResearchPipeline,
    questions: list[Question],
    *,
    concurrency: int = 4,
    max_sub_queries: int = 4,
    max_tool_calls_per_query: int = 8,
    memory: MemoryStrategy | None = None,
    cost_lookup: Callable[[str], float] | None = None,
    output_path: Path | None = None,
    trajectory_logger: "TrajectoryLogger | None" = None,
) -> list[EvalRecord]:
    """Run every question through the pipeline -- generation only, no
    grading. Returns ``EvalRecord``s with ``graded=False`` (placeholder
    ``metric="ungraded"``/``score=0.0``) for a successful run, or an
    already-final ``graded=True`` error record (``metric="error"``) for an
    infra failure -- there's nothing to judge when the pipeline itself
    never produced an answer.

    ``output_path``, if given, enables resumption (DR Tulu's
    ``generate-dataset`` pattern): question_ids already present in that
    file are skipped instead of re-run, and each newly-generated record is
    appended to it as it finishes -- not collected in memory and written
    once at the end, so a crash partway through a long run costs at most
    the one in-flight question, not every completed one.

    ``cost_lookup`` maps a ``question_id`` to the total USD spent on its
    LLM calls (typically ``TelemetryTracker.cost_for``). Kept as a callable
    so this module stays decoupled from the telemetry sink path.

    ``trajectory_logger``, if given, gets one ``.record(...)`` call per
    *error* record only, as it happens -- a successful record's trajectory
    is logged by ``grade_benchmark`` instead, once it's actually final
    (avoids logging a placeholder score that a graded pass immediately
    supersedes).
    """
    existing = _load_existing_records(output_path) if output_path else {}
    remaining = [q for q in questions if q.question_id not in existing]

    semaphore = asyncio.Semaphore(concurrency)

    async def _one(question: Question) -> EvalRecord:
        async with semaphore:
            try:
                with Timer() as t:
                    result = await run_pipeline(
                        pipeline,
                        question.prompt,
                        question_id=question.question_id,
                        max_sub_queries=max_sub_queries,
                        max_tool_calls_per_query=max_tool_calls_per_query,
                        memory=memory,
                    )
                record = EvalRecord(
                    question_id=question.question_id,
                    source=question.source,
                    metric="ungraded",
                    score=0.0,
                    final_answer=result.final_answer,
                    sub_questions=result.sub_questions,
                    latency_s=t.elapsed_s,
                    question=question.prompt,
                    gold_answer=question.answer,
                    findings=result.findings,
                    graded=False,
                )
            except Exception as e:
                record = EvalRecord(
                    question_id=question.question_id,
                    source=question.source,
                    metric="error",
                    score=0.0,
                    error=f"{type(e).__name__}: {e}",
                    question=question.prompt,
                    gold_answer=question.answer,
                    graded=True,
                )
            if cost_lookup is not None:
                record.cost_usd = cost_lookup(question.question_id)
            if record.graded and trajectory_logger is not None:
                trajectory_logger.record(record)
            if output_path is not None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("a", encoding="utf-8") as f:
                    f.write(_record_to_json(record) + "\n")
            return record

    new_records = await asyncio.gather(*[_one(q) for q in remaining])
    by_id = {**existing, **{r.question_id: r for r in new_records}}
    # Preserve caller's original question order -- asyncio.gather already
    # preserves it for `remaining`, but resumption can interleave reloaded
    # (`existing`) and freshly-generated records arbitrarily otherwise.
    return [by_id[q.question_id] for q in questions]


async def grade_benchmark(
    records: list[EvalRecord],
    questions: list[Question],
    *,
    judge: JudgeClient | None = None,
    concurrency: int = 8,
    cost_lookup: Callable[[str], float] | None = None,
    trajectory_logger: "TrajectoryLogger | None" = None,
) -> list[EvalRecord]:
    """Grade ``generate_benchmark``'s output. Independent concurrency from
    generation on purpose -- a judge call is cheap/fast relative to a full
    Manager/Browser/Synthesizer research loop, no reason to share its
    (typically much lower) concurrency budget.

    A record with ``graded=True`` already (an infra-error record from
    generation) passes through unchanged -- no judge call, nothing to log
    again (``generate_benchmark`` already logged it). Only a genuinely new
    grading failure here becomes an ``"error"`` record; unlike the old
    single-pass ``run_benchmark``, a judge failure no longer re-runs the
    (already-succeeded, expensive) generation step to retry it -- a
    direct, deliberate efficiency gain from the split, not an oversight.

    ``cost_lookup``, if given, re-reads the question's total cost *after*
    grading -- the judge call itself adds cost that ``generate_benchmark``
    couldn't see yet when it first populated ``cost_usd``.
    """
    questions_by_id = {q.question_id: q for q in questions}
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(record: EvalRecord) -> EvalRecord:
        if record.graded:
            return record
        async with semaphore:
            question = questions_by_id[record.question_id]
            try:
                graded = await _grade(question, record.final_answer, judge)
            except Exception as e:
                graded = EvalRecord(
                    question_id=record.question_id,
                    source=record.source,
                    metric="error",
                    score=0.0,
                    error=f"{type(e).__name__}: {e}",
                )
            # _grade() only sets question_id/source/metric/score/final_answer
            # -- carry over everything generate_benchmark already filled in.
            graded.sub_questions = record.sub_questions
            graded.latency_s = record.latency_s
            graded.question = record.question
            graded.gold_answer = record.gold_answer
            graded.findings = record.findings
            graded.cost_usd = (
                cost_lookup(record.question_id)
                if cost_lookup is not None
                else record.cost_usd
            )
            graded.graded = True
            if trajectory_logger is not None:
                trajectory_logger.record(graded)
            return graded

    return await asyncio.gather(*[_one(r) for r in records])


async def run_benchmark(
    pipeline: ResearchPipeline,
    questions: list[Question],
    *,
    judge: JudgeClient | None = None,
    concurrency: int = 4,
    max_sub_queries: int = 4,
    max_tool_calls_per_query: int = 8,
    memory: MemoryStrategy | None = None,
    cost_lookup: Callable[[str], float] | None = None,
    trajectory_logger: "TrajectoryLogger | None" = None,
) -> list[EvalRecord]:
    """Run every question through the pipeline and grade it -- a thin
    ``generate_benchmark`` + ``grade_benchmark`` wrapper (no resumption:
    ``output_path=None``) kept for callers that don't need the split.
    Same signature/return shape as before that split existed.
    """
    generated = await generate_benchmark(
        pipeline,
        questions,
        concurrency=concurrency,
        max_sub_queries=max_sub_queries,
        max_tool_calls_per_query=max_tool_calls_per_query,
        memory=memory,
        cost_lookup=cost_lookup,
        trajectory_logger=trajectory_logger,
    )
    return await grade_benchmark(
        generated,
        questions,
        judge=judge,
        concurrency=concurrency,
        cost_lookup=cost_lookup,
        trajectory_logger=trajectory_logger,
    )


def records_to_df(records: list[EvalRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "question_id": r.question_id,
                "source": r.source,
                "metric": r.metric,
                "score": r.score,
                "final_answer": r.final_answer,
                "sub_questions": r.sub_questions,
                "error": r.error,
                "latency_s": r.latency_s,
                "cost_usd": r.cost_usd,
                "question": r.question,
                "gold_answer": r.gold_answer,
                "findings": r.findings,
                "graded": r.graded,
            }
            for r in records
        ]
    )


def write_results(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_df(records).write_parquet(path)
