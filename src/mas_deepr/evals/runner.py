"""Benchmark runner: drives the pipeline over a question set and grades results.

Grading is dispatched by ``Question.source``: verifiable-answer sources
(frames/musique/hotpotqa) use exact match; browsecomp/research_rubrics
require a ``JudgeClient``. Concurrency is bounded so eval runs don't hammer
the inference endpoint or the web-cache-backed tools.
"""

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from mas_deepr.agents import ResearchPipeline, run_pipeline
from mas_deepr.data.schema import Question
from mas_deepr.evals.graders import exact_match
from mas_deepr.evals.judge import JudgeClient

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


async def _grade(
    question: Question, final_answer: str, judge: JudgeClient | None
) -> EvalRecord:
    if question.source in _VERIFIABLE_SOURCES:
        assert question.answer is not None
        ok = exact_match(final_answer, question.answer, question.answer_aliases)
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

    if question.source == "research_rubrics":
        assert question.rubrics is not None
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


async def run_benchmark(
    pipeline: ResearchPipeline,
    questions: list[Question],
    *,
    judge: JudgeClient | None = None,
    concurrency: int = 4,
    max_sub_queries: int = 4,
    max_tool_calls_per_query: int = 8,
) -> list[EvalRecord]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(question: Question) -> EvalRecord:
        async with semaphore:
            last_error = ""
            for _attempt in range(2):  # one retry: transient infra hiccups
                # shouldn't cost a question a hard 0 in a milestone score.
                try:
                    result = await run_pipeline(
                        pipeline,
                        question.prompt,
                        question_id=question.question_id,
                        max_sub_queries=max_sub_queries,
                        max_tool_calls_per_query=max_tool_calls_per_query,
                    )
                    record = await _grade(question, result.final_answer, judge)
                    record.sub_questions = result.sub_questions
                    return record
                except Exception as e:
                    last_error = f"{type(e).__name__}: {e}"
            return EvalRecord(
                question_id=question.question_id,
                source=question.source,
                metric="error",
                score=0.0,
                error=last_error,
            )

    return await asyncio.gather(*[_one(q) for q in questions])


def records_to_df(records: list[EvalRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "question_id": r.question_id,
                "source": r.source,
                "metric": r.metric,
                "score": r.score,
                "final_answer": r.final_answer,
                "error": r.error,
            }
            for r in records
        ]
    )


def write_results(records: list[EvalRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_df(records).write_parquet(path)
