"""Tests for the generate/grade split (evals/runner.py::generate_benchmark
+ grade_benchmark), ported from DR Tulu's decoupled generate-dataset /
scripts/evaluate.py pattern -- see the DR-Tulu-eval-adoption plan.

``run_pipeline`` is monkeypatched, same convention as test_eval_runner.py --
these exercise the split/resumption/regrading behavior, not the real MAF
agent loop.
"""

from pathlib import Path
from typing import Any, cast

import pytest

from mas_deepr.agents.topology import PipelineResult, ResearchPipeline
from mas_deepr.data.schema import Question
from mas_deepr.evals import runner as runner_module
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.evals.runner import generate_benchmark, grade_benchmark, run_benchmark

_FAKE_PIPELINE = cast(ResearchPipeline, object())


def _patch_run_pipeline(
    monkeypatch: pytest.MonkeyPatch, answers: dict[str, str]
) -> list[str]:
    """Returns the list of question_ids actually generated -- so a test can
    assert resumption skipped the ones it expected to skip."""
    called: list[str] = []

    async def fake_run_pipeline(
        pipeline: object, question: str, *, question_id: str, **kwargs: Any
    ) -> PipelineResult:
        called.append(question_id)
        return PipelineResult(
            question_id=question_id,
            question=question,
            sub_questions=["s"],
            findings=["f"],
            final_answer=answers[question_id],
        )

    monkeypatch.setattr(runner_module, "run_pipeline", fake_run_pipeline)
    return called


class _FakeJudge:
    def __init__(self, rubric_score: float = 0.75) -> None:
        self._rubric_score = rubric_score

    async def grade_browsecomp(self, **kwargs: Any) -> bool:
        return True

    async def grade_research_rubrics(
        self, **kwargs: Any
    ) -> tuple[float, list[dict[str, Any]]]:
        return self._rubric_score, []


def _q(qid: str, answer: str = "Paris") -> Question:
    return Question(
        question_id=qid,
        source="frames",
        split="test",
        prompt="capital of france?",
        answer=answer,
    )


@pytest.mark.asyncio
async def test_split_produces_same_final_scores_as_run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generate_benchmark + grade_benchmark, called separately, must reach
    the exact same final EvalRecords run_benchmark's single-pass wrapper
    does -- the split must not change results, only when/how they happen."""
    questions = [_q("f1", "Paris"), _q("f2", "London")]
    _patch_run_pipeline(monkeypatch, {"f1": "Paris", "f2": "wrong"})
    via_wrapper = await run_benchmark(_FAKE_PIPELINE, questions)

    _patch_run_pipeline(monkeypatch, {"f1": "Paris", "f2": "wrong"})
    generated = await generate_benchmark(_FAKE_PIPELINE, questions)
    via_split = await grade_benchmark(generated, questions)

    assert [(r.question_id, r.score, r.metric) for r in via_wrapper] == [
        (r.question_id, r.score, r.metric) for r in via_split
    ]


@pytest.mark.asyncio
async def test_generate_benchmark_marks_success_ungraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, {"f1": "Paris"})
    records = await generate_benchmark(_FAKE_PIPELINE, [_q("f1")])
    assert records[0].graded is False
    assert records[0].metric == "ungraded"
    assert records[0].final_answer == "Paris"


@pytest.mark.asyncio
async def test_generate_benchmark_marks_infra_error_already_graded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raising(*args: Any, **kwargs: Any) -> PipelineResult:
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(runner_module, "run_pipeline", raising)
    records = await generate_benchmark(_FAKE_PIPELINE, [_q("e1")])
    assert records[0].graded is True
    assert records[0].metric == "error"
    assert "agent exploded" in (records[0].error or "")


@pytest.mark.asyncio
async def test_grade_benchmark_leaves_already_graded_records_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raising(*args: Any, **kwargs: Any) -> PipelineResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(runner_module, "run_pipeline", raising)
    generated = await generate_benchmark(_FAKE_PIPELINE, [_q("e1")])
    # judge=None would raise for a graded=False browsecomp record, but this
    # one is already graded=True (an infra error) -- must pass through
    # without ever touching the judge.
    graded = await grade_benchmark(generated, [_q("e1")], judge=None)
    assert graded[0].metric == "error"
    assert graded[0] is generated[0] or graded[0].error == generated[0].error


@pytest.mark.asyncio
async def test_regrading_does_not_rerun_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct efficiency claim: grading twice with two different judges
    must invoke the pipeline exactly once (at generation), not twice."""
    called = _patch_run_pipeline(monkeypatch, {"r1": "a research report"})
    q = _q("r1")
    q.source = "research_rubrics"
    q.rubrics = None
    from mas_deepr.data.schema import RubricCriterion

    q.rubrics = [RubricCriterion(criterion="c1", weight=1.0, axis="Grounding")]

    generated = await generate_benchmark(_FAKE_PIPELINE, [q])
    assert called == ["r1"]

    judge_a = cast(JudgeClient, _FakeJudge(rubric_score=0.6))
    judge_b = cast(JudgeClient, _FakeJudge(rubric_score=0.9))
    graded_a = await grade_benchmark(generated, [q], judge=judge_a)
    graded_b = await grade_benchmark(generated, [q], judge=judge_b)

    assert called == ["r1"]  # still just the one generation call
    assert graded_a[0].score == 0.6
    assert graded_b[0].score == 0.9


@pytest.mark.asyncio
async def test_generate_benchmark_resumes_skipping_completed_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "generations.jsonl"
    questions = [_q("f1"), _q("f2"), _q("f3")]

    called_first = _patch_run_pipeline(
        monkeypatch, {"f1": "Paris", "f2": "Paris", "f3": "Paris"}
    )
    first = await generate_benchmark(
        _FAKE_PIPELINE, questions[:2], output_path=output_path
    )
    assert sorted(called_first) == ["f1", "f2"]
    assert len(first) == 2

    called_second = _patch_run_pipeline(monkeypatch, {"f3": "Paris"})
    second = await generate_benchmark(
        _FAKE_PIPELINE, questions, output_path=output_path
    )
    # f1/f2 already on disk -- only f3 should have actually run the pipeline.
    assert called_second == ["f3"]
    assert {r.question_id for r in second} == {"f1", "f2", "f3"}
    # Original question order preserved regardless of resumption.
    assert [r.question_id for r in second] == ["f1", "f2", "f3"]


@pytest.mark.asyncio
async def test_generate_benchmark_writes_incrementally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "generations.jsonl"
    _patch_run_pipeline(monkeypatch, {"f1": "Paris", "f2": "Paris"})
    await generate_benchmark(
        _FAKE_PIPELINE, [_q("f1"), _q("f2")], output_path=output_path
    )
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_grade_benchmark_relooks_up_cost_after_grading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cost_lookup passed to grade_benchmark must win over whatever
    generate_benchmark already set -- the judge call itself adds cost
    generate_benchmark couldn't have seen yet."""
    _patch_run_pipeline(monkeypatch, {"f1": "Paris"})
    generated = await generate_benchmark(
        _FAKE_PIPELINE, [_q("f1")], cost_lookup=lambda qid: 0.001
    )
    assert generated[0].cost_usd == 0.001
    graded = await grade_benchmark(generated, [_q("f1")], cost_lookup=lambda qid: 0.009)
    assert graded[0].cost_usd == 0.009
