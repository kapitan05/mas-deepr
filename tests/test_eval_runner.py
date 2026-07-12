"""Unit tests for the benchmark-dispatch logic in evals/runner.py.

``run_pipeline`` is monkeypatched to a canned coroutine so these exercise
only the grading dispatch (exact-match vs judge vs error), not the real MAF
agent loop -- that's covered separately by test_pipeline_smoke.py.
"""

from typing import Any, cast

import pytest

from mas_deepr.agents.topology import PipelineResult, ResearchPipeline
from mas_deepr.data.schema import Question, RubricCriterion
from mas_deepr.evals import runner as runner_module
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.evals.runner import run_benchmark

_FAKE_PIPELINE = cast(ResearchPipeline, object())


def _patch_run_pipeline(monkeypatch: pytest.MonkeyPatch, final_answer: str) -> None:
    async def fake_run_pipeline(
        pipeline: object, question: str, *, question_id: str, **kwargs: Any
    ) -> PipelineResult:
        return PipelineResult(
            question_id=question_id,
            question=question,
            sub_questions=["s"],
            findings=["f"],
            final_answer=final_answer,
        )

    monkeypatch.setattr(runner_module, "run_pipeline", fake_run_pipeline)


class _FakeJudge:
    def __init__(
        self, *, browsecomp_ok: bool = True, rubric_score: float = 0.75
    ) -> None:
        self._browsecomp_ok = browsecomp_ok
        self._rubric_score = rubric_score

    async def grade_browsecomp(self, **kwargs: Any) -> bool:
        return self._browsecomp_ok

    async def grade_research_rubrics(
        self, **kwargs: Any
    ) -> tuple[float, list[dict[str, Any]]]:
        return self._rubric_score, []


@pytest.mark.asyncio
async def test_exact_match_source_scores_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, "Paris")
    q = Question(
        question_id="f1",
        source="frames",
        split="test",
        prompt="capital of france?",
        answer="Paris",
    )
    records = await run_benchmark(_FAKE_PIPELINE, [q])
    assert records[0].score == 1.0
    assert records[0].metric == "exact_match"


@pytest.mark.asyncio
async def test_exact_match_source_wrong_answer_scores_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, "London")
    q = Question(
        question_id="f2",
        source="frames",
        split="test",
        prompt="capital of france?",
        answer="Paris",
    )
    records = await run_benchmark(_FAKE_PIPELINE, [q])
    assert records[0].score == 0.0


@pytest.mark.asyncio
async def test_browsecomp_source_requires_judge_and_uses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, "some answer")
    q = Question(
        question_id="b1",
        source="browsecomp",
        split="test",
        prompt="obscure question?",
        answer="obscure answer",
    )
    judge = cast(JudgeClient, _FakeJudge(browsecomp_ok=True))
    records = await run_benchmark(_FAKE_PIPELINE, [q], judge=judge)
    assert records[0].score == 1.0
    assert records[0].metric == "browsecomp_judge"


@pytest.mark.asyncio
async def test_browsecomp_source_without_judge_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, "some answer")
    q = Question(
        question_id="b2",
        source="browsecomp",
        split="test",
        prompt="q",
        answer="a",
    )
    records = await run_benchmark(_FAKE_PIPELINE, [q], judge=None)
    assert records[0].metric == "error"
    assert records[0].error is not None and "JudgeClient" in records[0].error


@pytest.mark.asyncio
async def test_research_rubrics_source_uses_weighted_judge_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_pipeline(monkeypatch, "a research report")
    q = Question(
        question_id="r1",
        source="research_rubrics",
        split="test",
        prompt="write a report",
        rubrics=[RubricCriterion(criterion="c1", weight=1.0, axis="Grounding")],
    )
    judge = cast(JudgeClient, _FakeJudge(rubric_score=0.6))
    records = await run_benchmark(_FAKE_PIPELINE, [q], judge=judge)
    assert records[0].score == 0.6
    assert records[0].metric == "rubric_compliance"


@pytest.mark.asyncio
async def test_pipeline_exception_produces_error_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raising_run_pipeline(*args: Any, **kwargs: Any) -> PipelineResult:
        raise RuntimeError("agent exploded")

    monkeypatch.setattr(runner_module, "run_pipeline", raising_run_pipeline)
    q = Question(
        question_id="e1", source="frames", split="test", prompt="q", answer="a"
    )
    records = await run_benchmark(_FAKE_PIPELINE, [q])
    assert records[0].metric == "error"
    assert records[0].error is not None and "agent exploded" in records[0].error
