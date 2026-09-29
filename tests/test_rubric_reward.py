"""Rubric-reward scoring and GRPO group-advantage math -- pure logic, no
network/model dependency (unlike test_cold_start.py/test_dpo.py)."""

from typing import Any, cast

import pytest

from mas_deepr.data.schema import RubricCriterion
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.rl.rubric_reward import (
    group_advantages,
    prune_zero_variance_rubrics,
    score_rollout_group,
    score_rollout_group_verifiable,
)


class _FakeJudge:
    def __init__(self, scores: list[float]) -> None:
        self._scores = iter(scores)

    async def grade_research_rubrics(
        self, **kwargs: Any
    ) -> tuple[float, list[dict[str, Any]]]:
        return next(self._scores), []


@pytest.mark.asyncio
async def test_score_rollout_group_grades_each_rollout_independently() -> None:
    judge = cast(JudgeClient, _FakeJudge([0.9, 0.3, 0.6]))
    rubrics = [RubricCriterion(criterion="c1", weight=1.0, axis="Grounding")]
    outcomes = await score_rollout_group(
        judge=judge,
        prompt="p",
        question_id="q1",
        rollouts=["r1", "r2", "r3"],
        rubrics=rubrics,
    )
    assert [o.score for o in outcomes] == [0.9, 0.3, 0.6]
    assert all(o.question_id == "q1" for o in outcomes)


def test_prune_zero_variance_rubrics_drops_all_same_columns() -> None:
    rubrics = [
        RubricCriterion(criterion="always true", weight=1.0, axis="A"),
        RubricCriterion(criterion="mixed", weight=1.0, axis="B"),
        RubricCriterion(criterion="always false", weight=1.0, axis="C"),
    ]
    verdicts = [
        [True, True, False],
        [True, False, False],
        [True, True, False],
    ]
    kept = prune_zero_variance_rubrics(rubrics, verdicts)
    assert [r.criterion for r in kept] == ["mixed"]


def test_group_advantages_zero_mean_unit_variance_shape() -> None:
    advantages = group_advantages([0.2, 0.5, 0.8])
    assert sum(advantages) == pytest.approx(0.0, abs=1e-9)
    assert advantages[0] < advantages[1] < advantages[2]


def test_group_advantages_zero_variance_group_returns_zeros() -> None:
    assert group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0]


def test_group_advantages_single_rollout_returns_zero() -> None:
    assert group_advantages([0.7]) == [0.0]


@pytest.mark.asyncio
async def test_score_rollout_group_verifiable_exact_match_scores_one() -> None:
    outcomes = await score_rollout_group_verifiable(
        prompt="p",
        question_id="q1",
        gold_answer="Paris",
        aliases=[],
        rollouts=["FINAL ANSWER: Paris"],
    )
    assert outcomes[0].score == 1.0


@pytest.mark.asyncio
async def test_score_rollout_group_verifiable_mismatch_scores_zero() -> None:
    outcomes = await score_rollout_group_verifiable(
        prompt="p",
        question_id="q1",
        gold_answer="Paris",
        aliases=[],
        rollouts=["FINAL ANSWER: London"],
    )
    assert outcomes[0].score == 0.0


@pytest.mark.asyncio
async def test_score_rollout_group_verifiable_matches_via_alias() -> None:
    outcomes = await score_rollout_group_verifiable(
        prompt="p",
        question_id="q1",
        gold_answer="Paris",
        aliases=["City of Light"],
        rollouts=["FINAL ANSWER: City of Light"],
    )
    assert outcomes[0].score == 1.0


@pytest.mark.asyncio
async def test_score_rollout_group_verifiable_empty_rollouts() -> None:
    outcomes = await score_rollout_group_verifiable(
        prompt="p", question_id="q1", gold_answer="Paris", aliases=[], rollouts=[]
    )
    assert outcomes == []
