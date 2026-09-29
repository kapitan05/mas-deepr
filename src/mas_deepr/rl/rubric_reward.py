"""DR-Tulu-style evolving rubric reward, built on ``evals/judge.py``.

Scoring mechanism is the exact one ``JudgeClient.grade_research_rubrics``
already uses (weighted rubric-compliance) -- what's new here is running it
across a *group* of rollouts for the same question (a GRPO group) and
deriving the group-normalized advantage GRPO trains on. Self-hosted policy
model as judge keeps judging cost near zero during training (DR Tulu's own
ablation: ~1pt loss vs GPT-4.1) -- pass a ``JudgeClient`` built against the
policy model itself, not necessarily a separate frontier judge.

Deliberately scoped down from DR Tulu's own ~27,000 GPU-hr run and dynamic
rubric-pool *generation* -- this module assumes a fixed rubric pool per
question (e.g. from ResearchRubrics or a hand-authored pool) and focuses on
the scoring/credit-assignment mechanics plan-v2.md's gradient-mechanics
section actually needs, not on rubric synthesis.
"""

import statistics
from dataclasses import dataclass

from mas_deepr.data.schema import RubricCriterion
from mas_deepr.evals.graders import exact_match, extract_final_answer
from mas_deepr.evals.judge import JudgeClient


@dataclass
class RolloutOutcome:
    question_id: str
    prompt: str
    response: str
    score: float  # weighted rubric-compliance in [0, 1]


async def score_rollout_group(
    *,
    judge: JudgeClient,
    prompt: str,
    question_id: str,
    rollouts: list[str],
    rubrics: list[RubricCriterion],
) -> list[RolloutOutcome]:
    """Grade every rollout in one GRPO group against the same rubric pool.

    The rubric pool is held fixed across the group so GRPO's within-group
    advantage compares apples to apples -- one grading call per rollout,
    reusing ``JudgeClient.grade_research_rubrics`` rather than a bespoke
    grader.
    """
    outcomes = []
    for response in rollouts:
        score, _verdicts = await judge.grade_research_rubrics(
            prompt=prompt, rubrics=rubrics, response=response, question_id=question_id
        )
        outcomes.append(
            RolloutOutcome(
                question_id=question_id, prompt=prompt, response=response, score=score
            )
        )
    return outcomes


async def score_rollout_group_verifiable(
    *,
    prompt: str,
    question_id: str,
    gold_answer: str,
    aliases: list[str],
    rollouts: list[str],
) -> list[RolloutOutcome]:
    """Judge-free reward for verifiable-answer questions (MuSiQue/HotpotQA):
    exact_match against each rollout's extracted final answer, 1.0/0.0.

    No judge call, no cost, no judge-model dependency -- for the wiki-
    answerable half of a GRPO training mix that the rubric-graded path
    above was never built to score (those questions carry ``.answer``, not
    ``.rubrics`` -- see ``rl/rollout.py``'s dispatch on ``Question``).
    Reuses the exact grader ``evals/runner.py`` already uses for FRAMES,
    not a bespoke one.
    """
    outcomes = []
    for response in rollouts:
        predicted = extract_final_answer(response)
        score = 1.0 if exact_match(predicted, gold_answer, aliases) else 0.0
        outcomes.append(
            RolloutOutcome(
                question_id=question_id, prompt=prompt, response=response, score=score
            )
        )
    return outcomes


def prune_zero_variance_rubrics(
    rubrics: list[RubricCriterion], verdicts_per_rollout: list[list[bool]]
) -> list[RubricCriterion]:
    """Drop criteria every rollout satisfies or every rollout fails.

    DR Tulu's own pruning rule: a rubric with zero variance across a GRPO
    group carries no learning signal for that group (it can't distinguish
    any rollout from any other), so keeping it only adds judge-call cost.
    """
    keep = []
    for i, rubric in enumerate(rubrics):
        column = [v[i] for v in verdicts_per_rollout if i < len(v)]
        if column and len(set(column)) > 1:
            keep.append(rubric)
    return keep


def group_advantages(scores: list[float]) -> list[float]:
    """GRPO's group-normalized advantage: ``A_i = (r_i - mean) / std``.

    No critic network needed -- this is the whole reason GRPO fits a 1xH100
    budget (one fewer resident model than PPO). Zero-variance groups (every
    rollout scored identically, or fewer than 2 rollouts) return all-zero
    advantages instead of dividing by zero: there's no learning signal in a
    group where nothing distinguishes any rollout.
    """
    if len(scores) < 2:
        return [0.0 for _ in scores]
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores)
    if stdev == 0:
        return [0.0 for _ in scores]
    return [(s - mean) / stdev for s in scores]
