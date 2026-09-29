"""LLM-as-judge grading for BrowseComp and any rubric-graded source
(ResearchRubrics, ResearchQA, HealthBench, ...).

The BrowseComp grader prompt is the published OpenAI simple-evals template
(github.com/openai/simple-evals/browsecomp_eval.py) so scored accuracy is
comparable to published baselines. The rubric grader applies each
question's rubric criteria independently and reports a weighted-compliance
score, clamped to [0, 1] -- weights may be negative (HealthBench marks
undesirable behavior this way; a satisfied negative-weight criterion
subtracts from the score instead of adding to it). Per the plan (Phase 3),
calibrate this judge against hand labels before trusting it as a GRPO
reward signal.
"""

import json
import re
from typing import Any

from agent_framework import Agent

from mas_deepr.config import ModelSpec, Settings
from mas_deepr.data.schema import RubricCriterion
from mas_deepr.llm import build_chat_client
from mas_deepr.telemetry import TelemetryTracker, Timer, usage_from_response

# Verbatim from openai/simple-evals browsecomp_eval.py (GRADER_TEMPLATE).
_BROWSECOMP_GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
""".strip()

_RUBRIC_JUDGE_INSTRUCTIONS = """
You are a strict, careful research-report evaluator. You will be given a
research prompt, a response to it, and a numbered list of rubric criteria.
For each criterion, decide whether the response satisfies it.

Respond with ONLY a JSON array, one object per criterion, in this exact
form: [{"index": 0, "satisfied": true}, {"index": 1, "satisfied": false}, ...]
No prose before or after the JSON.
""".strip()

_HALLUCINATION_JUDGE_INSTRUCTIONS = """
You are a strict fact-checker. You will be given a research question, the
evidence an assistant gathered while researching it (its "findings"), and
the assistant's final answer. Decide whether the final answer asserts any
concrete claim (a fact, number, name, date, or quote) that is NOT supported
by the findings -- i.e. the assistant made something up rather than basing
its answer on what it actually found.

Respond with ONLY a JSON object in this exact form:
{"hallucinated": true} or {"hallucinated": false}
No prose before or after the JSON. If the findings are too sparse to judge
either way, answer {"hallucinated": false} -- absence of evidence for a
claim is not the same as evidence the claim is fabricated, and this check
should only catch clear, checkable fabrication.
""".strip()


class JudgeClient:
    """Wraps a frontier judge model behind the same Agent/telemetry pattern
    used by the research pipeline, so judge calls are logged identically."""

    def __init__(
        self, *, spec: ModelSpec, settings: Settings, tracker: TelemetryTracker
    ) -> None:
        client = build_chat_client(spec, settings)
        self._agent = Agent(
            client,
            instructions="You are a precise, literal grading assistant.",
            name="judge",
        )
        self._spec = spec
        self._tracker = tracker

    async def _run(self, prompt: str, *, question_id: str) -> str:
        with Timer() as t:
            resp = await self._agent.run(prompt)
        input_tokens, output_tokens = usage_from_response(resp)
        self._tracker.record(
            role="judge",
            spec=self._spec,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=t.elapsed_s,
            question_id=question_id,
        )
        return resp.text or ""

    async def grade_browsecomp(
        self, *, question: str, correct_answer: str, response: str, question_id: str
    ) -> bool:
        prompt = _BROWSECOMP_GRADER_TEMPLATE.format(
            question=question, correct_answer=correct_answer, response=response
        )
        out = await self._run(prompt, question_id=question_id)
        # Last match, not first: the grader template's verdict line comes
        # after ``reasoning``, which could itself echo the word "correct:".
        matches = re.findall(r"correct:\s*(yes|no)", out.lower())
        return bool(matches and matches[-1] == "yes")

    async def grade_research_rubrics(
        self,
        *,
        prompt: str,
        rubrics: list[RubricCriterion],
        response: str,
        question_id: str,
    ) -> tuple[float, list[dict[str, Any]]]:
        rubric_lines = "\n".join(
            f"{i}. [{r.axis}, weight={r.weight}] {r.criterion}"
            for i, r in enumerate(rubrics)
        )
        judge_prompt = (
            f"{_RUBRIC_JUDGE_INSTRUCTIONS}\n\n"
            f"Research prompt: {prompt}\n\n"
            f"Response:\n{response}\n\n"
            f"Rubric criteria:\n{rubric_lines}"
        )
        out = await self._run(judge_prompt, question_id=question_id)
        verdicts = _parse_rubric_verdicts(out, num_criteria=len(rubrics))
        score = _weighted_rubric_score(rubrics, verdicts)
        return score, [
            {"criterion": r.criterion, "satisfied": v}
            for r, v in zip(rubrics, verdicts, strict=True)
        ]

    async def grade_hallucination(
        self,
        *,
        question: str,
        findings: list[str],
        response: str,
        question_id: str,
    ) -> bool:
        """Cheap, reusable fabrication check for GRPO reward shaping
        (``rl/rubric_reward.py::apply_reward_shaping``) -- same judge client
        already configured for the rollout (no new provider, billed the
        same way as every other judge call). Deliberately conservative:
        defaults to "not hallucinating" both when the judge says so AND
        when its response fails to parse -- a judge hiccup must not
        silently zero out a rollout's reward via the shaping penalty (see
        ``_parse_hallucination_verdict``).
        """
        findings_block = (
            "\n".join(f"- {f}" for f in findings) if findings else "(no findings)"
        )
        judge_prompt = (
            f"{_HALLUCINATION_JUDGE_INSTRUCTIONS}\n\n"
            f"Question: {question}\n\n"
            f"Findings:\n{findings_block}\n\n"
            f"Final answer: {response}"
        )
        out = await self._run(judge_prompt, question_id=question_id)
        return _parse_hallucination_verdict(out)


def _weighted_rubric_score(
    rubrics: list[RubricCriterion], verdicts: list[bool]
) -> float:
    """Weighted-compliance score in [0, 1], signed-weight aware.

    Denominator is the sum of *positive-weight* criteria only, not all
    criteria -- matching OpenAI's own HealthBench reference implementation
    (``calculate_score()`` in openai/simple-evals/healthbench_eval.py:
    ``total_possible_points = sum(points for points > 0)``). Some sources
    (HealthBench) mark undesirable criteria with a *negative* weight
    ("advises against seeing a doctor") -- these only ever subtract when
    satisfied, they don't enlarge the denominator (a response isn't
    penalized in the denominator for merely having temptations to avoid).
    For an all-positive rubric set (ResearchRubrics, ResearchQA) every
    weight is already positive, so this is identical to
    ``sum(r.weight for r in rubrics)`` -- unchanged output for those
    sources.

    Clamped at 0 (OpenAI's own pipeline clips at the aggregate-mean level
    instead of per-example, but mas-deepr's ``EvalRecord.score`` is
    consumed per-question by bootstrap_ci/accuracy-percentage plots/W&B
    metrics that all assume each individual score is already in [0, 1]):
    a response that triggers mostly undesirable criteria scores 0 (as bad
    as it gets), not a confusing negative "accuracy."
    """
    total_weight = sum(r.weight for r in rubrics if r.weight > 0) or 1.0
    achieved_weight = sum(r.weight for r, v in zip(rubrics, verdicts, strict=True) if v)
    return max(0.0, achieved_weight / total_weight)


def _parse_rubric_verdicts(raw: str, *, num_criteria: int) -> list[bool]:
    """Parse the judge's JSON array; default unparseable/missing entries to False."""
    verdicts = [False] * num_criteria
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return verdicts
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return verdicts
    for item in parsed:
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < num_criteria:
            verdicts[idx] = bool(item.get("satisfied", False))
    return verdicts


def _parse_hallucination_verdict(raw: str) -> bool:
    """Parse ``{"hallucinated": bool}``; default to False (not hallucinating)
    on any parse failure -- same "default-safe" shape as
    ``_parse_rubric_verdicts``, but the safe default here is specifically
    "don't penalize" rather than "don't credit": a garbled judge response
    must not silently tank a rollout's reward through the shaping penalty.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return False
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False
    return bool(parsed.get("hallucinated", False))
