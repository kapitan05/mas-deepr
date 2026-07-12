"""LLM-as-judge grading for BrowseComp and ResearchRubrics.

The BrowseComp grader prompt is the published OpenAI simple-evals template
(github.com/openai/simple-evals/browsecomp_eval.py) so scored accuracy is
comparable to published baselines. The ResearchRubrics grader applies each
official per-question rubric criterion independently and reports a
weighted-compliance score; per the plan (Phase 3), calibrate this judge
against hand labels before trusting it as a GRPO reward signal.
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
        match = re.search(r"correct:\s*(yes|no)", out.lower())
        return bool(match and match.group(1) == "yes")

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

        total_weight = sum(r.weight for r in rubrics) or 1.0
        satisfied_weight = sum(
            r.weight for r, v in zip(rubrics, verdicts, strict=True) if v
        )
        return satisfied_weight / total_weight, [
            {"criterion": r.criterion, "satisfied": v}
            for r, v in zip(rubrics, verdicts, strict=True)
        ]


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
