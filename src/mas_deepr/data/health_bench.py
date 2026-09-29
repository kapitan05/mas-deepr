"""HealthBench loader (eval-only, wiki+paper scope).

openai/healthbench -- 5,000 realistic health conversations, each graded
against a physician-authored rubric. Using the main
``2025-05-07-06-14-12_oss_eval.jsonl`` file (the full open-model eval set,
not the smaller ``hard``/``consensus`` variants). MIT-licensed, no gate.

Two real adaptations from ResearchRubrics' shape, not a plain rename
(confirmed against real rows, not assumed):

1. ``prompt`` is a *conversation* (``list[{role, content}]``), not a flat
   string -- 41% of real rows are multi-turn (up to 19 turns), so taking
   only the last user turn would silently drop context earlier turns
   established (e.g. a patient's symptoms). Flattened to
   ``"{role}: {content}"`` lines, joined -- lossy in format but not in
   content, and ``evals/graders.py``/the judge only ever see plain text
   anyway.
2. Rubric ``points`` can be **negative** -- physician-marked *undesirable*
   criteria ("advises against seeing a doctor"), not just unweighted-vs-
   weighted like ResearchQA. Kept signed on ``RubricCriterion.weight``,
   not ``abs()``'d -- ``evals/judge.py::grade_research_rubrics``'s scoring
   formula handles signed weights correctly (see its own docstring/fix).
"""

import json

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question, RubricCriterion

_REPO_ID = "openai/healthbench"
_FILENAME = "2025-05-07-06-14-12_oss_eval.jsonl"


def _flatten_prompt(turns: list[dict[str, str]]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def _axis_from_tags(tags: list[str]) -> str:
    for tag in tags:
        if tag.startswith("axis:"):
            return tag.removeprefix("axis:")
    return "health"


def _row_to_question(d: dict) -> Question:
    rubrics = [
        RubricCriterion(
            criterion=r["criterion"],
            weight=float(r["points"]),
            axis=_axis_from_tags(r.get("tags", [])),
        )
        for r in d.get("rubrics", [])
    ]
    return Question(
        question_id=f"health_bench-{d['prompt_id']}",
        source="health_bench",
        split="test",
        prompt=_flatten_prompt(d["prompt"]),
        rubrics=rubrics,
        metadata={"example_tags": d.get("example_tags", [])},
    )


def load_health_bench(
    settings: Settings, *, limit: int | None = None
) -> list[Question]:
    path = hf_file(settings, repo_id=_REPO_ID, filename=_FILENAME)
    questions = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            questions.append(_row_to_question(json.loads(line)))
    return questions
