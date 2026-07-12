"""ResearchRubrics loader (eval-only). ScaleAI/researchrubrics, processed_data.jsonl."""

import json

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question, RubricCriterion

_REPO_ID = "ScaleAI/researchrubrics"
_FILENAME = "processed_data.jsonl"


def load_research_rubrics(
    settings: Settings, *, limit: int | None = None
) -> list[Question]:
    path = hf_file(settings, repo_id=_REPO_ID, filename=_FILENAME)
    questions = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            questions.append(
                Question(
                    question_id=f"research_rubrics-{d['sample_id']}",
                    source="research_rubrics",
                    split="test",
                    prompt=d["prompt"],
                    rubrics=[RubricCriterion(**r) for r in d.get("rubrics", [])],
                    metadata={
                        "domain": d.get("domain", ""),
                        "conceptual_breadth": d.get("conceptual_breadth", ""),
                        "logical_nesting": d.get("logical_nesting", ""),
                        "exploration": d.get("exploration", ""),
                    },
                )
            )
    return questions
