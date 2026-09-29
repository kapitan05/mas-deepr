"""ResearchQA loader (eval-only, wiki+paper scope).

realliyifei/ResearchQA -- scholarly QA across 75 fields, survey-mined
questions + rubrics, validated by PhD-level annotators. Using the
``test_mini`` split (776 rows): right-sized for eval, same ballpark as
ResearchRubrics, and keeps the larger ``full``/``test`` splits available
untouched for anyone building a training pool from this source later.

Rubric items have no per-criterion weight in the source data (unlike
ResearchRubrics) -- weighted 1.0 uniformly, matching ResearchRubrics' own
convention of equal weight wherever the source itself provides none. Each
item's ``type`` is a list in the raw data (e.g. ``["Comparison"]``,
confirmed against a real row, not assumed) -- joined into one string for
``RubricCriterion.axis``.
"""

import polars as pl

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question, RubricCriterion

_REPO_ID = "realliyifei/ResearchQA"
_FILENAME = "data/test_mini-00000-of-00001.parquet"


def _row_to_question(row: dict) -> Question:
    rubrics = [
        RubricCriterion(
            criterion=item["rubric_item"],
            weight=1.0,
            axis=", ".join(item["type"]) if item.get("type") else "",
        )
        for item in row["rubric"]
    ]
    return Question(
        question_id=f"research_qa-{row['id']}",
        source="research_qa",
        split="test",
        prompt=row["query"],
        rubrics=rubrics,
        metadata={
            "general_domain": row.get("general_domain", ""),
            "subdomain": row.get("subdomain", ""),
            "field": row.get("field", ""),
        },
    )


def load_research_qa(settings: Settings, *, limit: int | None = None) -> list[Question]:
    path = hf_file(settings, repo_id=_REPO_ID, filename=_FILENAME)
    df = pl.read_parquet(path)
    if limit is not None:
        df = df.head(limit)
    return [_row_to_question(row) for row in df.iter_rows(named=True)]
