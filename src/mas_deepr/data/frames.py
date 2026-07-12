"""FRAMES loader (eval-only, never trained on). google/frames-benchmark, test.tsv."""

import polars as pl

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question

_REPO_ID = "google/frames-benchmark"
_FILENAME = "test.tsv"


def load_frames(settings: Settings, *, limit: int | None = None) -> list[Question]:
    path = hf_file(settings, repo_id=_REPO_ID, filename=_FILENAME)
    df = pl.read_csv(path, separator="\t")
    if limit is not None:
        df = df.head(limit)

    questions = []
    for i, row in enumerate(df.iter_rows(named=True)):
        wiki_links = [
            v
            for k, v in row.items()
            if k.startswith("wikipedia_link") and v not in (None, "")
        ]
        questions.append(
            Question(
                question_id=f"frames-{i}",
                source="frames",
                split="test",
                prompt=row["Prompt"],
                answer=row["Answer"],
                metadata={
                    "reasoning_types": row.get("reasoning_types") or "",
                    "wiki_links": wiki_links,
                },
            )
        )
    return questions
