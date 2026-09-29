"""DR-Tulu's own RL training data, filtered to paper-search scope.

``rl-research/dr-tulu-rl-data`` (4,881 rows) is DR-Tulu's RLER training set,
not an eval benchmark -- disjoint from ResearchRubrics/ResearchQA/
HealthBench, so importing it as GRPO training data carries zero leakage
risk against mas-deepr's own eval suite (confirmed against HF's
datasets-server API: different repo entirely).

Its ``source`` column encodes which of three underlying corpora each row
came from (confirmed against real rows via datasets-server, exact counts
out of 4,881): ``rl_rag_train_sqa_...`` (ScholarQA, 1,000 rows),
``rl_rag_train_os_...`` (OpenScholar, 944 rows) -- both paper-search-
answerable -- and ``rl_rag_train_sa_...`` (SearchArena, 2,937 rows), which
is general web search, same scope reasoning that excludes BrowseComp from
eval. Only the ``sqa``/``os`` subset (1,944 rows) is loaded here.

Row shape (confirmed via a real sample row): ``ground_truth`` is a
JSON-*encoded string*, not a native object, containing
``{"query": ..., "rubrics": [{"description", "title", "weight"}, ...]}``.
``messages`` duplicates ``query`` as a single user turn -- ``ground_truth``
is used directly instead since it's already parsed structure.

The parquet lives only on HF's auto-conversion ref, not ``main``
(confirmed via ``datasets-server.huggingface.co/parquet``) -- see
``data/hf.py``'s ``revision`` param.
"""

import hashlib
import json

import polars as pl

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question, RubricCriterion

_REPO_ID = "rl-research/dr-tulu-rl-data"
_FILENAME = "default/train/0000.parquet"
_REVISION = "refs/convert/parquet"
# ScholarQA, OpenScholar -- paper-search-answerable. Deliberately excludes
# "_sa_" (SearchArena, general web search) -- see module docstring.
_PAPER_SOURCE_MARKERS = ("_sqa_", "_os_")


def _is_paper_scope(source: str) -> bool:
    return any(marker in source for marker in _PAPER_SOURCE_MARKERS)


def _row_to_question(row: dict) -> Question:
    gt = json.loads(row["ground_truth"])
    rubrics = [
        RubricCriterion(
            criterion=r["description"], weight=r["weight"], axis=r.get("title", "")
        )
        for r in gt["rubrics"]
    ]
    # No stable id ships with this dataset -- hash the source tag + query
    # text into a short, deterministic id (stable across reruns, unique
    # enough for this row count).
    digest = hashlib.sha1((row["source"] + gt["query"]).encode()).hexdigest()[:16]
    return Question(
        question_id=f"dr_tulu_rl-{digest}",
        source="dr_tulu_rl_data",
        split="train",
        prompt=gt["query"],
        rubrics=rubrics,
        metadata={"dr_tulu_source": row["source"]},
    )


def load_dr_tulu_paper_rubrics(
    settings: Settings, *, limit: int | None = None
) -> list[Question]:
    """DR-Tulu's OpenScholar/ScholarQA-sourced rows only, mapped to rubric-
    graded training ``Question``s (see module docstring)."""
    path = hf_file(settings, repo_id=_REPO_ID, filename=_FILENAME, revision=_REVISION)
    df = pl.read_parquet(path)
    df = df.filter(
        pl.col("source").map_elements(_is_paper_scope, return_dtype=pl.Boolean)
    )
    if limit is not None:
        df = df.head(limit)
    return [_row_to_question(row) for row in df.iter_rows(named=True)]
