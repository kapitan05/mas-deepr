"""Train/val pool: MuSiQue + HotpotQA (multi-hop QA, verifiable answers).

Benchmarks (FRAMES/BrowseComp/ResearchRubrics) are eval-only per the thesis's
leakage-prevention design -- DSPy and GRPO train only against this pool, split
deterministically by a hash of each question's id so the split is stable
across reruns without needing a stored manifest, though ``build_train_pool``
also returns a manifest for the record.
"""

import hashlib
import json

import polars as pl

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file
from mas_deepr.data.schema import Question

_MUSIQUE_REPO_ID = "dgslibisey/MuSiQue"
_HOTPOTQA_REPO_ID = "hotpotqa/hotpot_qa"


def assign_split(question_id: str, *, val_fraction: float) -> str:
    """Deterministic train/val split from a stable hash of ``question_id``."""
    digest = hashlib.sha256(question_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < val_fraction else "train"


def load_musique(
    settings: Settings, *, val_fraction: float = 0.2, limit: int | None = None
) -> list[Question]:
    path = hf_file(
        settings,
        repo_id=_MUSIQUE_REPO_ID,
        filename="musique_ans_v1.0_train.jsonl",
    )
    questions = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            d = json.loads(line)
            qid = f"musique-{d['id']}"
            questions.append(
                Question(
                    question_id=qid,
                    source="musique",
                    split=assign_split(qid, val_fraction=val_fraction),
                    prompt=d["question"],
                    answer=d["answer"],
                    answer_aliases=d.get("answer_aliases", []),
                    metadata={"answerable": d.get("answerable", True)},
                )
            )
    return questions


def load_hotpotqa(
    settings: Settings, *, val_fraction: float = 0.2, limit: int | None = None
) -> list[Question]:
    path = hf_file(
        settings,
        repo_id=_HOTPOTQA_REPO_ID,
        filename="distractor/train-00000-of-00002.parquet",
    )
    df = pl.read_parquet(path)
    if limit is not None:
        df = df.head(limit)

    questions = []
    for row in df.iter_rows(named=True):
        qid = f"hotpotqa-{row['id']}"
        questions.append(
            Question(
                question_id=qid,
                source="hotpotqa",
                split=assign_split(qid, val_fraction=val_fraction),
                prompt=row["question"],
                answer=row["answer"],
                metadata={"type": row.get("type", ""), "level": row.get("level", "")},
            )
        )
    return questions


def build_train_pool(
    settings: Settings,
    *,
    val_fraction: float = 0.2,
    musique_limit: int | None = None,
    hotpot_limit: int | None = None,
) -> list[Question]:
    """Combined MuSiQue + HotpotQA pool, split into train/val by question id."""
    return load_musique(
        settings, val_fraction=val_fraction, limit=musique_limit
    ) + load_hotpotqa(settings, val_fraction=val_fraction, limit=hotpot_limit)
