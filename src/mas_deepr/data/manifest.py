"""Split manifests: commit which question ids are train/val/test so the test
vault stays auditable and milestone evals can assert they never touched it.
"""

import json
from pathlib import Path

from mas_deepr.data.schema import Question


def write_manifest(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"question_id": q.question_id, "source": q.source, "split": q.split}
        for q in questions
    ]
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, str]:
    """Return ``{question_id: split}`` from a committed manifest file."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["question_id"]: row["split"] for row in rows}
