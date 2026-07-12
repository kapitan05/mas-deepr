from pathlib import Path

from mas_deepr.data.manifest import load_manifest, write_manifest
from mas_deepr.data.schema import Question


def test_write_load_manifest_roundtrip(tmp_path: Path) -> None:
    questions = [
        Question(
            question_id="a-1", source="frames", split="test", prompt="p1", answer="x"
        ),
        Question(
            question_id="b-2", source="musique", split="train", prompt="p2", answer="y"
        ),
    ]
    path = tmp_path / "manifest.json"
    write_manifest(questions, path)

    manifest = load_manifest(path)
    assert manifest == {"a-1": "test", "b-2": "train"}
