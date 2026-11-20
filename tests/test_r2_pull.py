"""R2 download (storage/r2.py::download_paths, scripts/pull_from_r2.py)
with a fake S3 client -- no network, no boto3 needed."""

import hashlib
from pathlib import Path
from typing import Any

from mas_deepr.storage import download_paths


class FakeR2:
    """Dict-backed bucket: list (paginated), head, download."""

    def __init__(self, objects: dict[str, bytes], *, page_size: int = 2) -> None:
        self.data = dict(objects)
        self.meta = {k: hashlib.sha256(v).hexdigest() for k, v in objects.items()}
        self.page_size = page_size
        self.downloads: list[str] = []

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        keys = sorted(k for k in self.data if k.startswith(kwargs["Prefix"]))
        start = int(kwargs.get("ContinuationToken", 0))
        page = keys[start : start + self.page_size]
        more = start + self.page_size < len(keys)
        out: dict[str, Any] = {"Contents": [{"Key": k} for k in page]}
        if more:
            out.update(
                IsTruncated=True, NextContinuationToken=str(start + self.page_size)
            )
        return out

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {
            "ContentLength": len(self.data[Key]),
            "Metadata": {"sha256": self.meta[Key]},
        }

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        self.downloads.append(Key)
        Path(Filename).write_bytes(self.data[Key])


def _bucket() -> FakeR2:
    return FakeR2(
        {
            "mas-deepr/runs/queue/status.jsonl": b'{"stage": 1}\n',
            "mas-deepr/runs/grpo/telemetry.jsonl": b"t\n",
            "mas-deepr/.art/mas-deepr/models/m/checkpoints/0100/adapter.bin": b"w",
            "mas-deepr/wandb/run-1/output.log": b"log\n",
            "other-prefix/runs/x.txt": b"not ours",
        }
    )


def test_downloads_requested_folders_to_same_relative_paths(tmp_path: Path) -> None:
    bucket = _bucket()
    report = download_paths(bucket, "b", ["runs", ".art"], root=tmp_path)
    assert report.ok
    assert len(report.downloaded) == 3
    assert (tmp_path / "runs/queue/status.jsonl").read_bytes() == b'{"stage": 1}\n'
    assert (tmp_path / ".art/mas-deepr/models/m/checkpoints/0100/adapter.bin").exists()
    assert not (tmp_path / "wandb").exists()
    assert not list(tmp_path.rglob("*.r2-partial"))


def test_identical_local_files_are_skipped(tmp_path: Path) -> None:
    bucket = _bucket()
    download_paths(bucket, "b", ["runs"], root=tmp_path)
    again = download_paths(bucket, "b", ["runs"], root=tmp_path)
    assert again.downloaded == []
    assert len(again.skipped) == 2


def test_differing_local_file_is_kept_unless_overwrite(tmp_path: Path) -> None:
    bucket = _bucket()
    local = tmp_path / "runs/grpo/telemetry.jsonl"
    local.parent.mkdir(parents=True)
    local.write_bytes(b"my local edit\n")
    report = download_paths(bucket, "b", ["runs/grpo"], root=tmp_path)
    assert report.conflicts == ["mas-deepr/runs/grpo/telemetry.jsonl"]
    assert local.read_bytes() == b"my local edit\n"
    forced = download_paths(bucket, "b", ["runs/grpo"], root=tmp_path, overwrite=True)
    assert forced.downloaded == ["mas-deepr/runs/grpo/telemetry.jsonl"]
    assert local.read_bytes() == b"t\n"


def test_hash_mismatch_fails_without_writing(tmp_path: Path) -> None:
    bucket = _bucket()
    bucket.meta["mas-deepr/wandb/run-1/output.log"] = "0" * 64
    report = download_paths(bucket, "b", ["wandb"], root=tmp_path)
    assert not report.ok
    assert not (tmp_path / "wandb/run-1/output.log").exists()
    assert not list(tmp_path.rglob("*.r2-partial"))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    bucket = _bucket()
    report = download_paths(bucket, "b", [""], root=tmp_path, dry_run=True)
    assert len(report.downloaded) == 4  # whole archive, other prefixes excluded
    assert bucket.downloads == []
    assert list(tmp_path.iterdir()) == []


def test_keys_escaping_the_project_folder_are_refused(tmp_path: Path) -> None:
    bucket = FakeR2({"mas-deepr/runs/../../evil.txt": b"x"})
    report = download_paths(bucket, "b", ["runs"], root=tmp_path / "proj")
    assert not report.ok
    assert not (tmp_path / "evil.txt").exists()
