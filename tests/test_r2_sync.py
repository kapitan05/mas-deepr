"""R2 sync (storage/r2.py, scripts/sync_to_r2.py, --r2-sync) with a fake S3
client -- no network, no boto3 needed."""

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from mas_deepr.config import Settings
from mas_deepr.storage import R2Config, R2ConfigError, RunSync, sync_paths
from mas_deepr.storage.r2 import iter_files, object_key


class _NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("404")
        self.response = {"Error": {"Code": "404"}}


class FakeS3:
    """Dict-backed stand-in for the boto3 S3 client's head/upload calls."""

    def __init__(self, *, corrupt: bool = False, fail_on: str | None = None) -> None:
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.uploads: list[str] = []
        self.corrupt = corrupt
        self.fail_on = fail_on

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        return self.objects[(Bucket, Key)]

    def upload_file(
        self, Filename: str, Bucket: str, Key: str, ExtraArgs: dict[str, Any]
    ) -> None:
        if self.fail_on and Key.endswith(self.fail_on):
            raise ConnectionError("network down")
        data = Path(Filename).read_bytes()
        self.uploads.append(Key)
        self.objects[(Bucket, Key)] = {
            "ContentLength": len(data) + (1 if self.corrupt else 0),
            "Metadata": dict(ExtraArgs.get("Metadata", {})),
        }


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "runs" / "dev-abc").mkdir(parents=True)
    (tmp_path / "runs" / "dev-abc" / "telemetry.jsonl").write_text('{"a": 1}\n')
    (tmp_path / "runs" / "dev-abc" / "results.parquet").write_bytes(b"PAR1xx")
    (tmp_path / "wandb" / "run-1").mkdir(parents=True)
    (tmp_path / "wandb" / "run-1" / "output.log").write_text("log\n")
    return tmp_path


def _settings(**overrides: str) -> Settings:
    values = {
        "cloudflare_account_id": "acct123",
        "r2_access_key_id": "AKID",
        "r2_secret_access_key": "SECRET",
        "r2_bucket_name": "bucket",
        **overrides,
    }
    s = Settings()
    for k, v in values.items():
        setattr(s, k, v)
    return s


def test_keys_keep_project_relative_layout(project: Path) -> None:
    path = project / "runs" / "dev-abc" / "telemetry.jsonl"
    assert object_key(path, project) == "mas-deepr/runs/dev-abc/telemetry.jsonl"


def test_uploads_then_skips_unchanged_and_reuploads_changed(project: Path) -> None:
    s3 = FakeS3()
    paths = [project / "runs", project / "wandb"]

    first = sync_paths(s3, "bucket", paths, root=project)
    assert sorted(first.uploaded) == [
        "mas-deepr/runs/dev-abc/results.parquet",
        "mas-deepr/runs/dev-abc/telemetry.jsonl",
        "mas-deepr/wandb/run-1/output.log",
    ]
    meta = s3.objects[("bucket", "mas-deepr/runs/dev-abc/telemetry.jsonl")]
    assert meta["Metadata"]["sha256"] == hashlib.sha256(b'{"a": 1}\n').hexdigest()

    second = sync_paths(s3, "bucket", paths, root=project)
    assert second.uploaded == [] and len(second.skipped) == 3

    with (project / "runs" / "dev-abc" / "telemetry.jsonl").open("a") as f:
        f.write('{"b": 2}\n')  # an appended telemetry file
    third = sync_paths(s3, "bucket", paths, root=project)
    assert third.uploaded == ["mas-deepr/runs/dev-abc/telemetry.jsonl"]


def test_dry_run_uploads_nothing(project: Path) -> None:
    s3 = FakeS3()
    report = sync_paths(s3, "bucket", [project / "runs"], root=project, dry_run=True)
    assert len(report.uploaded) == 2 and report.dry_run
    assert s3.uploads == [] and s3.objects == {}
    assert "would upload 2" in report.summary()


def test_failed_verification_and_errors_are_reported_not_raised(
    project: Path,
) -> None:
    corrupt = sync_paths(FakeS3(corrupt=True), "b", [project / "runs"], root=project)
    assert len(corrupt.failed) == 2 and not corrupt.ok
    assert "verification failed" in corrupt.failed[0][1]

    flaky = sync_paths(
        FakeS3(fail_on="results.parquet"), "b", [project / "runs"], root=project
    )
    assert flaky.uploaded == ["mas-deepr/runs/dev-abc/telemetry.jsonl"]
    assert flaky.failed[0][0] == "mas-deepr/runs/dev-abc/results.parquet"


def test_symlinks_are_not_followed(project: Path) -> None:
    link = project / "wandb" / "latest-run"
    try:
        link.symlink_to(project / "wandb" / "run-1", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks not permitted here")
    files = iter_files([project / "wandb"])
    assert files == [(project / "wandb" / "run-1" / "output.log").resolve()]


def test_missing_credentials_name_the_env_vars() -> None:
    with pytest.raises(R2ConfigError) as exc:
        R2Config.from_settings(_settings(r2_access_key_id="", r2_secret_access_key=" "))
    assert "R2_ACCESS_KEY_ID" in str(exc.value)
    assert "R2_SECRET_ACCESS_KEY" in str(exc.value)
    assert "CLOUDFLARE_ACCOUNT_ID" not in str(exc.value)


def test_config_endpoint_and_secret_not_in_repr() -> None:
    config = R2Config.from_settings(_settings())
    assert config.endpoint_url == "https://acct123.r2.cloudflarestorage.com"
    assert "SECRET" not in repr(config)


def test_run_sync_uploads_outputs_and_only_new_wandb_runs(project: Path) -> None:
    s3 = FakeS3()
    sync = RunSync(enabled=True, settings=_settings(), root=project, client=s3)
    (project / "wandb" / "run-2").mkdir()  # created by "this run"
    (project / "wandb" / "run-2" / "output.log").write_text("new\n")

    report = sync.finish([project / "runs" / "dev-abc"])

    assert report is not None and report.ok
    assert "mas-deepr/wandb/run-2/output.log" in report.uploaded
    assert not any("run-1" in k for k in report.uploaded)  # pre-existing run
    assert "mas-deepr/runs/dev-abc/telemetry.jsonl" in report.uploaded


def test_run_sync_disabled_or_misconfigured_never_raises(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s3 = FakeS3()
    assert (
        RunSync(enabled=False, settings=_settings(), root=project, client=s3).finish(
            [project / "runs"]
        )
        is None
    )
    assert s3.uploads == []

    bad = RunSync(
        enabled=True, settings=_settings(r2_bucket_name=""), root=project, client=s3
    )
    assert bad.finish([project / "runs"]) is None
    assert "R2_BUCKET_NAME" in capsys.readouterr().out


def _load_script() -> Any:
    path = Path(__file__).resolve().parents[1] / "scripts" / "sync_to_r2.py"
    spec = importlib.util.spec_from_file_location("sync_to_r2_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sync_to_r2_script"] = module
    spec.loader.exec_module(module)
    return module


def test_script_exit_code_on_missing_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _load_script()
    monkeypatch.setattr(script, "get_settings", lambda: _settings(r2_bucket_name=""))
    assert script.main(["--dry-run"]) == 2
    assert "R2_BUCKET_NAME" in capsys.readouterr().err


@pytest.mark.parametrize(
    "script_name", ["run_baseline.py", "run_milestone_eval.py", "train_grpo.py"]
)
def test_run_scripts_offer_r2_sync_flag(script_name: str) -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts" / script_name).read_text(
        encoding="utf-8"
    )
    assert '"--r2-sync"' in text
    assert "RunSync(enabled=args.r2_sync" in text
    assert "r2.finish(" in text
