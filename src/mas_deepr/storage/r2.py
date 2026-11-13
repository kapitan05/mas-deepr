"""Upload local run outputs (``runs/``, ``wandb/``) to Cloudflare R2.

R2 speaks the S3 API, so this uses boto3 (the ``r2`` extra:
``uv sync --extra r2``) pointed at ``https://<account_id>.r2.cloudflarestorage.com``.
boto3 is imported lazily inside ``make_client`` -- nothing else here needs
it, so the core install, CI and the tests (which use a fake client) don't.

Layout: every file keeps its path relative to the project root, under one
prefix -- ``runs/dev-5ebf503b/telemetry.jsonl`` ->
``s3://<bucket>/mas-deepr/runs/dev-5ebf503b/telemetry.jsonl``.

Idempotent: each object carries its file's SHA-256 in its metadata, and a
file whose remote object has the same size and hash is skipped, so
re-running a sync only uploads new or changed files (an appended
``telemetry.jsonl`` is re-uploaded, an untouched one isn't). Every upload
is verified afterwards (remote size + hash must match). Nothing is ever
deleted, locally or remotely.
"""

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mas_deepr.config import Settings

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "mas-deepr"
_HASH_META_KEY = "sha256"
_NOT_FOUND_CODES = {"404", "NoSuchKey", "NotFound"}


class R2ConfigError(RuntimeError):
    """R2 credentials missing -- the message names the env vars to set."""


class S3Client(Protocol):
    """The three boto3 S3 client methods this module uses."""

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]: ...

    def upload_file(
        self, Filename: str, Bucket: str, Key: str, ExtraArgs: dict[str, Any]
    ) -> None: ...


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str = field(repr=False)
    bucket: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "R2Config":
        values = {
            "CLOUDFLARE_ACCOUNT_ID": settings.cloudflare_account_id,
            "R2_ACCESS_KEY_ID": settings.r2_access_key_id,
            "R2_SECRET_ACCESS_KEY": settings.r2_secret_access_key,
            "R2_BUCKET_NAME": settings.r2_bucket_name,
        }
        missing = [name for name, value in values.items() if not value.strip()]
        if missing:
            raise R2ConfigError(
                "R2 sync needs these set in .env: " + ", ".join(missing)
            )
        return cls(
            account_id=settings.cloudflare_account_id.strip(),
            access_key_id=settings.r2_access_key_id.strip(),
            secret_access_key=settings.r2_secret_access_key.strip(),
            bucket=settings.r2_bucket_name.strip(),
        )


def make_client(config: R2Config) -> S3Client:
    """A boto3 S3 client for R2. Needs the ``r2`` extra."""
    try:
        import boto3
    except ImportError as e:
        raise R2ConfigError("boto3 is not installed -- run `uv sync --extra r2`") from e
    client: S3Client = boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key_id,
        aws_secret_access_key=config.secret_access_key,
        region_name="auto",
    )
    return client


@dataclass
class SyncReport:
    uploaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    dry_run: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed

    def summary(self) -> str:
        verb = "would upload" if self.dry_run else "uploaded"
        return (
            f"R2 sync: {verb} {len(self.uploaded)}, unchanged {len(self.skipped)}, "
            f"failed {len(self.failed)}"
        )


def iter_files(paths: Iterable[Path]) -> list[Path]:
    """Regular files under ``paths`` (each a file or a directory), sorted,
    de-duplicated; symlinks are skipped (W&B's ``latest-run`` link would
    otherwise upload a whole run twice)."""
    found: set[Path] = set()
    for path in paths:
        if path.is_symlink() or not path.exists():
            continue
        if path.is_file():
            found.add(path.resolve())
            continue
        for p in path.rglob("*"):
            if not p.is_file() or p.is_symlink():
                continue
            inside = p.relative_to(path).parents  # folders between path and p
            if any((path / d).is_symlink() for d in inside if d != Path(".")):
                continue
            found.add(p.resolve())
    return sorted(found)


def object_key(path: Path, root: Path, prefix: str = DEFAULT_PREFIX) -> str:
    """``mas-deepr/<path relative to root, POSIX separators>``."""
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return f"{prefix.strip('/')}/{rel}" if prefix else rel


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote(client: S3Client, bucket: str, key: str) -> dict[str, Any] | None:
    """``head_object`` result, or ``None`` if the object doesn't exist."""
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as e:
        code = str(getattr(e, "response", {}).get("Error", {}).get("Code", ""))
        if code in _NOT_FOUND_CODES:
            return None
        raise


def _matches(remote: dict[str, Any] | None, size: int, digest: str) -> bool:
    if remote is None:
        return False
    meta = remote.get("Metadata") or {}
    return (
        int(remote.get("ContentLength", -1)) == size
        and meta.get(_HASH_META_KEY) == digest
    )


def sync_paths(
    client: S3Client,
    bucket: str,
    paths: Sequence[Path],
    *,
    root: Path,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
) -> SyncReport:
    """Upload new/changed files under ``paths``; see module docstring."""
    report = SyncReport(dry_run=dry_run)
    for path in iter_files(paths):
        key = str(path)
        try:
            key = object_key(path, root, prefix)
            size = path.stat().st_size
            digest = _sha256(path)
            if _matches(_remote(client, bucket, key), size, digest):
                report.skipped.append(key)
                continue
            if dry_run:
                report.uploaded.append(key)
                continue
            client.upload_file(
                str(path), bucket, key, ExtraArgs={"Metadata": {_HASH_META_KEY: digest}}
            )
            if not _matches(_remote(client, bucket, key), size, digest):
                raise RuntimeError("verification failed: remote size/hash mismatch")
            report.uploaded.append(key)
        except Exception as e:
            logger.warning("R2 upload failed for %s: %s", key, e)
            report.failed.append((key, f"{type(e).__name__}: {e}"))
    return report


def sync_to_r2(
    paths: Sequence[Path],
    *,
    settings: "Settings",
    root: Path,
    prefix: str = DEFAULT_PREFIX,
    dry_run: bool = False,
    client: S3Client | None = None,
) -> SyncReport:
    """Config from ``settings`` + a real client (unless one is given)."""
    config = R2Config.from_settings(settings)
    return sync_paths(
        client or make_client(config),
        config.bucket,
        paths,
        root=root,
        prefix=prefix,
        dry_run=dry_run,
    )


def _wandb_run_dirs(wandb_root: Path) -> set[Path]:
    if not wandb_root.is_dir():
        return set()
    return {
        p.resolve()
        for p in wandb_root.iterdir()
        if p.is_dir()
        and not p.is_symlink()
        and p.name.startswith(("run-", "offline-run-"))
    }


class RunSync:
    """``--r2-sync`` for the run scripts: create at start (snapshots which
    ``wandb/run-*`` folders already exist), call ``finish(paths)`` at the
    end to upload ``paths`` plus every W&B run folder this run created.

    Never raises: a finished run must not be marked failed because of
    storage -- problems are logged as warnings and printed.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        settings: "Settings",
        root: Path,
        wandb_root: Path | None = None,
        client: S3Client | None = None,
    ) -> None:
        self.enabled = enabled
        self.settings = settings
        self.root = root
        self.wandb_root = wandb_root if wandb_root is not None else root / "wandb"
        self.client = client
        self._wandb_before = _wandb_run_dirs(self.wandb_root) if enabled else set()

    def new_wandb_dirs(self) -> list[Path]:
        return sorted(_wandb_run_dirs(self.wandb_root) - self._wandb_before)

    def finish(self, paths: Sequence[Path]) -> SyncReport | None:
        if not self.enabled:
            return None
        targets = [*paths, *self.new_wandb_dirs()]
        try:
            report = sync_to_r2(
                targets, settings=self.settings, root=self.root, client=self.client
            )
        except Exception as e:
            logger.warning("R2 sync skipped: %s", e)
            print(f"\nR2 sync skipped (run results are saved locally): {e}")
            return None
        print(f"\n{report.summary()}")
        for key, err in report.failed:
            print(f"  failed: {key}: {err}")
        return report
