"""Off-machine archive of local run outputs (Cloudflare R2)."""

from mas_deepr.storage.r2 import (
    DownloadReport,
    R2Config,
    R2ConfigError,
    RunSync,
    SyncReport,
    download_from_r2,
    download_paths,
    sync_paths,
    sync_to_r2,
)

__all__ = [
    "DownloadReport",
    "R2Config",
    "R2ConfigError",
    "RunSync",
    "SyncReport",
    "download_from_r2",
    "download_paths",
    "sync_paths",
    "sync_to_r2",
]
