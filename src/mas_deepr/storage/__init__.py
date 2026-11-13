"""Off-machine archive of local run outputs (Cloudflare R2)."""

from mas_deepr.storage.r2 import (
    R2Config,
    R2ConfigError,
    RunSync,
    SyncReport,
    sync_paths,
    sync_to_r2,
)

__all__ = [
    "R2Config",
    "R2ConfigError",
    "RunSync",
    "SyncReport",
    "sync_paths",
    "sync_to_r2",
]
