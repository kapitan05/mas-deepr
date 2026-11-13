"""Upload local run outputs to Cloudflare R2 (see storage/r2.py).

Safe to re-run: files already in R2 with the same size and SHA-256 are
skipped, so this only sends new or changed files. Never deletes anything.

Needs the r2 extra (``uv sync --extra r2``) and CLOUDFLARE_ACCOUNT_ID,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME in .env.

Usage:
    uv run python scripts/sync_to_r2.py                 # runs/ and wandb/
    uv run python scripts/sync_to_r2.py --dry-run       # list what would upload
    uv run python scripts/sync_to_r2.py runs/dev-5ebf503b
"""

import argparse
import logging
import sys
from pathlib import Path

from mas_deepr.config import PROJECT_ROOT, get_settings
from mas_deepr.logging_config import configure_logging
from mas_deepr.storage import R2ConfigError, sync_to_r2
from mas_deepr.storage.r2 import DEFAULT_PREFIX


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files/folders to upload, relative to the project root "
        "(default: runs/ and wandb/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only list what would be uploaded"
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(logging.getLevelName(args.log_level.upper()))
    settings = get_settings()
    paths = [p if p.is_absolute() else PROJECT_ROOT / p for p in args.paths] or [
        settings.runs_dir,
        PROJECT_ROOT / "wandb",
    ]

    try:
        report = sync_to_r2(
            paths,
            settings=settings,
            root=PROJECT_ROOT,
            prefix=args.prefix,
            dry_run=args.dry_run,
        )
    except R2ConfigError as e:
        print(f"R2 sync not possible: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        for key in report.uploaded:
            print(f"would upload  {key}")
    for key, err in report.failed:
        print(f"FAILED        {key}: {err}", file=sys.stderr)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
