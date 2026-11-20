"""Download run outputs from Cloudflare R2 to this machine (see storage/r2.py).

The counterpart of ``scripts/sync_to_r2.py``: pulls what the GPU VM
uploaded (``runs/``, ``wandb/``, ``.art/``) into this project folder, at
the same relative paths. Files you already have with identical content
are skipped; a local file that differs is kept, never overwritten, unless
you pass ``--overwrite``. Nothing is ever deleted.

Needs the r2 extra (``uv sync --extra r2``) and CLOUDFLARE_ACCOUNT_ID,
R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME in .env.

Usage:
    uv run python scripts/pull_from_r2.py                  # runs/ wandb/ .art/
    uv run python scripts/pull_from_r2.py --dry-run        # list only
    uv run python scripts/pull_from_r2.py runs/queue       # one folder
"""

import argparse
import logging
import sys

from mas_deepr.config import PROJECT_ROOT, get_settings
from mas_deepr.logging_config import configure_logging
from mas_deepr.storage import R2ConfigError, download_from_r2
from mas_deepr.storage.r2 import DEFAULT_PREFIX

DEFAULT_PATHS = ("runs", "wandb", ".art")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Project-relative folders to download "
        f"(default: {' '.join(DEFAULT_PATHS)})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only list what would download"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace local files whose content differs from R2",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)

    configure_logging(logging.getLevelName(args.log_level.upper()))
    try:
        report = download_from_r2(
            list(args.paths) or list(DEFAULT_PATHS),
            settings=get_settings(),
            root=PROJECT_ROOT,
            prefix=args.prefix,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except R2ConfigError as e:
        print(f"R2 pull not possible: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        for key in report.downloaded:
            print(f"would download  {key}")
    for key in report.conflicts:
        print(f"kept local (differs from R2, use --overwrite)  {key}")
    for key, err in report.failed:
        print(f"FAILED          {key}: {err}", file=sys.stderr)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
