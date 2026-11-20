"""Power the GPU machine off if the experiment queue dies or stalls.

Started (detached) by ``scripts/run_queue.py --shutdown``, which stops it
again before its own normal power-off. So if this ever acts, something went
wrong -- and it exists so that "something went wrong at 3 a.m." costs an
upload and a stopped VM rather than a GPU billing all weekend:

- the queue process is gone (crashed or killed) for ``--grace-minutes``, or
- nothing under ``runs/queue/logs/`` has changed for ``--idle-minutes``
  (every job streams its output there, so silence means a hang).

Either way it uploads ``runs/``, ``wandb/`` and ``.art/`` to R2, then powers
off (``runpodctl stop pod`` on RunPod, ``sudo shutdown -h now`` elsewhere --
on Google Cloud that stops the VM and keeps its disk).
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "runs" / "queue" / "logs"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def last_activity(log_dir: Path, default: float) -> float:
    """Newest modification time of any file under ``log_dir``."""
    times = [p.stat().st_mtime for p in log_dir.rglob("*") if p.is_file()]
    return max(times, default=default)


def decide(
    *,
    now: float,
    queue_alive: bool,
    dead_since: float | None,
    last_output: float,
    idle_s: float,
    grace_s: float,
) -> str:
    """``"ok"``, ``"queue-died"`` or ``"stalled"``."""
    if not queue_alive and dead_since is not None and now - dead_since >= grace_s:
        return "queue-died"
    if now - last_output >= idle_s:
        return "stalled"
    return "ok"


def upload_and_power_off(python: str, *, dry_run: bool) -> None:
    sync = [python, "scripts/sync_to_r2.py", "runs", "wandb", ".art"]
    pod = os.environ.get("RUNPOD_POD_ID")
    off = (
        ["runpodctl", "stop", "pod", pod] if pod else ["sudo", "shutdown", "-h", "now"]
    )
    if dry_run:
        print(f"watchdog (dry run) would run: {' '.join(sync)} ; {' '.join(off)}")
        return
    try:
        subprocess.run(sync, cwd=PROJECT_ROOT, timeout=45 * 60, check=False)
    finally:
        subprocess.run(off, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-pid", type=int, required=True)
    parser.add_argument("--idle-minutes", type=float, default=45)
    parser.add_argument("--grace-minutes", type=float, default=5)
    parser.add_argument("--check-seconds", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    dead_since: float | None = None
    while True:
        now = time.time()
        alive = pid_alive(args.queue_pid)
        if alive:
            dead_since = None
        elif dead_since is None:
            dead_since = now
        verdict = decide(
            now=now,
            queue_alive=alive,
            dead_since=dead_since,
            last_output=last_activity(LOG_DIR, default=started),
            idle_s=args.idle_minutes * 60,
            grace_s=args.grace_minutes * 60,
        )
        if verdict != "ok":
            print(f"watchdog: {verdict} -- uploading results and powering off")
            upload_and_power_off(sys.executable, dry_run=args.dry_run)
            return 0
        time.sleep(args.check_seconds)


if __name__ == "__main__":
    sys.exit(main())
