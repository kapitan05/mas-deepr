"""Run the whole GRPO experiment unattended on the GPU VM.

Four stacked stages, each a NEW LoRA trained from the untrained base model
(``--base-model``, step 0) with the previous stage's flags plus its own:

    s1-baseline  memory folding (+ the token-budget guard, always on)
    s2-shaping   + --penalize-turns --penalize-hallucination
    s3-ngu       + --never-give-up
    s4-rubrics   + --evolving-rubrics

For each stage: train (``train_grpo.py --backend local``) -> evaluate the
final checkpoint (``eval_grpo_checkpoint.py --backend local``, default
benchmarks) -> upload ``runs/``, ``wandb/`` and the stage's ``.art/`` model
folder to R2. Then post-RL DSPy (``compile_dspy.py --grpo-model-name`` on
the s4 checkpoint, ``--auto medium``) -> evaluate s4 with the compiled
prompts -> upload (including the ``*.compiled.yaml`` prompts) -> with
``--shutdown``, power the machine off.

Resumable: every finished job is recorded in ``runs/queue/<state>.json``;
rerunning the same command skips finished jobs, and an interrupted training
stage continues from its last checkpoint (``--num-steps`` = what's left,
``--skip-batches`` = what's done). A failed job is logged and the queue
moves on -- the stages don't depend on each other; only the DSPy jobs need
s4's training. Each job's output goes to ``runs/queue/logs/`` and the
console; ``runs/queue/status.jsonl`` gets one line per job.

The MCP tool server (``run_mcp_server.py``) is started in the background
unless one is already listening. With ``--shutdown``, ``scripts/vm/
watchdog.py`` also runs, detached: it uploads and powers off if the queue
dies or produces no output for ``--idle-minutes``.

Usage (on the VM, inside the project folder, with .venv-gpu):
    .venv-gpu/bin/python scripts/run_queue.py --smoke          # ~15-min check
    .venv-gpu/bin/python scripts/run_queue.py --shutdown       # the real run
    .venv-gpu/bin/python scripts/run_queue.py --dry-run        # print the plan
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_PREFIX = "qwen4b"
MEMORY = "folding"

# (stage name, flags added by this stage) -- each stage also gets every
# earlier stage's flags (stacking), see stage_train_flags().
STAGES: list[tuple[str, list[str]]] = [
    ("s1-baseline", []),
    ("s2-shaping", ["--penalize-turns", "--penalize-hallucination"]),
    ("s3-ngu", ["--never-give-up"]),
    ("s4-rubrics", ["--evolving-rubrics"]),
]
DSPY_STAGE = "s4-rubrics"

# Model flags shared by training, evaluation and DSPy, so a checkpoint is
# served with the settings it was trained with. Empty = bf16 trainer, the
# configuration proven on the L4 with sequences up to 6,144 tokens (see
# rl/local_backend.py). "--load-in-4bit" (QLoRA; vLLM stays bf16) did not
# help at 12,288 -- training memory there is dominated by the sequence
# length, not the weights.
LOCAL_MODEL_FLAGS: list[str] = []

COMMON_TRAIN_FLAGS = [
    "--base-model", BASE_MODEL,
    "--backend", "local",
    *LOCAL_MODEL_FLAGS,
    "--memory", MEMORY,
    "--group-size", "4",
    "--seed", "1",
    "--no-prefer-compiled",
    "--r2-sync",
]  # fmt: skip
SMOKE_EVAL_LIMITS = "research_qa=2,health_bench=2,frames=2"
# ART keeps its vLLM runtime, the base-model weights and compile caches
# under /tmp/art-cache unless told otherwise -- /tmp is wiped on reboot
# (and isn't on RunPod's persistent volume), so every VM start would
# re-download ~20 GB. Keep them in the project folder instead (gitignored;
# scripts/vm/setup_vm.sh fills the same place).
ART_CACHE_DIR = PROJECT_ROOT / ".cache" / "art"


@dataclass
class Job:
    job_id: str
    argv: list[str]
    needs: list[str] = field(default_factory=list)  # job_ids that must be done
    stage: str | None = None  # training jobs: the stage whose LoRA they train
    # Rerun on every invocation: the uploads. sync_to_r2.py only sends what
    # changed, and a done-marked upload from an earlier invocation (e.g. of a
    # training attempt that failed) must not skip uploading the retry.
    always: bool = False


@dataclass
class QueueConfig:
    steps: int = 100
    batch_size: int = 8
    smoke: bool = False
    stages: list[str] = field(default_factory=lambda: [s for s, _ in STAGES])
    dspy: bool = True
    dspy_auto: str = "medium"
    python: str = sys.executable
    run_tag: str = ""  # e.g. "v2": new model names, W&B runs and queue state

    @property
    def tag(self) -> str:
        base = "smoke" if self.smoke else "main"
        return f"{base}-{self.run_tag}" if self.run_tag else base

    def model(self, stage: str) -> str:
        return model_name(stage, smoke=self.smoke, run_tag=self.run_tag)


def model_name(stage: str, *, smoke: bool, run_tag: str = "") -> str:
    """``qwen4b-<stage>[-<run_tag>][-smoke]`` -- the ART model, its
    checkpoint folder and its W&B run all use this name."""
    tag = f"-{run_tag}" if run_tag else ""
    return f"{MODEL_PREFIX}-{stage}{tag}" + ("-smoke" if smoke else "")


def valid_run_tag(tag: str) -> bool:
    return tag == "" or all(c.isalnum() or c in "-_" for c in tag)


def stage_train_flags(stage: str) -> list[str]:
    """Flags for ``stage``: its own plus every earlier stage's (stacking)."""
    names = [s for s, _ in STAGES]
    if stage not in names:
        raise ValueError(f"unknown stage {stage!r}; known: {names}")
    flags: list[str] = []
    for _name, own in STAGES[: names.index(stage) + 1]:
        flags.extend(own)
    return flags


def latest_checkpoint_step(art_models_dir: Path, name: str) -> int:
    """Highest checkpoint step on disk for ART model ``name`` (0 if none) --
    ART keeps ``<models>/<name>/checkpoints/<step:04d>/`` and our pruning
    always keeps the latest."""
    ckpt_dir = art_models_dir / name / "checkpoints"
    if not ckpt_dir.is_dir():
        return 0
    steps = [int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(steps, default=0)


def train_argv(cfg: QueueConfig, stage: str, *, done_steps: int) -> list[str] | None:
    """``train_grpo.py`` command for ``stage``, resuming after ``done_steps``
    already-trained steps; ``None`` if nothing is left to train."""
    remaining = cfg.steps - done_steps
    if remaining <= 0:
        return None
    return [
        cfg.python,
        "scripts/train_grpo.py",
        "--model-name", cfg.model(stage),
        *COMMON_TRAIN_FLAGS,
        "--batch-size", str(cfg.batch_size),
        "--num-steps", str(remaining),
        "--skip-batches", str(done_steps),
        *stage_train_flags(stage),
    ]  # fmt: skip


def eval_argv(cfg: QueueConfig, stage: str, *, compiled: bool = False) -> list[str]:
    name = cfg.model(stage)
    argv = [
        cfg.python,
        "scripts/eval_grpo_checkpoint.py",
        "--model-name", name,
        "--base-model", BASE_MODEL,
        "--backend", "local",
        *LOCAL_MODEL_FLAGS,
        "--memory", MEMORY,
        "--r2-sync",
    ]  # fmt: skip
    if compiled:
        argv += ["--prefer-compiled", "--label", f"{name}-{MEMORY}-dspy"]
    if cfg.smoke:
        argv += ["--benchmark-limit", SMOKE_EVAL_LIMITS]
    return argv


def sync_argv(cfg: QueueConfig, paths: Sequence[str]) -> list[str]:
    return [cfg.python, "scripts/sync_to_r2.py", *paths]


def build_plan(cfg: QueueConfig, *, project: str) -> list[Job]:
    """The ordered job list. Training argv is filled in at run time (it
    depends on the checkpoint already on disk), so it's a placeholder here."""
    jobs: list[Job] = []
    for stage in cfg.stages:
        name = cfg.model(stage)
        jobs.append(Job(f"{stage}:train", [], stage=stage))
        jobs.append(Job(f"{stage}:eval", eval_argv(cfg, stage), [f"{stage}:train"]))
        jobs.append(
            Job(
                f"{stage}:sync",
                sync_argv(cfg, ["runs", "wandb", f".art/{project}/models/{name}"]),
                always=True,
            )
        )
    if cfg.dspy and DSPY_STAGE in cfg.stages:
        name = cfg.model(DSPY_STAGE)
        compile_argv = [
            cfg.python,
            "scripts/compile_dspy.py",
            "--grpo-model-name", name,
            "--base-model", BASE_MODEL,
            "--backend", "local",
            *LOCAL_MODEL_FLAGS,
            "--auto", "light" if cfg.smoke else cfg.dspy_auto,
        ]  # fmt: skip
        if cfg.smoke:
            compile_argv += ["--musique-limit", "10", "--hotpot-limit", "10"]
        jobs.append(Job("dspy:compile", compile_argv, [f"{DSPY_STAGE}:train"]))
        jobs.append(
            Job(
                "dspy:eval",
                eval_argv(cfg, DSPY_STAGE, compiled=True),
                ["dspy:compile"],
            )
        )
        jobs.append(
            Job(
                "dspy:sync",
                sync_argv(cfg, ["runs", "wandb", "src/mas_deepr/prompts/templates"]),
                always=True,
            )
        )
    jobs.append(
        Job("final:sync", sync_argv(cfg, ["runs", "wandb", ".art"]), always=True)
    )
    return jobs


class QueueState:
    """``{job_id: "done" | "failed"}`` persisted as JSON after every job."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.status: dict[str, str] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        )

    def done(self, job_id: str) -> bool:
        return self.status.get(job_id) == "done"

    def set(self, job_id: str, status: str) -> None:
        self.status[job_id] = status
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.status, indent=2), encoding="utf-8")


Runner = Callable[[list[str], Path], int]


def run_logged(argv: list[str], log_path: Path) -> int:
    """Run ``argv`` in the project folder, streaming its output to the
    console and appending it to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now(UTC).isoformat()} $ {' '.join(argv)}\n")
        proc = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()


def run_plan(
    jobs: list[Job],
    cfg: QueueConfig,
    *,
    state: QueueState,
    queue_dir: Path,
    art_models_dir: Path,
    runner: Runner = run_logged,
) -> dict[str, str]:
    """Run ``jobs`` in order, skipping finished ones and jobs whose
    prerequisites didn't finish. Returns this invocation's outcomes."""
    outcomes: dict[str, str] = {}
    status_log = queue_dir / "status.jsonl"
    for job in jobs:
        if state.done(job.job_id) and not job.always:
            outcomes[job.job_id] = "already done"
            continue
        missing = [n for n in job.needs if not state.done(n)]
        if missing:
            outcomes[job.job_id] = f"skipped (needs {', '.join(missing)})"
            continue
        argv = job.argv
        if job.stage is not None:
            done_steps = latest_checkpoint_step(art_models_dir, cfg.model(job.stage))
            maybe = train_argv(cfg, job.stage, done_steps=done_steps)
            if maybe is None:  # trained to the target already
                state.set(job.job_id, "done")
                outcomes[job.job_id] = "done (checkpoints complete)"
                continue
            argv = maybe
        started = time.monotonic()
        print(f"\n### [{job.job_id}] {' '.join(argv)}", flush=True)
        code = runner(argv, queue_dir / "logs" / f"{job.job_id.replace(':', '_')}.log")
        status = "done" if code == 0 else "failed"
        state.set(job.job_id, status)
        outcomes[job.job_id] = status if code == 0 else f"failed (exit {code})"
        with status_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "job": job.job_id,
                        "status": status,
                        "exit_code": code,
                        "minutes": round((time.monotonic() - started) / 60, 1),
                        "finished": datetime.now(UTC).isoformat(),
                    }
                )
                + "\n"
            )
    return outcomes


def port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def start_mcp_server(
    cfg: QueueConfig, host: str, port: int, log_path: Path, timeout_s: float = 180
) -> subprocess.Popen[bytes] | None:
    """Start ``run_mcp_server.py`` in the background unless something is
    already listening on its port; wait until it accepts connections."""
    if port_open(host, port):
        print(f"MCP server already running on {host}:{port}", flush=True)
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    proc = subprocess.Popen(
        [cfg.python, "scripts/run_mcp_server.py"],
        cwd=PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"MCP server exited early -- see {log_path}")
        if port_open(host, port):
            print(f"MCP server up on {host}:{port}", flush=True)
            return proc
        time.sleep(2)
    proc.terminate()
    raise SystemExit(f"MCP server didn't come up within {timeout_s:.0f}s")


def power_off() -> None:
    """Stop this machine: RunPod pods via runpodctl, anything else via
    ``shutdown`` (on Google Cloud that stops the VM; its disk is kept)."""
    pod = os.environ.get("RUNPOD_POD_ID")
    cmd = (
        ["runpodctl", "stop", "pod", pod] if pod else ["sudo", "shutdown", "-h", "now"]
    )
    print(f"Powering off: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="Training steps/stage")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick end-to-end check: 2 steps, batch 2, 2 questions per eval "
        "benchmark, light DSPy on 10+10 examples; separate '-smoke' model "
        "names and state, so it never touches the real runs.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(s for s, _ in STAGES),
        help="Comma-separated subset of stages to run (default: all four).",
    )
    parser.add_argument("--no-dspy", action="store_true", help="Skip the DSPy jobs")
    parser.add_argument(
        "--run-tag",
        default="",
        help="Suffix for a fresh set of runs, e.g. 'v2': every stage trains a "
        "new model (qwen4b-<stage>-v2) with its own W&B run, checkpoints and "
        "queue state; earlier runs are left untouched.",
    )
    parser.add_argument(
        "--dspy-auto", choices=("light", "medium", "heavy"), default="medium"
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Power the machine off at the end, and run the idle watchdog.",
    )
    parser.add_argument(
        "--idle-minutes",
        type=int,
        default=45,
        help="Watchdog: power off after this long without any queue output.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun jobs that failed in an earlier invocation.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and exit"
    )
    args = parser.parse_args(argv)

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        stage_train_flags(s)  # validates the name
    if not valid_run_tag(args.run_tag):
        raise SystemExit("--run-tag may contain only letters, digits, '-' and '_'")
    cfg = QueueConfig(
        steps=2 if args.smoke else args.steps,
        batch_size=2 if args.smoke else args.batch_size,
        smoke=args.smoke,
        stages=stages,
        dspy=not args.no_dspy,
        dspy_auto=args.dspy_auto,
        run_tag=args.run_tag,
    )

    os.environ.setdefault("ART_MEGATRON_CACHE_ROOT", str(ART_CACHE_DIR))
    os.environ.setdefault("HF_HOME", str(ART_CACHE_DIR / "huggingface"))
    # vLLM 0.25 runs dense models (Qwen3-4B) on its new V2 model runner by
    # default; ART 0.5.20-0.5.24's policy-span patch crashes in that runner's
    # start-up profiling run (KeyError on a dummy request id). The V1 runner
    # never goes through the patched path there -- and ART itself forces V1
    # in one of its own code paths.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from mas_deepr.config import get_settings

    settings = get_settings()
    project = settings.wandb_project
    queue_dir = PROJECT_ROOT / "runs" / "queue"
    art_models_dir = PROJECT_ROOT / ".art" / project / "models"
    jobs = build_plan(cfg, project=project)

    if args.dry_run:
        for job in jobs:
            argv_ = job.argv or train_argv(
                cfg,
                job.stage or "",
                done_steps=latest_checkpoint_step(
                    art_models_dir, cfg.model(job.stage or "")
                ),
            )
            print(f"[{job.job_id}] {' '.join(argv_ or ['(nothing left to train)'])}")
        return 0

    state = QueueState(queue_dir / f"state-{cfg.tag}.json")
    if args.retry_failed:
        for job_id, status in list(state.status.items()):
            if status == "failed":
                del state.status[job_id]

    watchdog: subprocess.Popen[bytes] | None = None
    if args.shutdown:
        watchdog = subprocess.Popen(
            [
                cfg.python,
                "scripts/vm/watchdog.py",
                f"--queue-pid={os.getpid()}",
                f"--idle-minutes={args.idle_minutes}",
            ],
            cwd=PROJECT_ROOT,
            start_new_session=True,  # survives the queue, by design
        )
    mcp = start_mcp_server(
        cfg,
        settings.mcp_server_host,
        settings.mcp_server_port,
        queue_dir / "logs" / "mcp_server.log",
    )
    try:
        outcomes = run_plan(
            jobs, cfg, state=state, queue_dir=queue_dir, art_models_dir=art_models_dir
        )
    finally:
        if mcp is not None:
            mcp.terminate()

    print("\n=== queue summary ===")
    for job_id, outcome in outcomes.items():
        print(f"  {job_id:<20} {outcome}")
    failed = any(o.startswith("failed") for o in outcomes.values())
    if args.shutdown:
        if watchdog is not None:
            watchdog.terminate()
        power_off()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
