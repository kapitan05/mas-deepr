"""scripts/run_queue.py (the unattended VM experiment queue) and
scripts/vm/watchdog.py -- plan, resume and failure logic with a fake runner;
no subprocesses, GPU or network."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rq = _load("scripts/run_queue.py", "run_queue_t")
wd = _load("scripts/vm/watchdog.py", "watchdog_t")


def test_stages_stack_flags_but_each_trains_its_own_model() -> None:
    assert rq.stage_train_flags("s1-baseline") == []
    assert rq.stage_train_flags("s3-ngu") == [
        "--penalize-turns",
        "--penalize-hallucination",
        "--never-give-up",
    ]
    assert rq.stage_train_flags("s4-rubrics")[-1] == "--evolving-rubrics"
    names = {rq.model_name(s, smoke=False) for s, _ in rq.STAGES}
    assert len(names) == 4  # a new LoRA per stage, never a shared one
    with pytest.raises(ValueError):
        rq.stage_train_flags("s9-nope")


def test_train_argv_fresh_resumed_and_complete() -> None:
    cfg = rq.QueueConfig(python="py")
    fresh = rq.train_argv(cfg, "s2-shaping", done_steps=0)
    assert fresh[:2] == ["py", "scripts/train_grpo.py"]
    i = fresh.index("--num-steps")
    assert fresh[i + 1] == "100"
    assert fresh[fresh.index("--skip-batches") + 1] == "0"
    assert "--penalize-hallucination" in fresh and "--never-give-up" not in fresh
    assert fresh[fresh.index("--memory") + 1] == "folding"
    assert "--no-prefer-compiled" in fresh

    resumed = rq.train_argv(cfg, "s2-shaping", done_steps=37)
    assert resumed[resumed.index("--num-steps") + 1] == "63"
    assert resumed[resumed.index("--skip-batches") + 1] == "37"
    assert rq.train_argv(cfg, "s2-shaping", done_steps=100) is None


def test_latest_checkpoint_step(tmp_path: Path) -> None:
    assert rq.latest_checkpoint_step(tmp_path, "m") == 0
    for step in ("0000", "0025", "0037"):
        (tmp_path / "m" / "checkpoints" / step).mkdir(parents=True)
    (tmp_path / "m" / "checkpoints" / "notes.txt").write_text("x")
    assert rq.latest_checkpoint_step(tmp_path, "m") == 37


def test_plan_order_dependencies_and_dspy() -> None:
    cfg = rq.QueueConfig(python="py")
    jobs = rq.build_plan(cfg, project="mas-deepr")
    ids = [j.job_id for j in jobs]
    assert ids[:3] == ["s1-baseline:train", "s1-baseline:eval", "s1-baseline:sync"]
    assert ids[-4:] == ["dspy:compile", "dspy:eval", "dspy:sync", "final:sync"]
    by_id = {j.job_id: j for j in jobs}
    assert by_id["s1-baseline:eval"].needs == ["s1-baseline:train"]
    assert by_id["dspy:compile"].needs == ["s4-rubrics:train"]
    compile_argv = by_id["dspy:compile"].argv
    assert compile_argv[compile_argv.index("--auto") + 1] == "medium"
    assert "--prefer-compiled" in by_id["dspy:eval"].argv
    assert ".art/mas-deepr/models/qwen4b-s1-baseline" in by_id["s1-baseline:sync"].argv
    assert by_id["final:sync"].always


def test_smoke_plan_is_separate_and_tiny() -> None:
    cfg = rq.QueueConfig(python="py", smoke=True, steps=2, batch_size=2)
    jobs = {j.job_id: j for j in rq.build_plan(cfg, project="p")}
    assert "qwen4b-s1-baseline-smoke" in jobs["s1-baseline:eval"].argv
    assert rq.SMOKE_EVAL_LIMITS in jobs["s1-baseline:eval"].argv
    assert "light" in jobs["dspy:compile"].argv
    assert cfg.tag == "smoke"


def test_no_dspy_without_s4() -> None:
    cfg = rq.QueueConfig(python="py", stages=["s1-baseline"])
    ids = [j.job_id for j in rq.build_plan(cfg, project="p")]
    assert not any(i.startswith("dspy") for i in ids)


def _run(tmp_path: Path, cfg: Any, codes: dict[str, int]) -> tuple[dict, list]:
    calls: list[list[str]] = []

    def fake_runner(argv: list[str], log_path: Path) -> int:
        calls.append(argv)
        job = log_path.stem.replace("_", ":", 1)
        return codes.get(job, 0)

    state = rq.QueueState(tmp_path / "state.json")
    outcomes = rq.run_plan(
        rq.build_plan(cfg, project="p"),
        cfg,
        state=state,
        queue_dir=tmp_path,
        art_models_dir=tmp_path / "models",
        runner=fake_runner,
    )
    return outcomes, calls


def test_failed_training_skips_its_eval_and_dspy_but_not_other_stages(
    tmp_path: Path,
) -> None:
    cfg = rq.QueueConfig(python="py")
    outcomes, _ = _run(tmp_path, cfg, {"s4-rubrics:train": 1})
    assert outcomes["s4-rubrics:train"] == "failed (exit 1)"
    assert outcomes["s4-rubrics:eval"].startswith("skipped")
    assert outcomes["dspy:compile"].startswith("skipped")
    assert outcomes["s1-baseline:eval"] == "done"
    lines = (tmp_path / "status.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["job"] == "s1-baseline:train"


def test_rerun_skips_done_jobs_but_always_reuploads(tmp_path: Path) -> None:
    cfg = rq.QueueConfig(python="py", stages=["s1-baseline"])
    _run(tmp_path, cfg, {})
    outcomes, calls = _run(tmp_path, cfg, {})
    assert outcomes["s1-baseline:train"] == "already done"
    assert outcomes["final:sync"] == "done"
    assert len(calls) == 1 and "sync_to_r2.py" in calls[0][1]


def test_interrupted_training_resumes_from_last_checkpoint(tmp_path: Path) -> None:
    cfg = rq.QueueConfig(python="py", stages=["s1-baseline"])
    (tmp_path / "models" / "qwen4b-s1-baseline" / "checkpoints" / "0040").mkdir(
        parents=True
    )
    _, calls = _run(tmp_path, cfg, {})
    train = calls[0]
    assert train[train.index("--num-steps") + 1] == "60"
    assert train[train.index("--skip-batches") + 1] == "40"


def test_fully_trained_stage_is_marked_done_without_running(tmp_path: Path) -> None:
    cfg = rq.QueueConfig(python="py", stages=["s1-baseline"])
    (tmp_path / "models" / "qwen4b-s1-baseline" / "checkpoints" / "0100").mkdir(
        parents=True
    )
    outcomes, calls = _run(tmp_path, cfg, {})
    assert outcomes["s1-baseline:train"] == "done (checkpoints complete)"
    assert not any("train_grpo.py" in " ".join(c) for c in calls)


def test_watchdog_decisions() -> None:
    kw: dict[str, Any] = {"idle_s": 2700, "grace_s": 300}
    assert (
        wd.decide(now=1000, queue_alive=True, dead_since=None, last_output=900, **kw)
        == "ok"
    )
    assert (
        wd.decide(now=5000, queue_alive=True, dead_since=None, last_output=1000, **kw)
        == "stalled"
    )
    assert (
        wd.decide(now=1200, queue_alive=False, dead_since=1000, last_output=1150, **kw)
        == "ok"
    )
    assert (
        wd.decide(now=1400, queue_alive=False, dead_since=1000, last_output=1350, **kw)
        == "queue-died"
    )


def test_eval_and_dspy_commands_parse_with_the_real_scripts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every eval/DSPy command the queue generates must get through the real
    script's argument parser (stopped right after parsing)."""
    import asyncio

    import mas_deepr.config as config_mod

    class _Parsed(Exception):
        pass

    def stop(*args: Any, **kwargs: Any) -> Any:
        raise _Parsed

    for smoke in (False, True):
        cfg = rq.QueueConfig(python="py", smoke=smoke)
        for job in rq.build_plan(cfg, project="p"):
            if not job.argv:  # training: argv is built at run time
                continue
            if "eval_grpo_checkpoint.py" in job.argv[1]:
                module = _load("scripts/eval_grpo_checkpoint.py", "eval_q_t")
                monkeypatch.setattr(config_mod, "get_settings", stop)
                monkeypatch.setattr(sys, "argv", job.argv[1:])
                with pytest.raises(_Parsed):
                    asyncio.run(module.main())
            elif "compile_dspy.py" in job.argv[1]:
                module = _load("scripts/compile_dspy.py", "compile_q_t")
                monkeypatch.setattr(module, "configure_logging", stop)
                monkeypatch.setattr(sys, "argv", job.argv[1:])
                with pytest.raises(_Parsed):
                    module.main()
