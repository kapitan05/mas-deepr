"""TrajectoryLogger: incremental JSONL trajectory log for eval runs.

Mirrors test_telemetry.py's pattern. The property that actually matters
here (beyond "the writer works") is crash resilience: each ``.record()``
call is a separate append, so a run that stops partway still has every
question logged up to that point.
"""

from pathlib import Path

from mas_deepr.evals.runner import EvalRecord
from mas_deepr.telemetry import TrajectoryLogger, load_trajectories
from mas_deepr.telemetry.trajectory_log import TrajectoryRecord


def _eval_record(qid: str) -> EvalRecord:
    return EvalRecord(
        question_id=qid,
        source="frames",
        metric="exact_match",
        score=1.0,
        final_answer="Paris",
        sub_questions=["Where is Paris?"],
        findings=["Paris is the capital of France."],
        question="What is the capital of France?",
        gold_answer="Paris",
        latency_s=1.5,
        cost_usd=0.002,
    )


def test_record_writes_one_jsonl_line_with_full_trajectory(tmp_path: Path) -> None:
    sink = tmp_path / "trajectories.jsonl"
    logger = TrajectoryLogger(sink, run_id="r1", phase="dev", model_key="qwen3-8b")

    rec = logger.record(_eval_record("q1"))

    assert isinstance(rec, TrajectoryRecord)
    assert sink.read_text().count("\n") == 1
    df = load_trajectories(tmp_path, use_cache=False)
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["question"] == "What is the capital of France?"
    assert row["gold_answer"] == "Paris"
    assert row["sub_questions"] == ["Where is Paris?"]
    assert row["findings"] == ["Paris is the capital of France."]
    assert row["final_answer"] == "Paris"
    assert row["run_id"] == "r1" and row["model_key"] == "qwen3-8b"


def test_partial_run_keeps_completed_questions_on_disk(tmp_path: Path) -> None:
    """The crash-resilience property: log 2 of 3 questions, "crash" (just
    stop calling record), confirm the 2 completed ones survive on disk."""
    sink = tmp_path / "trajectories.jsonl"
    logger = TrajectoryLogger(sink, run_id="r1", phase="dev", model_key="qwen3-8b")

    logger.record(_eval_record("q1"))
    logger.record(_eval_record("q2"))
    # simulated crash -- q3 never gets logged

    df = load_trajectories(tmp_path, use_cache=False)
    assert sorted(df["question_id"].to_list()) == ["q1", "q2"]


def test_load_trajectories_filters_by_phase_and_model(tmp_path: Path) -> None:
    a = TrajectoryLogger(
        tmp_path / "a" / "trajectories.jsonl",
        run_id="r1",
        phase="baseline",
        model_key="m1",
    )
    b = TrajectoryLogger(
        tmp_path / "b" / "trajectories.jsonl",
        run_id="r2",
        phase="post-dspy",
        model_key="m2",
    )
    a.record(_eval_record("q1"))
    b.record(_eval_record("q2"))

    assert load_trajectories(tmp_path, phase="baseline").height == 1
    assert load_trajectories(tmp_path, model_key="m2").height == 1
    assert load_trajectories(tmp_path).height == 2
