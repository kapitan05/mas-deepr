import json
from pathlib import Path

from mas_deepr.config.models import ModelSpec
from mas_deepr.telemetry import TelemetryTracker, read_telemetry, summarize

_SPEC = ModelSpec(
    key="test-model",
    model_id="test/model",
    family="qwen3",
    input_price_per_mtok=1.0,
    output_price_per_mtok=2.0,
)


def test_record_writes_row_with_correct_cost(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.jsonl"
    tracker = TelemetryTracker(sink, run_id="r1", phase="dev")

    rec = tracker.record(
        role="manager", spec=_SPEC, input_tokens=1000, output_tokens=500, latency_s=0.5
    )

    assert rec.cost_usd == (1000 * 1.0 + 500 * 2.0) / 1_000_000
    df = read_telemetry(sink)
    assert df.height == 1
    assert df["role"][0] == "manager"


def test_summarize_aggregates_across_calls(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.jsonl"
    tracker = TelemetryTracker(sink, run_id="r1", phase="dev")
    tracker.record(
        role="browser", spec=_SPEC, input_tokens=100, output_tokens=100, latency_s=1.0
    )
    tracker.record(
        role="browser", spec=_SPEC, input_tokens=200, output_tokens=200, latency_s=3.0
    )

    summary = summarize(sink)
    row = summary.filter(summary["role"] == "browser").row(0, named=True)
    assert row["calls"] == 2
    assert row["input_tokens"] == 300
    assert row["output_tokens"] == 300
    assert row["mean_latency_s"] == 2.0


def _write_run(sink: Path, run_id: str, phase: str, model_key: str, n: int) -> None:
    sink.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        lines.append(
            json.dumps(
                {
                    "record_id": f"{run_id}-{i}",
                    "ts": "2026-01-01T00:00:00+00:00",
                    "run_id": run_id,
                    "phase": phase,
                    "role": "manager",
                    "model_key": model_key,
                    "question_id": f"q{i}",
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "latency_s": 0.1,
                    "cost_usd": 0.0001,
                    "error": None,
                    "memory_strategy": None,
                    "pass_index": None,
                }
            )
        )
    sink.write_text("\n".join(lines) + "\n")


def test_load_runs_concatenates_and_filters(tmp_path: Path) -> None:
    from mas_deepr.telemetry import load_runs

    _write_run(tmp_path / "a" / "telemetry.jsonl", "r1", "baseline", "qwen3-8b", 3)
    _write_run(tmp_path / "b" / "telemetry.jsonl", "r2", "post-dspy", "qwen3-8b", 2)

    all_df = load_runs(tmp_path)
    assert all_df.height == 5

    baseline = load_runs(tmp_path, phase="baseline")
    assert baseline.height == 3
    assert baseline["run_id"].unique().to_list() == ["r1"]


def test_load_runs_cache_round_trip_and_invalidation(tmp_path: Path) -> None:
    import time

    from mas_deepr.telemetry import load_runs

    sink = tmp_path / "a" / "telemetry.jsonl"
    _write_run(sink, "r1", "baseline", "qwen3-8b", 2)

    load_runs(tmp_path)
    assert (tmp_path / "_telemetry_cache.parquet").exists()

    # Second load hits the cache (same fingerprint).
    assert load_runs(tmp_path).height == 2

    # Appending invalidates the cache -> new rows show up.
    time.sleep(0.01)
    with sink.open("a") as f:
        f.write(sink.read_text().splitlines()[0] + "\n")
    assert load_runs(tmp_path).height == 3
