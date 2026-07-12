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
