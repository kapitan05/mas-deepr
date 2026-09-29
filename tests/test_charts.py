"""Comparison charts: each fn returns a populated Figure off a synthetic DF.

Headless (Agg backend), no network, no real parquet files.
"""

import matplotlib

matplotlib.use("Agg")

import polars as pl
from matplotlib.figure import Figure

from mas_deepr.evals.charts import (
    accuracy_comparison_chart,
    comparison_table,
    cost_latency_chart,
)


def _synthetic_df() -> pl.DataFrame:
    rows = []
    for model, base in (("qwen3-8b", 0.4), ("gpt-4.1", 0.8)):
        for source in ("frames", "browsecomp"):
            for i in range(6):
                rows.append(
                    {
                        "question_id": f"{source}-{i}",
                        "source": source,
                        "metric": "exact_match",
                        "score": base if i % 2 == 0 else base - 0.2,
                        "error": None,
                        "latency_s": 10.0 if model == "qwen3-8b" else 25.0,
                        "cost_usd": 0.001 if model == "qwen3-8b" else 0.05,
                        "model": model,
                    }
                )
    # one infra-error row that must be excluded from accuracy
    rows.append(
        {
            "question_id": "frames-err",
            "source": "frames",
            "metric": "error",
            "score": 0.0,
            "error": "RuntimeError: boom",
            "latency_s": 0.0,
            "cost_usd": 0.0,
            "model": "qwen3-8b",
        }
    )
    return pl.DataFrame(rows)


def test_accuracy_comparison_chart_returns_figure() -> None:
    fig = accuracy_comparison_chart(_synthetic_df())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert len(ax.patches) > 0  # bars drawn


def test_cost_latency_chart_returns_figure() -> None:
    fig = cost_latency_chart(_synthetic_df())
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() and ax.get_ylabel()
    assert len(ax.collections) > 0  # scatter points


def test_comparison_table_has_a_row_per_benchmark_metric() -> None:
    table = comparison_table(_synthetic_df())
    assert set(table["benchmark"].unique().to_list()) == {"frames", "browsecomp"}
    assert set(table["metric"].unique().to_list()) == {
        "accuracy_pct",
        "cost_usd_mean",
        "latency_s_mean",
    }
    assert "qwen3-8b" in table.columns and "gpt-4.1" in table.columns
    # gpt-4.1 accuracy > qwen3-8b accuracy in this synthetic data
    acc = table.filter(
        (pl.col("benchmark") == "frames") & (pl.col("metric") == "accuracy_pct")
    )
    assert acc["gpt-4.1"][0] > acc["qwen3-8b"][0]
