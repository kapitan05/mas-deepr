"""Benchmark comparison charts: accuracy per benchmark, cost vs latency.

Each function returns a ``matplotlib.figure.Figure`` (the caller saves it) --
same pattern as ART-E's ``evaluate/charts.py``. Input is a flattened polars
DataFrame with at least ``model``, ``source`` (benchmark), ``score``,
``latency_s``, ``cost_usd``, ``error`` columns -- typically the concatenated
per-model ``{benchmark}_results.parquet`` files (see
``scripts/plot_milestone.py``).
"""

from typing import cast

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from mas_deepr.evals.stats import bootstrap_ci


def _ok(df: pl.DataFrame) -> pl.DataFrame:
    """Rows without an infra error -- the accuracy denominator."""
    return df.filter(pl.col("error").is_null())


def _mean(series: pl.Series) -> float:
    return cast(float, series.mean() or 0.0)


def accuracy_comparison_chart(
    df: pl.DataFrame,
    *,
    models: list[str] | None = None,
    title: str = "Percentage of questions answered correctly",
) -> Figure:
    """Grouped bar chart: x = benchmark, one bar per model, y = mean
    ``score * 100`` with 95% bootstrap-CI error bars."""
    data = _ok(df)
    model_order = models or sorted(data["model"].unique().to_list())
    benchmarks = sorted(data["source"].unique().to_list())

    fig, ax = plt.subplots(figsize=(1.6 * len(benchmarks) + 3, 4.5))
    width = 0.8 / max(len(model_order), 1)

    for mi, model in enumerate(model_order):
        heights, lo_err, hi_err = [], [], []
        for b in benchmarks:
            scores = data.filter((pl.col("model") == model) & (pl.col("source") == b))[
                "score"
            ].to_list()
            mean, lo, hi = bootstrap_ci(scores)
            heights.append(mean * 100)
            lo_err.append((mean - lo) * 100)
            hi_err.append((hi - mean) * 100)
        xs = [i + mi * width for i in range(len(benchmarks))]
        ax.bar(
            xs,
            heights,
            width=width,
            label=model,
            yerr=[lo_err, hi_err],
            capsize=3,
        )

    ax.set_xticks(
        [i + width * (len(model_order) - 1) / 2 for i in range(len(benchmarks))]
    )
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("% correct")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    return fig


def cost_latency_chart(
    df: pl.DataFrame,
    *,
    models: list[str] | None = None,
    title: str = "Full-run cost vs. latency",
) -> Figure:
    """Labeled scatter: x = mean full-run latency (s), y = mean full-run
    cost (USD), one point per model (averaged across all benchmarks)."""
    data = _ok(df)
    model_order = models or sorted(data["model"].unique().to_list())

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for model in model_order:
        rows = data.filter(pl.col("model") == model)
        if rows.height == 0:
            continue
        x = _mean(rows["latency_s"])
        y = _mean(rows["cost_usd"])
        ax.scatter([x], [y], s=80)
        ax.annotate(
            model, (x, y), textcoords="offset points", xytext=(6, 4), fontsize="small"
        )

    ax.set_xlabel("mean full-run latency (s)")
    ax.set_ylabel("mean full-run cost (USD)")
    ax.set_title(title)
    ax.margins(0.15)
    fig.tight_layout()
    return fig


def training_progress_chart(
    df: pl.DataFrame,
    *,
    benchmark: str,
    metric: str = "score_mean",
    models: list[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Line chart: x = GRPO training step, y = ``metric`` (default
    ``score_mean``), one line per model/arm.

    This is the ART-E-comparison gap named explicitly in
    ``docs/art-e-flow-for-mas-deepr.md`` ("Gap 4b: no charting module for
    turning logged trajectories into training-progress line charts").
    Input is the concatenation of several ``scripts/eval_grpo_checkpoint.py``
    runs' ``summary.parquet`` files (one row per model/benchmark, run at
    different checkpoint steps) with a ``step`` column added -- see
    ``scripts/watch_grpo_progress.py``, which builds exactly this frame.
    Filtered to one ``benchmark`` at a time (call once per benchmark to
    compare all three) rather than faceting on one axes, since the natural
    x-axis here is step, not benchmark.
    """
    data = df.filter(pl.col("benchmark") == benchmark)
    model_order = models or sorted(data["model"].unique().to_list())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in model_order:
        rows = data.filter(pl.col("model") == model).sort("step")
        if rows.height == 0:
            continue
        ax.plot(rows["step"].to_list(), rows[metric].to_list(), marker="o", label=model)

    ax.set_xlabel("GRPO training step")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{benchmark}: {metric} vs. training step")
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    return fig


def comparison_table(df: pl.DataFrame) -> pl.DataFrame:
    """Metric x model matrix: one row per (benchmark, metric), one column
    per model. Mirrors ``benchmark_prompted_models.py``'s transposed table."""
    data = _ok(df)
    models = sorted(data["model"].unique().to_list())
    benchmarks = sorted(data["source"].unique().to_list())

    rows: list[dict[str, object]] = []
    for b in benchmarks:
        for metric_name, col, scale in (
            ("accuracy_pct", "score", 100.0),
            ("cost_usd_mean", "cost_usd", 1.0),
            ("latency_s_mean", "latency_s", 1.0),
        ):
            row: dict[str, object] = {"benchmark": b, "metric": metric_name}
            for m in models:
                sel = data.filter((pl.col("model") == m) & (pl.col("source") == b))[col]
                row[m] = round(_mean(sel) * scale, 4)
            rows.append(row)
    return pl.DataFrame(rows)


def to_markdown(df: pl.DataFrame) -> str:
    """Minimal GFM table -- avoids a ``tabulate`` dependency for one table."""
    cols = df.columns
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"
