"""Turn a milestone eval's per-model result parquets into the comparison
plots + table.

Reads ``runs/milestones/{milestone}/{model}/{benchmark}_results.parquet``
(written by ``run_milestone_eval.py``), builds one flat DataFrame, and
writes:

    runs/milestones/{milestone}/plots/accuracy_comparison.png
    runs/milestones/{milestone}/plots/cost_latency.png
    runs/milestones/{milestone}/plots/comparison_table.{md,html}

If ``WANDB_API_KEY`` is set, the figures are also mirrored to the
``mas-deepr`` W&B project.

Usage:
    uv run python scripts/plot_milestone.py --milestone baseline
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl

from mas_deepr.config import get_settings
from mas_deepr.evals.charts import (
    accuracy_comparison_chart,
    comparison_table,
    cost_latency_chart,
    to_markdown,
)
from mas_deepr.telemetry import WandbSink, wandb_enabled


def _load_milestone_df(root: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name == "plots":
            continue
        for parquet in sorted(model_dir.glob("*_results.parquet")):
            frames.append(
                pl.read_parquet(parquet).with_columns(
                    pl.lit(model_dir.name).alias("model")
                )
            )
    if not frames:
        raise FileNotFoundError(f"No *_results.parquet under {root}")
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    milestone_dir = settings.runs_dir / "milestones" / args.milestone
    df = _load_milestone_df(milestone_dir)

    plots_dir = milestone_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    acc_fig = accuracy_comparison_chart(df)
    cost_fig = cost_latency_chart(df)
    acc_fig.savefig(plots_dir / "accuracy_comparison.png", dpi=150)
    cost_fig.savefig(plots_dir / "cost_latency.png", dpi=150)

    table = comparison_table(df)
    (plots_dir / "comparison_table.md").write_text(to_markdown(table), encoding="utf-8")
    (plots_dir / "comparison_table.html").write_text(
        table.to_pandas().to_html(index=False), encoding="utf-8"
    )

    print(f"Wrote plots + table to {plots_dir}")
    print(table)

    if wandb_enabled(disabled=args.no_wandb):
        with WandbSink(
            project=settings.wandb_project,
            run_name=f"eval-{args.milestone}-plots",
            config={"milestone": args.milestone},
            disabled=args.no_wandb,
        ) as sink:
            sink.log_figure("accuracy_comparison", acc_fig)
            sink.log_figure("cost_latency", cost_fig)


if __name__ == "__main__":
    main()
