"""Curated "did GRPO help, did folding help" comparison chart + table.

``scripts/plot_milestone.py`` has no model filter -- pointed at
``post-grpo`` it would dump all ~29 checkpoint-step directories into one
unreadable chart. This script instead hand-picks a small, representative
set of already-evaluated checkpoints (no new eval calls, pure local-data
read) and reuses the same tested chart functions
(``evals/charts.py::accuracy_comparison_chart``, ``comparison_table``,
``cost_latency_chart``) on just that subset -- the single clearest
"before vs. during vs. after GRPO, plus the folding fix" figure for the
thesis.

Usage:
    uv run python scripts/plot_grpo_results.py
"""

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

# (milestone dir name, model-key dir name, display label for the chart)
_CURATED: list[tuple[str, str, str]] = [
    ("baseline", "qwen3-14b-instruct-wandb", "base model"),
    ("post-grpo", "mas-deepr-grpo-step65", "GRPO step65"),
    ("post-grpo", "mas-deepr-grpo-step324", "GRPO step324 (best)"),
    ("post-grpo", "mas-deepr-grpo-folding-step324", "GRPO step324 + folding"),
]


def _load_curated(root: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for milestone, model_dir, label in _CURATED:
        model_path = root / milestone / model_dir
        parquets = sorted(model_path.glob("*_results.parquet"))
        if not parquets:
            raise FileNotFoundError(f"No *_results.parquet under {model_path}")
        for parquet in parquets:
            frames.append(
                pl.read_parquet(parquet).with_columns(pl.lit(label).alias("model"))
            )
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    settings = get_settings()
    df = _load_curated(settings.runs_dir / "milestones")

    out_dir = settings.runs_dir / "grpo" / "results_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_order = [label for _, _, label in _CURATED]
    fig = accuracy_comparison_chart(
        df,
        models=model_order,
        title="mas-deepr: base model -> GRPO training -> folding fix",
    )
    fig.savefig(out_dir / "accuracy_comparison.png", dpi=150)

    fig2 = cost_latency_chart(df, models=model_order)
    fig2.savefig(out_dir / "cost_latency.png", dpi=150)

    table = comparison_table(df)
    (out_dir / "comparison_table.md").write_text(to_markdown(table))

    print(f"Wrote charts + table to {out_dir}")
    print(table)


if __name__ == "__main__":
    main()
