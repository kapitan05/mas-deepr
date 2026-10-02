"""Turn a series of ``scripts/eval_grpo_checkpoint.py`` runs into an
eval-accuracy-vs-training-step chart, one line per GRPO arm.

Reads every ``runs/milestones/post-grpo/{model_name}-step{N}/summary.parquet``
(each written by one ``eval_grpo_checkpoint.py`` invocation -- see that
script's ``--label`` default), extracts the checkpoint step from the
directory name, and produces one ``training_progress_chart`` per benchmark
present in the data. This is the "Gap 4b" chart named in
``docs/art-e-flow-for-mas-deepr.md`` -- eval success rate over GRPO steps,
not just the training-time reward curve ART already logs to W&B.

Usage:
    uv run python scripts/watch_grpo_progress.py \\
        --model-name mas-deepr-grpo mas-deepr-grpo-turns
    # -> runs/grpo/progress/{benchmark}_progress.png, one line per
    #    --model-name given (each arm's own -step* directories).
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import polars as pl

from mas_deepr.config import get_settings
from mas_deepr.evals.charts import training_progress_chart
from mas_deepr.telemetry import WandbSink, wandb_enabled

_STEP_SUFFIX = re.compile(r"^(?P<arm>.+)-step(?P<step>\d+)$")


def _load_progress_df(root: Path, model_names: list[str]) -> pl.DataFrame:
    """Every ``{model_name}-step{N}/summary.parquet`` under ``root``,
    tagged with the arm name (``model_name``, constant across its own
    steps) and the checkpoint ``step`` (int, parsed from the directory
    name) -- so ``training_progress_chart`` can group by arm and sort by
    step within it."""
    frames: list[pl.DataFrame] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        match = _STEP_SUFFIX.match(model_dir.name)
        if match is None or match.group("arm") not in model_names:
            continue
        parquet = model_dir / "summary.parquet"
        if not parquet.exists():
            continue
        frames.append(
            pl.read_parquet(parquet).with_columns(
                pl.lit(match.group("arm")).alias("model"),
                pl.lit(int(match.group("step"))).alias("step"),
            )
        )
    if not frames:
        raise FileNotFoundError(
            f"No {{model_name}}-step{{N}}/summary.parquet under {root} for "
            f"model names {model_names} -- run eval_grpo_checkpoint.py first."
        )
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        nargs="+",
        required=True,
        help="One or more ART model names (arms) to compare -- each arm's "
        "own set of --step evaluations, one line per arm per chart.",
    )
    parser.add_argument("--metric", default="score_mean")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    root = settings.runs_dir / "milestones" / "post-grpo"
    df = _load_progress_df(root, args.model_name)

    out_dir = settings.runs_dir / "grpo" / "progress"
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = sorted(df["benchmark"].unique().to_list())
    figures = {}
    for benchmark in benchmarks:
        fig = training_progress_chart(df, benchmark=benchmark, metric=args.metric)
        fig.savefig(out_dir / f"{benchmark}_progress.png", dpi=150)
        figures[benchmark] = fig

    print(f"Wrote {len(benchmarks)} chart(s) to {out_dir}")
    print(
        df.sort(["model", "benchmark", "step"]).select(
            "model", "benchmark", "step", args.metric, "n_errors"
        )
    )

    if wandb_enabled(disabled=args.no_wandb):
        with WandbSink(
            project=settings.wandb_project,
            run_name="grpo-progress",
            config={"model_names": args.model_name, "metric": args.metric},
            disabled=args.no_wandb,
        ) as sink:
            for benchmark, fig in figures.items():
                sink.log_figure(f"progress_{benchmark}", fig)


if __name__ == "__main__":
    main()
