"""Thesis milestone eval: full test-set run, gated behind an explicit --milestone flag.

This is the ONLY script that should touch the full benchmark test sets. Run
it exactly once per {baseline, post-dspy, post-grpo} milestone per the plan's
leakage-prevention design -- re-running it to "peek" at test performance
defeats the point of holding out a vault.

Usage:
    uv run python scripts/run_milestone_eval.py --milestone baseline \\
        --models qwen3-4b,qwen3-8b,qwen3-14b,gpt-oss-20b

    # Smoke-test the wiring on a handful of questions (NOT an official run):
    uv run python scripts/run_milestone_eval.py --milestone baseline \\
        --models qwen3-8b --smoke-limit 5
"""

import argparse
import asyncio

import polars as pl

from mas_deepr.agents import build_pipeline
from mas_deepr.config import get_model, get_settings
from mas_deepr.data import (
    load_browsecomp,
    load_frames,
    load_research_rubrics,
    write_manifest,
)
from mas_deepr.evals import JudgeClient, bootstrap_ci, run_benchmark, write_results
from mas_deepr.prompts import has_compiled_prompt
from mas_deepr.telemetry import TelemetryTracker, summarize

_ROLES = ("manager", "browser", "synthesizer")

_BENCHMARK_LOADERS = {
    "frames": load_frames,
    "browsecomp": load_browsecomp,
    "research_rubrics": load_research_rubrics,
}
_MILESTONES = ["baseline", "post-dspy", "post-grpo"]


async def _run_one(
    *, model_key: str, benchmark: str, milestone: str, smoke_limit: int | None
) -> pl.DataFrame:
    settings = get_settings()
    settings.ensure_dirs()
    spec = get_model(model_key)

    out_dir = settings.runs_dir / "milestones" / milestone / model_key
    run_id = f"{milestone}-{model_key}-{benchmark}"
    tracker = TelemetryTracker(
        out_dir / "telemetry.jsonl", run_id=run_id, phase=milestone
    )

    prefer_compiled = milestone != "baseline"
    if prefer_compiled and not all(has_compiled_prompt(role) for role in _ROLES):
        missing = [role for role in _ROLES if not has_compiled_prompt(role)]
        raise FileNotFoundError(
            f"--milestone {milestone} requires compiled prompts, but none exist "
            f"for role(s) {missing} -- run scripts/compile_dspy.py first. "
            "Silently falling back to hand-written prompts would make this "
            "milestone indistinguishable from 'baseline'."
        )
    pipeline = build_pipeline(
        spec=spec, settings=settings, tracker=tracker, prefer_compiled=prefer_compiled
    )

    questions = _BENCHMARK_LOADERS[benchmark](settings, limit=smoke_limit)
    write_manifest(questions, out_dir / f"{benchmark}_manifest.json")

    judge = None
    if benchmark in ("browsecomp", "research_rubrics"):
        judge = JudgeClient(
            spec=get_model(settings.judge_model), settings=settings, tracker=tracker
        )

    records = await run_benchmark(
        pipeline,
        questions,
        judge=judge,
        concurrency=settings.eval_concurrency,
        max_sub_queries=settings.max_sub_queries,
        max_tool_calls_per_query=settings.max_tool_calls_per_query,
    )
    write_results(records, out_dir / f"{benchmark}_results.parquet")

    # Exclude infra-error records from the accuracy CI -- an endpoint outage
    # or judge parse crash isn't a wrong answer; n_errors below still makes a
    # high-error run visible instead of silently dragging the mean down.
    mean, lo, hi = bootstrap_ci([r.score for r in records if r.error is None])
    return pl.DataFrame(
        [
            {
                "milestone": milestone,
                "model": model_key,
                "benchmark": benchmark,
                "n": len(records),
                "score_mean": mean,
                "score_ci_lo": lo,
                "score_ci_hi": hi,
                "n_errors": sum(1 for r in records if r.error),
            }
        ]
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone", required=True, choices=_MILESTONES)
    parser.add_argument(
        "--models", required=True, help="Comma-separated model registry keys"
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(_BENCHMARK_LOADERS),
        help="Comma-separated benchmark names (default: all three)",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="If set, cap questions per benchmark -- for pipeline smoke tests ONLY, "
        "not a valid thesis milestone number",
    )
    args = parser.parse_args()

    if args.smoke_limit is not None:
        print(
            f"** SMOKE MODE: limiting to {args.smoke_limit} questions/benchmark. "
            "This run does NOT count as an official milestone result. **"
        )

    settings = get_settings()
    models = args.models.split(",")
    benchmarks = args.benchmarks.split(",")

    summaries = []
    for model_key in models:
        for benchmark in benchmarks:
            print(f"Running {args.milestone} / {model_key} / {benchmark} ...")
            summaries.append(
                await _run_one(
                    model_key=model_key,
                    benchmark=benchmark,
                    milestone=args.milestone,
                    smoke_limit=args.smoke_limit,
                )
            )

    summary_df = pl.concat(summaries)
    out_path = settings.runs_dir / "milestones" / args.milestone / "summary.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_parquet(out_path)
    print(summary_df)
    print(f"\nSummary written to {out_path}")

    for model_key in models:
        telemetry_glob = (
            settings.runs_dir
            / "milestones"
            / args.milestone
            / model_key
            / "telemetry.jsonl"
        )
        if telemetry_glob.exists():
            print(f"\nTelemetry summary for {model_key}:")
            print(summarize(telemetry_glob))


if __name__ == "__main__":
    asyncio.run(main())
