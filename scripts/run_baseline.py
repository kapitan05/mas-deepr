"""Dev-loop eval runner: run one model against one benchmark, any sample size.

Unlike ``run_milestone_eval.py`` this has no vault gating -- use it for quick
iteration on the pipeline/prompts. Milestone (thesis) numbers must come from
``run_milestone_eval.py`` instead.

Usage:
    uv run python scripts/run_baseline.py --model qwen3-8b --benchmark frames --limit 20
"""

import argparse
import asyncio
import uuid

from mas_deepr.agents import build_pipeline
from mas_deepr.config import get_model, get_settings
from mas_deepr.data import load_browsecomp, load_frames, load_research_rubrics
from mas_deepr.evals import JudgeClient, bootstrap_ci, run_benchmark, write_results
from mas_deepr.telemetry import TelemetryTracker

_BENCHMARK_LOADERS = {
    "frames": load_frames,
    "browsecomp": load_browsecomp,
    "research_rubrics": load_research_rubrics,
}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Model registry key, e.g. qwen3-8b"
    )
    parser.add_argument(
        "--benchmark", required=True, choices=sorted(_BENCHMARK_LOADERS)
    )
    parser.add_argument("--limit", type=int, default=20, help="Number of questions")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    spec = get_model(args.model)

    run_id = uuid.uuid4().hex[:8]
    telemetry_path = settings.runs_dir / f"dev-{run_id}" / "telemetry.jsonl"
    tracker = TelemetryTracker(telemetry_path, run_id=run_id, phase="dev")

    pipeline = build_pipeline(spec=spec, settings=settings, tracker=tracker)

    questions = _BENCHMARK_LOADERS[args.benchmark](settings, limit=args.limit)
    print(f"Loaded {len(questions)} {args.benchmark} questions")

    judge = None
    if args.benchmark in ("browsecomp", "research_rubrics"):
        judge_spec = get_model(args.judge_model)
        judge = JudgeClient(spec=judge_spec, settings=settings, tracker=tracker)

    records = await run_benchmark(
        pipeline,
        questions,
        judge=judge,
        concurrency=settings.eval_concurrency,
        max_sub_queries=settings.max_sub_queries,
        max_tool_calls_per_query=settings.max_tool_calls_per_query,
    )

    scores = [r.score for r in records]
    mean, lo, hi = bootstrap_ci(scores)
    n_errors = sum(1 for r in records if r.error)
    print(
        f"\n{args.benchmark} / {args.model}: "
        f"score={mean:.3f} (95% CI [{lo:.3f}, {hi:.3f}]) "
        f"n={len(records)} errors={n_errors}"
    )

    results_path = (
        settings.runs_dir / f"dev-{run_id}" / f"{args.benchmark}_results.parquet"
    )
    write_results(records, results_path)
    print(f"Results written to {results_path}")
    print(f"Telemetry written to {telemetry_path}")


if __name__ == "__main__":
    asyncio.run(main())
