"""Dev-loop eval runner: run one model against one benchmark, any sample size.

Unlike ``run_milestone_eval.py`` this has no vault gating -- use it for quick
iteration on the pipeline/prompts. Milestone (thesis) numbers must come from
``run_milestone_eval.py`` instead.

Usage:
    # Start the MCP tool backend first (Browser's tools proxy through it):
    uv run python scripts/run_mcp_server.py &

    uv run python scripts/run_baseline.py --model qwen3-8b --benchmark frames --limit 20

    # Resume the same model/benchmark's previous run instead of starting
    # over (skips already-generated questions):
    uv run python scripts/run_baseline.py --model qwen3-8b --benchmark frames \\
        --limit 20 --resume
"""

import argparse
import asyncio
import logging
import uuid

from mas_deepr.agents import build_pipeline
from mas_deepr.config import get_model, get_settings
from mas_deepr.data import load_browsecomp, load_frames, load_research_rubrics
from mas_deepr.evals import (
    JudgeClient,
    bootstrap_ci,
    generate_benchmark,
    grade_benchmark,
    write_results,
)
from mas_deepr.logging_config import configure_logging
from mas_deepr.memory import MEMORY_REGISTRY, get_memory_strategy
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
    parser.add_argument("--judge-model", default="gemini-2.5-flash-lite")
    parser.add_argument(
        "--memory",
        default="stateless",
        choices=sorted(MEMORY_REGISTRY),
        help="Memory/context strategy (Phase 3, see memory/); default is the "
        "pre-Phase-3 single-pass behavior",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same --model/--benchmark's previous dev run instead of "
        "starting fresh -- skips questions already completed in that run's "
        "generations file (see evals/runner.py::generate_benchmark's resumption "
        "support, ported from DR Tulu's generate-dataset). Without this flag, "
        "every invocation gets its own fresh run_id, same as before this existed.",
    )
    args = parser.parse_args()

    configure_logging(logging.getLevelName(args.log_level.upper()))
    settings = get_settings()
    settings.ensure_dirs()
    spec = get_model(args.model)

    # --resume needs a *stable* id to find the same output path again next
    # time; without it, a fresh random id every invocation is unchanged
    # from before --resume existed (never accidentally collides with a
    # concurrent unrelated dev run).
    run_id = f"{args.model}-{args.benchmark}" if args.resume else uuid.uuid4().hex[:8]
    telemetry_path = settings.runs_dir / f"dev-{run_id}" / "telemetry.jsonl"
    tracker = TelemetryTracker(telemetry_path, run_id=run_id, phase="dev")

    pipeline = build_pipeline(spec=spec, settings=settings, tracker=tracker)

    questions = _BENCHMARK_LOADERS[args.benchmark](settings, limit=args.limit)
    print(f"Loaded {len(questions)} {args.benchmark} questions")

    judge = None
    if args.benchmark in ("browsecomp", "research_rubrics"):
        judge_spec = get_model(args.judge_model)
        judge = JudgeClient(spec=judge_spec, settings=settings, tracker=tracker)

    def cost_lookup(qid: str) -> float:
        # LLM cost + tool-call cost (MCP backend -- $0 by default, nonzero
        # only if the opt-in Tavily fallback is enabled).
        return tracker.cost_for(qid) + tracker.tool_cost_for(qid)

    generations_path = (
        settings.runs_dir / f"dev-{run_id}" / f"{args.benchmark}_generations.jsonl"
    )
    generated = await generate_benchmark(
        pipeline,
        questions,
        concurrency=settings.eval_concurrency,
        max_sub_queries=settings.max_sub_queries,
        max_tool_calls_per_query=settings.max_tool_calls_per_query,
        memory=get_memory_strategy(args.memory),
        cost_lookup=cost_lookup,
        # Always written incrementally (crash-safety) -- only *resuming*
        # from it on a later invocation is gated on --resume (see run_id
        # above: without --resume this path is unique per run anyway, so
        # there's nothing to resume from and this is a no-op safety net).
        output_path=generations_path,
    )
    records = await grade_benchmark(
        generated,
        questions,
        judge=judge,
        concurrency=settings.eval_concurrency,
        cost_lookup=cost_lookup,
    )

    # CI is computed over non-error records only -- an infra hiccup (endpoint
    # outage, judge parse crash) isn't a wrong answer and shouldn't silently
    # drag the accuracy estimate toward 0; n_errors is still reported so a
    # run with many infra failures doesn't read as a clean high score either.
    scores = [r.score for r in records if r.error is None]
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
