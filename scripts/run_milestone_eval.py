"""Thesis milestone eval: full test-set run, gated behind an explicit --milestone flag.

This is the ONLY script that should touch the full benchmark test sets. Run
it exactly once per {baseline, post-dspy, post-grpo} milestone per the plan's
leakage-prevention design -- re-running it to "peek" at test performance
defeats the point of holding out a vault.

Usage:
    # Start the MCP tool backend first (Browser's tools proxy through it):
    uv run python scripts/run_mcp_server.py &

    uv run python scripts/run_milestone_eval.py --milestone baseline \\
        --models qwen3-4b,qwen3-8b,qwen3-14b,gpt-oss-20b

    # Smoke-test the wiring on a handful of questions (NOT an official run):
    uv run python scripts/run_milestone_eval.py --milestone baseline \\
        --models qwen3-8b --smoke-limit 5

    # Resume an interrupted run (crash, Ctrl-C) instead of starting over --
    # copy the invocation id the interrupted run printed:
    uv run python scripts/run_milestone_eval.py --milestone baseline \\
        --models qwen3-4b,qwen3-8b --resume-id a1b2c3d4
"""

import argparse
import asyncio
import logging
import uuid

import polars as pl

from mas_deepr.agents import ResearchPipeline, build_pipeline
from mas_deepr.config import Settings, get_model, get_settings, get_tool_scope
from mas_deepr.data import (
    load_frames,
    load_health_bench,
    load_research_qa,
    load_research_rubrics,
    write_manifest,
)
from mas_deepr.evals import (
    JudgeClient,
    bootstrap_ci,
    generate_benchmark,
    grade_benchmark,
    write_results,
)
from mas_deepr.logging_config import configure_logging
from mas_deepr.memory import MEMORY_REGISTRY, get_memory_strategy
from mas_deepr.prompts import has_compiled_prompt
from mas_deepr.telemetry import (
    TelemetryTracker,
    TrajectoryLogger,
    WandbSink,
    summarize,
    summarize_tool_calls,
)

logger = logging.getLogger(__name__)

_ROLES = ("manager", "browser", "synthesizer")


async def _warm_up(pipeline: ResearchPipeline) -> None:
    """One throwaway call before the timed benchmark loop starts.

    Confirmed live: a serverless-hosted model (llama-3.1-8b-wandb via W&B
    Inference) can take ~600s on its first call after being idle -- a real
    GPU cold start, not a bug -- versus <1s on every call after. Without
    this, that cost lands on whichever question happens to go first,
    massively skewing that question's (and the run's mean) latency for a
    reason that has nothing to do with the model's actual performance.
    Best-effort: a warm-up failure must not abort the real eval run --
    that failure will surface again, correctly, on the first real question.
    """
    try:
        await pipeline.manager.run("Reply with just: OK")
    except Exception as e:
        logger.warning("warm-up call failed (continuing anyway): %s", e)


_BENCHMARK_LOADERS = {
    "frames": load_frames,
    "research_rubrics": load_research_rubrics,
    "research_qa": load_research_qa,
    "health_bench": load_health_bench,
}
# Which MCP tools the Browser gets per benchmark -- "wiki_paper" (drops
# web_search/tavily_search) for benchmarks answerable from Wikipedia +
# academic paper search alone, "general" for genuinely-open-web tasks.
# See config/tool_scopes.py.
_BENCHMARK_TOOL_SCOPE = {
    "frames": "wiki_paper",
    "research_qa": "wiki_paper",
    "health_bench": "wiki_paper",
    "research_rubrics": "general",
}
assert _BENCHMARK_LOADERS.keys() == _BENCHMARK_TOOL_SCOPE.keys(), (
    "every benchmark needs a tool-scope entry and vice versa"
)
_MILESTONES = ["baseline", "post-dspy", "post-grpo"]


def _parse_benchmark_limits(raw: str) -> dict[str, int]:
    """Parse ``"health_bench=200,frames=50"`` into ``{"health_bench": 200,
    "frames": 50}``. Distinct from ``--smoke-limit``: this is a real,
    permanent per-benchmark size cap for an official milestone run (e.g.
    HealthBench's 5,000-row file vs. the other benchmarks' ~700-800), not
    a "this run doesn't count" smoke test."""
    limits: dict[str, int] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        if name not in _BENCHMARK_LOADERS:
            raise ValueError(
                f"--benchmark-limit: unknown benchmark {name!r}. "
                f"Known: {sorted(_BENCHMARK_LOADERS)}"
            )
        limits[name] = int(value)
    return limits


async def _run_one(
    *,
    model_key: str,
    benchmark: str,
    milestone: str,
    smoke_limit: int | None,
    benchmark_limits: dict[str, int],
    memory: str,
    invocation_id: str,
    prefer_compiled_override: bool | None = None,
) -> pl.DataFrame:
    settings = get_settings()
    settings.ensure_dirs()
    settings.mcp_enabled_tools = get_tool_scope(_BENCHMARK_TOOL_SCOPE[benchmark])
    spec = get_model(model_key)

    out_dir = settings.runs_dir / "milestones" / milestone / model_key
    # telemetry.jsonl/tool_calls.jsonl at this path are append-only and
    # deliberately reused across every invocation for this (milestone,
    # model_key) pair -- so a smoke test run five times leaves five times
    # the rows on disk. invocation_id makes run_id unique per *invocation*
    # (not per question -- every benchmark/question in one `main()` call
    # shares it), so summarize()/summarize_tool_calls() at the end of this
    # script -- and any later ad-hoc read of the same file -- can group by
    # run_id without silently summing today's run together with every past
    # one. Confirmed live 2026-09-25: without this, a fresh 1-question
    # smoke test on deepseek-chat/frames reported 2.5M input tokens --
    # the sum of every past smoke run ever pointed at this same file, not
    # that one question.
    run_id = f"{milestone}-{model_key}-{benchmark}-{invocation_id}"
    tracker = TelemetryTracker(
        out_dir / "telemetry.jsonl", run_id=run_id, phase=milestone
    )
    # Public-benchmark-only trajectory log (question/sub-questions/Browser
    # findings/final answer) -- see telemetry/trajectory_log.py for the
    # deliberate, narrow privacy-exception rationale.
    trajectory_logger = TrajectoryLogger(
        out_dir / "trajectories.jsonl",
        run_id=run_id,
        phase=milestone,
        model_key=model_key,
    )

    prefer_compiled = (
        prefer_compiled_override
        if prefer_compiled_override is not None
        else milestone != "baseline"
    )
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
    await _warm_up(pipeline)

    # --smoke-limit (if set) wins across the board -- it's the "just check
    # the wiring works" override. Otherwise a per-benchmark --benchmark-limit
    # applies (e.g. capping HealthBench's 5,000 rows to a size comparable
    # to the other benchmarks); benchmarks with no override run uncapped.
    limit = smoke_limit if smoke_limit is not None else benchmark_limits.get(benchmark)
    questions = _BENCHMARK_LOADERS[benchmark](settings, limit=limit)
    write_manifest(questions, out_dir / f"{benchmark}_manifest.json")

    # Data-driven, not a hardcoded benchmark-name list: any rubric-bearing
    # question needs a judge (matches evals/runner.py::_grade's dispatch),
    # so a new rubric-graded benchmark added to _BENCHMARK_LOADERS needs no
    # matching entry here.
    judge = None
    if any(q.rubrics is not None for q in questions):
        judge = JudgeClient(
            spec=get_model(settings.judge_model), settings=settings, tracker=tracker
        )

    def cost_lookup(qid: str) -> float:
        # LLM cost + tool-call cost (MCP backend -- $0 by default, nonzero
        # only if the opt-in Tavily fallback is enabled).
        return tracker.cost_for(qid) + tracker.tool_cost_for(qid)

    # Generation vs. grading split (ported from DR Tulu's decoupled
    # generate-dataset/evaluate.py, see the DR-Tulu-eval-adoption plan) --
    # re-grading (a judge fix, a different judge model) never needs to
    # re-run the expensive part. Resumption is keyed off invocation_id, not
    # (milestone, model_key, benchmark) alone -- same reasoning as run_id
    # above: a fresh smoke test must never silently skip questions because
    # an unrelated past invocation's generations file happens to share this
    # path. Pass --resume-id <that run's invocation_id> to actually resume.
    generations_path = out_dir / f"{benchmark}_generations_{invocation_id}.jsonl"
    generated = await generate_benchmark(
        pipeline,
        questions,
        concurrency=settings.eval_concurrency,
        max_sub_queries=settings.max_sub_queries,
        max_tool_calls_per_query=settings.max_tool_calls_per_query,
        memory=get_memory_strategy(memory),
        cost_lookup=cost_lookup,
        output_path=generations_path,
        trajectory_logger=trajectory_logger,
    )
    records = await grade_benchmark(
        generated,
        questions,
        judge=judge,
        concurrency=settings.eval_concurrency,
        cost_lookup=cost_lookup,
        trajectory_logger=trajectory_logger,
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
                "latency_s_mean": (
                    sum(r.latency_s for r in records) / len(records) if records else 0.0
                ),
                "cost_usd_mean": (
                    sum(r.cost_usd for r in records) / len(records) if records else 0.0
                ),
                "cost_usd_total": sum(r.cost_usd for r in records),
            }
        ]
    )


def _mirror_to_wandb(
    summary_df: pl.DataFrame, settings: Settings, milestone: str, *, disabled: bool
) -> None:
    """One W&B run per (milestone, model): per-benchmark accuracy + mean
    full-run cost/latency, plus a browsable per-question results table
    (score / latency / cost / final answer -- public benchmark content)."""
    from mas_deepr.telemetry import wandb_enabled

    if not wandb_enabled(disabled=disabled):
        return
    project = settings.wandb_project
    milestone_dir = settings.runs_dir / "milestones" / milestone
    for model_key in summary_df["model"].unique().to_list():
        rows = summary_df.filter(pl.col("model") == model_key)
        metrics: dict[str, float] = {}
        for row in rows.iter_rows(named=True):
            b = row["benchmark"]
            metrics[f"accuracy/{b}"] = row["score_mean"] * 100
            metrics[f"cost_usd_mean/{b}"] = row["cost_usd_mean"]
            metrics[f"latency_s_mean/{b}"] = row["latency_s_mean"]
            metrics[f"n_errors/{b}"] = float(row["n_errors"])
        table_cols = [
            "question_id",
            "source",
            "score",
            "latency_s",
            "cost_usd",
            "error",
            "question",
            "gold_answer",
            "sub_questions",
            "findings",
            "final_answer",
        ]
        table_rows: list[list[object]] = []
        for parquet in sorted((milestone_dir / model_key).glob("*_results.parquet")):
            for r in pl.read_parquet(parquet).iter_rows(named=True):
                table_rows.append(
                    [
                        r["question_id"],
                        r["source"],
                        r["score"],
                        round(r["latency_s"], 2),
                        round(r["cost_usd"], 5),
                        r["error"] or "",
                        str(r["question"])[:2000],
                        str(r["gold_answer"] or ""),
                        " | ".join(r["sub_questions"]),
                        str(" | ".join(r["findings"]))[:2000],
                        str(r["final_answer"])[:2000],
                    ]
                )
        # Isolated per model, same reasoning as _warm_up above: confirmed
        # live that a single model's transient W&B connectivity hiccup
        # (wandb.init()'s resume-status GraphQL check timing out) used to
        # abort this whole for loop, silently skipping the mirror for
        # every OTHER model too, even ones that would have synced fine.
        # Local results (summary.parquet, *_results.parquet) are already
        # complete regardless -- this only affects the optional W&B copy.
        try:
            with WandbSink(
                project=project,
                run_name=f"eval-{milestone}-{model_key}",
                config={"milestone": milestone, "model": model_key},
                disabled=disabled,
            ) as sink:
                sink.log_summary(metrics)
                if table_rows:
                    sink.log_table("results", table_cols, table_rows)
        except Exception as e:
            logger.warning(
                "W&B mirror failed for model=%s (continuing with other "
                "models, local results are already saved): %s",
                model_key,
                e,
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
        help=(
            "Comma-separated benchmark names "
            f"(default: all -- {sorted(_BENCHMARK_LOADERS)})"
        ),
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="If set, cap questions per benchmark -- for pipeline smoke tests ONLY, "
        "not a valid thesis milestone number",
    )
    parser.add_argument(
        "--benchmark-limit",
        default="",
        help="Comma-separated name=N per-benchmark size caps for an official "
        "milestone run, e.g. 'health_bench=200' -- unlike --smoke-limit, this "
        "still counts as a real milestone result, just intentionally sized "
        "(HealthBench's file is 5,000 rows vs. ~700-800 for the others).",
    )
    parser.add_argument(
        "--memory",
        default="stateless",
        choices=sorted(MEMORY_REGISTRY),
        help="Memory/context strategy (Phase 3, see memory/); default is the "
        "pre-Phase-3 single-pass behavior",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable the optional W&B mirror even if WANDB_API_KEY is set",
    )
    parser.add_argument(
        "--resume-id",
        default=None,
        help="Reuse a previous run's invocation id (printed as 'Invocation id: "
        "...' by that run) instead of generating a fresh one -- questions "
        "already present in that invocation's *_generations_<id>.jsonl files "
        "are skipped instead of re-run. Omit for a normal fresh invocation "
        "(default, unchanged behavior).",
    )
    args = parser.parse_args()

    configure_logging(logging.getLevelName(args.log_level.upper()))

    if args.smoke_limit is not None:
        print(
            f"** SMOKE MODE: limiting to {args.smoke_limit} questions/benchmark. "
            "This run does NOT count as an official milestone result. **"
        )

    settings = get_settings()
    models = args.models.split(",")
    benchmarks = args.benchmarks.split(",")
    benchmark_limits = _parse_benchmark_limits(args.benchmark_limit)
    for name, cap in benchmark_limits.items():
        print(f"Benchmark size cap: {name}={cap} (still a real milestone result)")

    # One id for this whole invocation (every model/benchmark pair below
    # shares it) -- see _run_one's run_id comment for why. --resume-id
    # reuses a prior one instead, so this invocation's generations files
    # land at the same paths that run's did.
    invocation_id = args.resume_id or uuid.uuid4().hex[:8]
    print(f"Invocation id: {invocation_id}")

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
                    benchmark_limits=benchmark_limits,
                    memory=args.memory,
                    invocation_id=invocation_id,
                )
            )

    summary_df = pl.concat(summaries)
    out_path = settings.runs_dir / "milestones" / args.milestone / "summary.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.write_parquet(out_path)
    print(summary_df)
    print(f"\nSummary written to {out_path}")

    # Best-effort: the W&B mirror is a reporting convenience, not the
    # source of truth (that's the parquet/JSONL already written above). A
    # W&B-side hiccup (bad row shape, network, quota) must not crash the
    # run after the real results are already safely on disk.
    try:
        _mirror_to_wandb(summary_df, settings, args.milestone, disabled=args.no_wandb)
    except Exception as e:
        print(f"\nW&B mirror failed (non-fatal, results are already saved): {e}")

    for model_key in models:
        telemetry_glob = (
            settings.runs_dir
            / "milestones"
            / args.milestone
            / model_key
            / "telemetry.jsonl"
        )
        # Suffix filter, not a fresh read: summarize()/summarize_tool_calls()
        # still (correctly) aggregate the whole accumulated file -- this
        # just prints only the groups this invocation actually produced,
        # not every historical smoke run sharing the same model_key/milestone
        # on disk. See _run_one's run_id comment.
        this_run = pl.col("run_id").str.ends_with(f"-{invocation_id}")
        if telemetry_glob.exists():
            print(f"\nTelemetry summary for {model_key}:")
            print(summarize(telemetry_glob).filter(this_run))

        tool_calls_path = telemetry_glob.parent / "tool_calls.jsonl"
        if tool_calls_path.exists():
            print(f"\nTool-call summary for {model_key} (expect $0 by default):")
            print(summarize_tool_calls(tool_calls_path).filter(this_run))


if __name__ == "__main__":
    asyncio.run(main())
