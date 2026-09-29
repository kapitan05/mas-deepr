"""Ad-hoc eval of a GRPO-trained checkpoint against the same capped
benchmark set used for the baseline/post-dspy runs, for a direct
before/after comparison.

A trained ``TrainableModel`` checkpoint is only servable through ART's own
registration mechanism (``model.get_inference_name(step=...)``), not
through a static registry key -- so this reuses ``run_milestone_eval.py``'s
own ``_run_one`` (generation + grading + telemetry, imported by path since
scripts/ isn't a package) against a ModelSpec built for that dynamic
endpoint, instead of duplicating any of that machinery.

Deliberately forces ``prefer_compiled=False`` (hand-written prompts) by
default: ``scripts/train_grpo.py`` was run with ``--no-prefer-compiled``
for this checkpoint (the DSPy-compiled prompts measurably hurt this model
on eval -- see the baseline/post-dspy comparison), so the fair "after GRPO"
number must be evaluated against the exact prompt source it was actually
trained against. Pass ``--prefer-compiled`` to override if you trained
against compiled prompts instead.

Usage:
    uv run python scripts/eval_grpo_checkpoint.py \\
        --model-name mas-deepr-grpo --base-model OpenPipe/Qwen3-14B-Instruct \\
        --benchmarks frames,research_qa,health_bench \\
        --benchmark-limit research_qa=100,health_bench=80,frames=50
"""

import argparse
import asyncio
import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import polars as pl

_SCRIPT_PATH = Path(__file__).resolve().with_name("run_milestone_eval.py")


def _load_run_milestone_eval() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_milestone_eval_grpo_checkpoint", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_milestone_eval_grpo_checkpoint"] = module
    spec.loader.exec_module(module)
    return module


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="mas-deepr-grpo")
    parser.add_argument("--project", default="mas-deepr")
    parser.add_argument("--base-model", default="OpenPipe/Qwen3-14B-Instruct")
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to eval (default: latest, via model.get_step())",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Registry key + output dir name (default: <model-name>-step<N>)",
    )
    parser.add_argument("--benchmarks", default="frames,research_qa,health_bench")
    parser.add_argument(
        "--benchmark-limit", default="research_qa=100,health_bench=80,frames=50"
    )
    parser.add_argument(
        "--prefer-compiled", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    import art
    from art.serverless.backend import ServerlessBackend

    model: art.TrainableModel = art.TrainableModel(
        name=args.model_name, project=args.project, base_model=args.base_model
    )
    backend = ServerlessBackend()
    await model.register(backend)
    step = args.step if args.step is not None else await model.get_step()
    inference_name = model.get_inference_name(step=step)
    print(f"Evaluating checkpoint step={step}: {inference_name}")
    if step == 0:
        print(
            "WARNING: step=0 is the untrained base model (no training has "
            "happened yet for this model name) -- did you mean to pass "
            "--step explicitly, or train first?"
        )

    if model.inference_api_key:
        os.environ["MAS_GRPO_CHECKPOINT_API_KEY"] = model.inference_api_key

    rme = _load_run_milestone_eval()

    from mas_deepr.config import ModelSpec
    from mas_deepr.config.models import MODEL_REGISTRY

    model_key = args.label or f"{args.model_name}-step{step}"
    MODEL_REGISTRY[model_key] = ModelSpec(
        key=model_key,
        model_id=inference_name,
        family="qwen3",
        self_hosted=False,
        api_base_url=model.inference_base_url,
        api_key_env="MAS_GRPO_CHECKPOINT_API_KEY",
        params_b=14,
    )

    milestone = "post-grpo"
    benchmark_limits = rme._parse_benchmark_limits(args.benchmark_limit)
    invocation_id = uuid.uuid4().hex[:8]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    print(f"Invocation id: {invocation_id}")

    rows: list[pl.DataFrame] = []
    for benchmark in benchmarks:
        print(f"Running {milestone} / {model_key} / {benchmark} ...")
        df = await rme._run_one(
            model_key=model_key,
            benchmark=benchmark,
            milestone=milestone,
            smoke_limit=None,
            benchmark_limits=benchmark_limits,
            memory="stateless",
            invocation_id=invocation_id,
            prefer_compiled_override=args.prefer_compiled,
        )
        rows.append(df)

    summary_df = pl.concat(rows)
    print(summary_df)
    out_dir = rme.get_settings().runs_dir / "milestones" / milestone / model_key
    out_path = out_dir / "summary.parquet"
    summary_df.write_parquet(out_path)
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
