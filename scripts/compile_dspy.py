"""Phase 2: compile Manager/Browser/Synthesizer prompts with DSPy's MIPROv2.

Optimizes against the train-pool split (MuSiQue + HotpotQA) only -- FRAMES,
ResearchRubrics, ResearchQA, and HealthBench are never touched here. Writes
``*.compiled.yaml`` to the prompt registry; run the ``post-dspy`` milestone
afterward to evaluate the frozen result.

Usage:
    uv run python scripts/compile_dspy.py --model qwen3-8b --auto light

    # Then, to score it:
    uv run python scripts/run_milestone_eval.py --milestone post-dspy \\
        --models qwen3-8b

    # Post-RL: optimize against a GRPO-trained checkpoint served on this
    # machine's GPU (same local-GPU flags as its training run):
    python scripts/compile_dspy.py --grpo-model-name qwen4b-s4-rubrics \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 --backend local --auto medium
"""

import argparse
import asyncio
from pathlib import Path
from typing import Any

from mas_deepr.config import Settings, get_model, get_settings, get_tool_scope
from mas_deepr.logging_config import configure_logging
from mas_deepr.optimize import compile_pipeline


async def _compile_against_checkpoint(
    args: argparse.Namespace, settings: Settings, compile_kwargs: dict[str, Any]
) -> dict[str, Path]:
    """Serve the trained checkpoint (rl/checkpoint_serving.py), then run the
    synchronous MIPROv2 compile in a worker thread so ART's event loop --
    and with it the model server -- stays alive while DSPy calls it."""
    from mas_deepr.rl.checkpoint_serving import (
        register_checkpoint_spec,
        serve_checkpoint,
    )
    from mas_deepr.rl.local_backend import local_model_kwargs

    served = await serve_checkpoint(
        model_name=args.grpo_model_name,
        project=args.project or settings.wandb_project,
        base_model=args.base_model,
        backend_kind=args.backend,
        model_kwargs=local_model_kwargs(args),
        step=args.step,
    )
    print(f"Compiling against checkpoint step={served.step}: {served.inference_name}")
    spec = register_checkpoint_spec(
        served,
        model_key=f"{args.grpo_model_name}-step{served.step}",
        base_model=args.base_model,
    )
    result: dict[str, Path] = await asyncio.to_thread(
        compile_pipeline, spec=spec, settings=settings, **compile_kwargs
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--model", help="Model registry key, e.g. qwen3-8b")
    target.add_argument(
        "--grpo-model-name",
        help="Optimize against this GRPO-trained ART model's checkpoint "
        "instead of a registry model (post-RL DSPy).",
    )
    parser.add_argument(
        "--base-model",
        help="Base model of --grpo-model-name, e.g. Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument(
        "--project", default=None, help="ART project (default: settings' W&B project)"
    )
    parser.add_argument(
        "--backend",
        choices=("local", "serverless"),
        default="local",
        help="Where --grpo-model-name's checkpoint is served.",
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Checkpoint step (default: latest)"
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 search budget",
    )
    parser.add_argument(
        "--musique-limit", type=int, default=200, help="Cap MuSiQue examples pulled"
    )
    parser.add_argument(
        "--hotpot-limit", type=int, default=200, help="Cap HotpotQA examples pulled"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="Train-pool val split fraction"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Compiled-prompt version tag (default: timestamp)",
    )
    from mas_deepr.rl.local_backend import add_local_gpu_args

    add_local_gpu_args(parser)
    args = parser.parse_args()
    if args.grpo_model_name and not args.base_model:
        parser.error("--grpo-model-name needs --base-model")

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    # Train-pool (MuSiQue+HotpotQA) is wiki-answerable -- wiki_paper scope
    # avoids hammering SearXNG's scraped engines at compile-time rollout
    # volume, same reliability reasoning as the eval side (see
    # config/tool_scopes.py).
    settings.mcp_enabled_tools = get_tool_scope("wiki_paper")
    compile_kwargs: dict[str, Any] = {
        "auto": args.auto,
        "musique_limit": args.musique_limit,
        "hotpot_limit": args.hotpot_limit,
        "val_fraction": args.val_fraction,
        "version": args.version,
    }
    if args.grpo_model_name:
        written = asyncio.run(
            _compile_against_checkpoint(args, settings, compile_kwargs)
        )
    else:
        written = compile_pipeline(
            spec=get_model(args.model), settings=settings, **compile_kwargs
        )

    print("Compiled prompts written:")
    for role, path in written.items():
        print(f"  {role}: {path}")


if __name__ == "__main__":
    main()
