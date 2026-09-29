"""Phase 4 / M6: T3S anchor-masked SFT cold start.

Reads role-tagged (prompt, completion) examples from a JSONL file --
build one from a milestone eval's ``raw_findings``/pass records (see
``PipelineResult`` in ``agents/topology.py``), or from hand-curated
examples for a first smoke run. Each line: {"role": ..., "prompt": ...,
"completion": ...}.

Paper's own config (batch 64, lr 1e-5, 50 steps, hundreds of examples,
single 8B run) is directly reproducible on 1xH100 -- pass it via the flags
below. Defaults here are for a CPU smoke run, not the thesis config.

Usage:
    uv run python scripts/train_cold_start.py \\
        --model-name Qwen/Qwen3-4B --examples path/to/examples.jsonl \\
        --output-dir runs/cold_start/run1 --learning-rate 1e-5 --max-steps 50
"""

import argparse
import json
import os
from pathlib import Path

from mas_deepr.config import get_settings
from mas_deepr.logging_config import configure_logging
from mas_deepr.rl.cold_start import ColdStartConfig, run_cold_start_training
from mas_deepr.rl.dataset import SFTExample
from mas_deepr.rl.lora import LoraSettings
from mas_deepr.telemetry import upload_lora_artifact


def _load_examples(path: Path) -> list[SFTExample]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        examples.append(
            SFTExample(
                role=data["role"], prompt=data["prompt"], completion=data["completion"]
            )
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="HF model id/path")
    parser.add_argument(
        "--examples", required=True, type=Path, help="JSONL of role/prompt/completion"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-percentile", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Train a PEFT LoRA adapter instead of full fine-tune -- required "
        "to publish to W&B Serverless LoRA Inference",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--publish-to-wandb",
        default=None,
        metavar="NAME",
        help="Upload the trained adapter as a W&B 'lora' artifact under this "
        "name (requires --lora and WANDB_API_KEY); prints the "
        "wandb-artifact:///... URI to use as a new ModelSpec.model_id",
    )
    args = parser.parse_args()

    configure_logging()
    # TRL's report_to="wandb" reads WANDB_PROJECT -- pin it to the shared
    # project so cold-start/DPO runs sit next to eval + GRPO runs.
    os.environ.setdefault("WANDB_PROJECT", get_settings().wandb_project)
    examples = _load_examples(args.examples)
    print(f"Loaded {len(examples)} cold-start examples from {args.examples}")

    lora = LoraSettings(r=args.lora_r, alpha=args.lora_alpha) if args.lora else None
    config = ColdStartConfig(
        model_name=args.model_name,
        anchor_percentile=args.anchor_percentile,
        lora=lora,
    )
    run_cold_start_training(
        config,
        examples,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )
    print(f"Cold-start SFT done, checkpoint dir: {args.output_dir}")

    if args.publish_to_wandb:
        if not args.lora:
            raise SystemExit(
                "--publish-to-wandb needs --lora (base fine-tunes\n"
                "aren't PEFT adapters and can't be uploaded the same way)"
            )
        settings = get_settings()
        uri = upload_lora_artifact(
            adapter_dir=args.output_dir,
            name=args.publish_to_wandb,
            base_model=args.model_name,
            project=settings.wandb_project,
        )
        print(f"Published to W&B. Use as a new ModelSpec.model_id:\n  {uri}")


if __name__ == "__main__":
    main()
