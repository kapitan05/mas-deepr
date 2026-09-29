"""Phase 4: DPO fine-tuning via Hugging Face TRL, from offline preference pairs.

Reads (prompt, chosen, rejected) triples from a JSONL file -- build one
with ``rl/rubric_reward.py::score_rollout_group`` +
``rl/dataset.py::build_dpo_pairs`` over multiple pipeline rollouts per
question, graded against a rubric pool. Each line:
{"prompt": ..., "chosen": ..., "rejected": ...}.

Usage:
    uv run python scripts/train_dpo.py \\
        --model-name Qwen/Qwen3-4B --pairs path/to/pairs.jsonl \\
        --output-dir runs/dpo/run1
"""

import argparse
import json
import os
from pathlib import Path

from mas_deepr.config import get_settings
from mas_deepr.logging_config import configure_logging
from mas_deepr.rl.dataset import DPOExample
from mas_deepr.rl.dpo import DPOTrainConfig, run_dpo_training
from mas_deepr.rl.lora import LoraSettings
from mas_deepr.telemetry import upload_lora_artifact


def _load_pairs(path: Path) -> list[DPOExample]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        pairs.append(
            DPOExample(
                prompt=data["prompt"], chosen=data["chosen"], rejected=data["rejected"]
            )
        )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="HF model id/path")
    parser.add_argument(
        "--pairs", required=True, type=Path, help="JSONL of prompt/chosen/rejected"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
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
    examples = _load_pairs(args.pairs)
    print(f"Loaded {len(examples)} DPO preference pairs from {args.pairs}")

    lora = LoraSettings(r=args.lora_r, alpha=args.lora_alpha) if args.lora else None
    config = DPOTrainConfig(
        model_name=args.model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lora=lora,
    )
    run_dpo_training(config, examples, output_dir=args.output_dir)
    print(f"DPO training done, checkpoint dir: {args.output_dir}")

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
