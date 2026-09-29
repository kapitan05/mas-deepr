"""DPO fine-tuning via Hugging Face TRL, from offline preference pairs.

Chosen over an online (GRPO-style) loop for this module because TRL's
``DPOTrainer`` trains directly against a static (prompt, chosen, rejected)
dataset -- no live rollout server needed, so it runs and is verifiable on
whatever machine has the model weights, no GPU rollout infra required. The
preference pairs themselves (``rl/dataset.py::build_dpo_pairs``) come from
grading multiple pipeline rollouts per question with
``rl/rubric_reward.py`` and keeping the best/worst pair -- so the *signal*
DPO trains on is the same rubric-reward mechanism GRPO would use, just
consumed offline instead of online.
"""

from dataclasses import dataclass

import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from trl import DPOConfig, DPOTrainer

from mas_deepr.rl.dataset import DPOExample
from mas_deepr.rl.lora import LoraSettings, to_peft_config
from mas_deepr.telemetry import wandb_enabled


@dataclass
class DPOTrainConfig:
    model_name: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    num_train_epochs: float = 1.0
    max_steps: int = -1
    max_length: int = 512
    # None -> full fine-tune (today's default, what the tests exercise).
    # Set to train a PEFT adapter instead -- required if the result needs
    # to be servable via W&B's Serverless LoRA Inference.
    lora: LoraSettings | None = None


def build_dpo_dataset(examples: list[DPOExample]) -> datasets.Dataset:
    """Convert ``DPOExample`` records into TRL's expected column schema."""
    return datasets.Dataset.from_dict(
        {
            "prompt": [e.prompt for e in examples],
            "chosen": [e.chosen for e in examples],
            "rejected": [e.rejected for e in examples],
        }
    )


def run_dpo_training(
    config: DPOTrainConfig,
    examples: list[DPOExample],
    *,
    output_dir: str,
) -> PreTrainedModel:
    """Run DPO end to end on a static preference-pair dataset.

    Returns the trained model. Caller decides what to do with it (save,
    push to a registry, hand to ``evals/runner.py`` for an A/B). No
    reference-model argument is passed explicitly: for full fine-tuning,
    ``DPOTrainer`` builds its own frozen copy of the starting weights
    (policy vs. frozen reference, no separate reward model); with
    ``config.lora`` set, it needs no separate reference model at all --
    TRL gets reference logits by disabling the adapter on the same
    weights, which is also why LoRA+DPO uses meaningfully less memory
    than full-fine-tune+DPO.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name)

    peft_config = to_peft_config(config.lora) if config.lora is not None else None
    dataset = build_dpo_dataset(examples)
    args = DPOConfig(
        output_dir=output_dir,
        beta=config.beta,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        max_length=config.max_length,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb" if wandb_enabled() else "none",
    )
    trainer = DPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model
