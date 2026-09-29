"""T3S: anchor-token-masked SFT cold start.

T3S's own finding: naive full-sequence SFT loss includes "anchor tokens" --
completion positions the base model already predicts confidently before any
tuning -- and training on them suppresses gradient signal on the genuinely
informative tokens. Masking anchors out of the loss (``labels=-100``, same
mechanism a shifted-causal-LM loss already uses to ignore prompt tokens)
reproduces the paper's own "standard SFT hurts, T3S helps" comparison.
Paper's own config (batch 64, lr 1e-5, 50 steps, hundreds of examples,
single 8B run) is directly reproducible on 1xH100; ``tests/test_cold_start.py``
exercises the identical mechanism at CPU/tiny-model smoke scale.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from mas_deepr.rl.dataset import SFTExample
from mas_deepr.rl.lora import LoraSettings, to_peft_config
from mas_deepr.telemetry import wandb_enabled


@dataclass
class ColdStartConfig:
    model_name: str
    anchor_percentile: float = 0.2  # bottom-loss fraction masked as anchors
    max_length: int = 512
    # None -> full fine-tune (today's default, what the tests exercise).
    # Set to train a PEFT adapter instead -- required if the result needs
    # to be servable via W&B's Serverless LoRA Inference.
    lora: LoraSettings | None = None


def tokenize_example(
    example: SFTExample, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> dict[str, torch.Tensor]:
    """Tokenize one (prompt, completion) pair; prompt tokens are always
    masked from the loss (label=-100), matching standard SFT convention --
    the anchor mask below only ever narrows the *completion* tokens further.
    """
    prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(example.completion, add_special_tokens=False)[
        "input_ids"
    ]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + list(completion_ids))[:max_length]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


@torch.no_grad()
def compute_anchor_mask(
    model: PreTrainedModel,
    tokenized: dict[str, torch.Tensor],
    *,
    anchor_percentile: float,
) -> torch.Tensor:
    """Boolean mask, True at completion-token positions the base model
    already predicts confidently (anchors) -- these get excluded from the
    SFT loss. Confidence is measured as this base model's own per-token
    cross-entropy loss, the same signal T3S's own anchor definition uses.
    """
    labels = tokenized["labels"]
    mask = torch.zeros_like(labels, dtype=torch.bool)

    input_ids = tokenized["input_ids"].unsqueeze(0)
    logits = model(input_ids).logits[0]
    # Predict token t from logits at position t-1 (standard shifted LM loss).
    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    completion_positions = (shift_labels != -100).nonzero(as_tuple=True)[0]
    if len(completion_positions) == 0:
        return mask

    losses = torch.nn.functional.cross_entropy(
        shift_logits[completion_positions],
        shift_labels[completion_positions],
        reduction="none",
    )
    threshold = torch.quantile(losses, anchor_percentile)
    is_anchor = losses <= threshold
    mask[1:][completion_positions] = is_anchor
    return mask


def mask_anchors(
    tokenized: dict[str, torch.Tensor], anchor_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Set anchor-token labels to -100 (ignored by cross-entropy loss)."""
    labels = tokenized["labels"].clone()
    labels[anchor_mask] = -100
    return {**tokenized, "labels": labels}


def prepare_cold_start_batch(
    config: ColdStartConfig, examples: list[SFTExample]
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, list[dict[str, torch.Tensor]]]:
    """Load the base model/tokenizer and return anchor-masked training examples.

    One forward pass per example (to score anchors) before any gradient
    step -- cheap relative to training itself, and matches T3S's own
    two-stage design (score anchors with the frozen base model, then train).
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.eval()

    masked = []
    for ex in examples:
        tokenized = tokenize_example(ex, tokenizer, config.max_length)
        anchor_mask = compute_anchor_mask(
            model, tokenized, anchor_percentile=config.anchor_percentile
        )
        masked.append(mask_anchors(tokenized, anchor_mask))
    return model, tokenizer, masked


class _PadCollator:
    """Pad a batch of variable-length ``{input_ids, labels}`` examples.

    ``labels`` pad with -100 (ignored by the loss), ``input_ids`` pad with
    the tokenizer's pad token -- plain collation, no reliance on
    ``DataCollatorForLanguageModeling`` since that recomputes labels from
    ``input_ids`` and would silently erase the anchor mask.
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full(
            (len(batch), max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch):
            n = len(item["input_ids"])
            input_ids[i, :n] = item["input_ids"]
            labels[i, :n] = item["labels"]
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


class _MaskedDataset(TorchDataset):
    def __init__(self, examples: list[dict[str, torch.Tensor]]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._examples[idx]


def run_cold_start_training(
    config: ColdStartConfig,
    examples: list[SFTExample],
    *,
    output_dir: str,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 1e-5,
    max_steps: int = -1,
) -> PreTrainedModel | Any:
    """Run anchor-masked SFT end to end: score anchors, mask, train.

    Return type is ``PreTrainedModel`` for full fine-tune, or a peft
    ``PeftModel``/``PeftMixedModel`` when ``config.lora`` is set --
    left as ``Any`` for the LoRA branch rather than importing peft's
    exact union, since callers only re-save or re-load it either way.

    Paper's own config is batch 64 / lr 1e-5 / 50 steps / hundreds of
    examples on a single 8B model -- pass those via
    ``per_device_train_batch_size``/``learning_rate``/``max_steps`` for the
    real run; the defaults here are smoke-test scale, not the thesis config.
    """
    base_model, tokenizer, masked = prepare_cold_start_batch(config, examples)
    model: Any = base_model
    if config.lora is not None:
        from peft import get_peft_model

        model = get_peft_model(base_model, to_peft_config(config.lora))
        model.print_trainable_parameters()
    model.train()

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb" if wandb_enabled() else "none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=_MaskedDataset(masked),
        data_collator=_PadCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model
