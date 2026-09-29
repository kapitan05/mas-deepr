"""Shared PEFT LoRA config for cold-start SFT and DPO.

Defaults (rank 16) match the ceiling W&B's Serverless LoRA Inference
documents for its supported base models (``meta-llama/Llama-3.1-8B-
Instruct`` included) -- an adapter trained here with the defaults is
servable there unmodified. See ``docs/wandb-inference-and-lora.md``.
"""

from dataclasses import dataclass

from peft import LoraConfig


@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None  # None -> peft's per-architecture default


def to_peft_config(settings: LoraSettings) -> LoraConfig:
    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=settings.target_modules,
        task_type="CAUSAL_LM",
    )
