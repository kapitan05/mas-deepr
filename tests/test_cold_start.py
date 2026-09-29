"""T3S anchor-masked SFT: real (tiny-scale) verification, not just mocked logic.

Uses ``sshleifer/tiny-gpt2`` -- a public, purpose-built-for-tests tiny GPT-2
checkpoint -- so this actually downloads a model, runs a real forward pass
to score anchors, and runs a real ``Trainer.train()`` step on CPU. Marked
``network`` so it can be skipped in fully offline environments; everything
else in the suite has no such dependency.
"""

import socket
from pathlib import Path

import pytest

# importorskip, not a bare `import torch` -- this module needs the optional
# rl-art extra (see pyproject.toml), not installed by CI. A bare import
# here would fail at *collection* time (before pytest.mark.art below ever
# gets a chance to deselect it via `-m 'not art'`), aborting the whole
# test run -- confirmed live 2026-09-29. importorskip turns a missing
# torch into a clean per-module skip instead.
torch = pytest.importorskip("torch")

from mas_deepr.rl.cold_start import (  # noqa: E402
    ColdStartConfig,
    compute_anchor_mask,
    mask_anchors,
    prepare_cold_start_batch,
    run_cold_start_training,
    tokenize_example,
)
from mas_deepr.rl.dataset import SFTExample  # noqa: E402

_TINY_MODEL = "sshleifer/tiny-gpt2"


def _has_network() -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=3).close()
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.art,
    pytest.mark.skipif(
        not _has_network(), reason="requires network to fetch tiny test model"
    ),
]


def test_anchor_mask_flags_only_completion_tokens_below_threshold() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    model.eval()

    example = SFTExample(
        role="synthesizer", prompt="Question: 2+2?", completion=" The answer is 4."
    )
    tokenized = tokenize_example(example, tokenizer, max_length=64)
    mask = compute_anchor_mask(model, tokenized, anchor_percentile=0.5)

    prompt_len = len(tokenizer(example.prompt, add_special_tokens=False)["input_ids"])
    # Prompt positions are never eligible for the anchor mask -- they're
    # already -100 in labels, so compute_anchor_mask must never flag them.
    assert not mask[:prompt_len].any()
    # About half the completion tokens should be flagged at percentile=0.5.
    completion_mask = mask[prompt_len:]
    assert 0 < completion_mask.sum().item() < len(completion_mask)


def test_mask_anchors_sets_flagged_positions_to_ignore_index() -> None:
    tokenized = {
        "input_ids": torch.tensor([1, 2, 3, 4]),
        "labels": torch.tensor([-100, -100, 3, 4]),
    }
    anchor_mask = torch.tensor([False, False, True, False])
    masked = mask_anchors(tokenized, anchor_mask)
    assert masked["labels"].tolist() == [-100, -100, -100, 4]
    # Original tokenized dict is untouched (mask_anchors returns a copy).
    assert tokenized["labels"].tolist() == [-100, -100, 3, 4]


def test_prepare_cold_start_batch_returns_masked_examples_for_every_input() -> None:
    examples = [
        SFTExample(role="manager", prompt="Q1", completion="A1 short."),
        SFTExample(role="browser", prompt="Q2", completion="A2 also short."),
    ]
    config = ColdStartConfig(model_name=_TINY_MODEL, anchor_percentile=0.3)
    _model, _tokenizer, masked = prepare_cold_start_batch(config, examples)
    assert len(masked) == 2
    for m in masked:
        assert "input_ids" in m and "labels" in m
        assert m["input_ids"].shape == m["labels"].shape


def test_run_cold_start_training_actually_updates_model_weights(
    tmp_path: Path,
) -> None:
    """Real, provable "it works": weights differ after one training step."""
    from transformers import AutoModelForCausalLM

    examples = [
        SFTExample(
            role="synthesizer",
            prompt="Question: capital of France?",
            completion=" The capital of France is Paris.",
        ),
        SFTExample(
            role="synthesizer",
            prompt="Question: 2+2?",
            completion=" The answer is 4.",
        ),
    ]
    config = ColdStartConfig(model_name=_TINY_MODEL, anchor_percentile=0.2)
    before = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    before_params = [p.clone() for p in before.parameters()]

    trained = run_cold_start_training(
        config,
        examples,
        output_dir=str(tmp_path / "cold_start_out"),
        num_train_epochs=1.0,
        per_device_train_batch_size=2,
        learning_rate=1e-2,  # large lr so a single tiny step is measurable
        max_steps=1,
    )

    after_params = [p.detach().cpu() for p in trained.parameters()]
    changed = any(
        not torch.equal(b, a) for b, a in zip(before_params, after_params, strict=True)
    )
    assert changed, "one training step should change at least one parameter tensor"

    out_dir = tmp_path / "cold_start_out"
    assert (out_dir / "config.json").exists()
    assert (
        any(out_dir.glob("*.safetensors")) or (out_dir / "pytorch_model.bin").exists()
    )
    assert (out_dir / "tokenizer_config.json").exists()


def test_run_cold_start_training_with_lora_trains_far_fewer_params(
    tmp_path: Path,
) -> None:
    """Real, provable "LoRA is actually on": trainable params « full model,
    and the adapter weights still change after a step."""
    from mas_deepr.rl.lora import LoraSettings

    examples = [
        SFTExample(
            role="synthesizer",
            prompt="Question: capital of France?",
            completion=" The capital of France is Paris.",
        ),
    ]
    config = ColdStartConfig(
        model_name=_TINY_MODEL, anchor_percentile=0.2, lora=LoraSettings(r=4, alpha=8)
    )
    trained = run_cold_start_training(
        config,
        examples,
        output_dir=str(tmp_path / "lora_out"),
        per_device_train_batch_size=1,
        learning_rate=1e-2,
        max_steps=1,
    )

    total = sum(p.numel() for p in trained.parameters())
    trainable = sum(p.numel() for p in trained.parameters() if p.requires_grad)
    assert 0 < trainable < total  # LoRA: only adapter params trainable

    # Standard LoRA init: A is random, B is zero -- so B provably moving off
    # zero is direct proof a real gradient step happened, no separate
    # "before" snapshot needed.
    lora_b_params = [
        p
        for name, p in trained.named_parameters()
        if "lora_B" in name and p.requires_grad
    ]
    assert lora_b_params, "expected at least one lora_B parameter"
    assert any(p.detach().abs().sum().item() > 0 for p in lora_b_params), (
        "lora_B should have moved off its zero init after a training step"
    )
