"""DPO fine-tuning: real (tiny-scale) verification via TRL's DPOTrainer.

Same tiny public checkpoint and network-skip convention as
``tests/test_cold_start.py``.
"""

import socket
from pathlib import Path

import pytest

# importorskip, not a bare `import torch` -- see test_cold_start.py's
# comment on the same line for why (rl-art extra, not installed by CI).
torch = pytest.importorskip("torch")

from mas_deepr.rl.dataset import DPOExample, build_dpo_pairs  # noqa: E402
from mas_deepr.rl.dpo import (  # noqa: E402
    DPOTrainConfig,
    build_dpo_dataset,
    run_dpo_training,
)
from mas_deepr.rl.rubric_reward import RolloutOutcome  # noqa: E402

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


def test_build_dpo_pairs_skips_zero_variance_groups() -> None:
    tie_group = [
        RolloutOutcome(question_id="q1", prompt="p", response="a", score=0.5),
        RolloutOutcome(question_id="q1", prompt="p", response="b", score=0.5),
    ]
    real_group = [
        RolloutOutcome(question_id="q2", prompt="p2", response="good", score=0.9),
        RolloutOutcome(question_id="q2", prompt="p2", response="bad", score=0.1),
    ]
    pairs = build_dpo_pairs([tie_group, real_group])
    assert len(pairs) == 1
    assert pairs[0].chosen == "good"
    assert pairs[0].rejected == "bad"


def test_build_dpo_dataset_has_expected_columns() -> None:
    examples = [DPOExample(prompt="p", chosen="c", rejected="r")]
    ds = build_dpo_dataset(examples)
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
    assert ds[0] == {"prompt": "p", "chosen": "c", "rejected": "r"}


def test_run_dpo_training_actually_updates_model_weights(tmp_path: Path) -> None:
    from transformers import AutoModelForCausalLM

    examples = [
        DPOExample(
            prompt="Question: capital of France?",
            chosen="The capital of France is Paris.",
            rejected="The capital of France is Berlin.",
        ),
        DPOExample(
            prompt="Question: 2+2?",
            chosen="The answer is 4.",
            rejected="The answer is 5.",
        ),
    ]
    config = DPOTrainConfig(
        model_name=_TINY_MODEL,
        learning_rate=1e-2,
        per_device_train_batch_size=2,
        max_steps=1,
    )
    before = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    before_params = [p.clone() for p in before.parameters()]

    trained = run_dpo_training(config, examples, output_dir=str(tmp_path / "dpo_out"))

    after_params = [p.detach().cpu() for p in trained.parameters()]
    changed = any(
        not torch.equal(b, a) for b, a in zip(before_params, after_params, strict=True)
    )
    assert changed, "one DPO training step should change at least one parameter tensor"


def test_run_dpo_training_with_lora_trains_far_fewer_params(tmp_path: Path) -> None:
    from mas_deepr.rl.lora import LoraSettings

    examples = [
        DPOExample(
            prompt="Question: capital of France?",
            chosen="The capital of France is Paris.",
            rejected="The capital of France is Berlin.",
        ),
    ]
    config = DPOTrainConfig(
        model_name=_TINY_MODEL,
        learning_rate=1e-2,
        per_device_train_batch_size=1,
        max_steps=1,
        lora=LoraSettings(r=4, alpha=8),
    )
    trained = run_dpo_training(config, examples, output_dir=str(tmp_path / "lora_dpo"))

    total = sum(p.numel() for p in trained.parameters())
    trainable = sum(p.numel() for p in trained.parameters() if p.requires_grad)
    assert 0 < trainable < total

    lora_b_params = [
        p
        for name, p in trained.named_parameters()
        if "lora_B" in name and p.requires_grad
    ]
    assert lora_b_params
    assert any(p.detach().abs().sum().item() > 0 for p in lora_b_params), (
        "lora_B should have moved off its zero init after a DPO step"
    )
