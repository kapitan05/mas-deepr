"""RL/Alignment module: T3S cold-start SFT, DPO (verified, TRL-based), and
a GRPO/ART rollout scaffold (documented, needs live GPU serving infra).

See docs/plan/plan-v2.md's Phase 4 for the full design and milestone table.

Lazy re-exports (PEP 562 module ``__getattr__``), not eager imports at
package-init time. Every name below pulls in torch/transformers/trl/peft
(the optional ``rl-art`` extra -- see ``pyproject.toml``), but
``mas_deepr.rl.trainable_client`` (needed unconditionally by
``agents/topology.py``'s GRPO rollout path, and itself needs none of
those) lives in this same package -- so an eager import here meant *any*
reach into this package at all, including just ``trainable_client``,
unconditionally pulled in torch. Confirmed live 2026-09-29: this broke
collection (not just execution) of six test files, ``rl-art``-marked or
not, in any environment without the extra installed -- exactly what CI
deliberately runs as (see the `art` pytest marker's own doc: tests
needing the real extra are meant to be *collectible but skippable* via
``-m 'not art'``, which an eager package-level import defeats regardless
of how the individual test file itself is marked.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mas_deepr.rl.cold_start import ColdStartConfig, run_cold_start_training
    from mas_deepr.rl.dataset import (
        DPOExample,
        SFTExample,
        build_dpo_pairs,
        build_sft_examples,
    )
    from mas_deepr.rl.dpo import DPOTrainConfig, run_dpo_training
    from mas_deepr.rl.lora import LoraSettings, to_peft_config
    from mas_deepr.rl.rubric_reward import (
        RolloutOutcome,
        group_advantages,
        score_rollout_group,
    )
    from mas_deepr.rl.s2l_po import S2LPOConfig, mixing_fraction

__all__ = [
    "ColdStartConfig",
    "DPOExample",
    "DPOTrainConfig",
    "LoraSettings",
    "RolloutOutcome",
    "S2LPOConfig",
    "SFTExample",
    "build_dpo_pairs",
    "build_sft_examples",
    "group_advantages",
    "mixing_fraction",
    "run_cold_start_training",
    "run_dpo_training",
    "score_rollout_group",
    "to_peft_config",
]

# name -> submodule it actually lives in.
_LAZY_SUBMODULE = {
    "ColdStartConfig": "cold_start",
    "run_cold_start_training": "cold_start",
    "DPOExample": "dataset",
    "SFTExample": "dataset",
    "build_dpo_pairs": "dataset",
    "build_sft_examples": "dataset",
    "DPOTrainConfig": "dpo",
    "run_dpo_training": "dpo",
    "LoraSettings": "lora",
    "to_peft_config": "lora",
    "RolloutOutcome": "rubric_reward",
    "group_advantages": "rubric_reward",
    "score_rollout_group": "rubric_reward",
    "S2LPOConfig": "s2l_po",
    "mixing_fraction": "s2l_po",
}


def __getattr__(name: str) -> object:
    submodule = _LAZY_SUBMODULE.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f"mas_deepr.rl.{submodule}")
    return getattr(mod, name)
