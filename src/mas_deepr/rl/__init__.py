"""RL/Alignment module: T3S cold-start SFT, DPO (verified, TRL-based), and
a GRPO/ART rollout scaffold (documented, needs live GPU serving infra).

See docs/plan/plan-v2.md's Phase 4 for the full design and milestone table.
"""

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
