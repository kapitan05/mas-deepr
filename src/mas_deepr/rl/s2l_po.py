"""S2L-PO: sibling-model rollout diversity via LoRA-adapter-swap.

Idea (from the paper): sample some GRPO group members from a frozen
*smaller* sibling model instead of temperature noise on the same model --
parameter-compression diversity is a cheaper/better exploration signal.
Realized here as a LoRA-adapter swap on the *same* base weights (adapter
on vs. adapter off/different adapter) rather than a second resident model,
so it fits a 1xH100 budget -- one set of base weights, not two models.

``mixing_fraction`` (the only piece of this module that's genuinely
infra-free and testable today) anneals from ``initial`` to 0 over the first
half of training, per the plan. The actual adapter swap
(``sibling_rollout_group``) needs a live ``peft``-wrapped model with a real
second adapter loaded -- scaffolded and documented, not verified, same
status as ``rollout.py``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peft import PeftModel


@dataclass
class S2LPOConfig:
    initial_mixing_fraction: float = 0.5
    anneal_over_fraction_of_training: float = 0.5  # anneal to 0 over first half


def mixing_fraction(config: S2LPOConfig, *, step: int, total_steps: int) -> float:
    """Linear anneal from ``initial_mixing_fraction`` to 0 over the first
    ``anneal_over_fraction_of_training`` of training, then stays at 0.

    Pure function, no model/backend dependency -- this is the actual
    add-on's control knob; everything else in this module is plumbing
    around it.
    """
    if total_steps <= 0:
        return 0.0
    anneal_steps = config.anneal_over_fraction_of_training * total_steps
    if anneal_steps <= 0:
        return 0.0
    progress = min(step / anneal_steps, 1.0)
    return config.initial_mixing_fraction * (1.0 - progress)


def should_use_sibling(
    config: S2LPOConfig, *, step: int, total_steps: int, draw: float
) -> bool:
    """Given a uniform random ``draw`` in [0, 1), decide whether this GRPO
    group member should come from the frozen sibling adapter."""
    return draw < mixing_fraction(config, step=step, total_steps=total_steps)


def swap_to_sibling_adapter(model: "PeftModel", *, sibling_adapter_name: str) -> None:
    """Activate the frozen sibling LoRA adapter on the shared base weights.

    **Unverified** -- needs a real ``peft.PeftModel`` with a second adapter
    already loaded via ``model.load_adapter(path, adapter_name=...)``. Left
    as a two-line wrapper (rather than inlined at call sites) so the actual
    swap mechanism has one obvious place to fix once run against real
    weights.
    """
    model.set_adapter(sibling_adapter_name)


def swap_to_policy_adapter(
    model: "PeftModel", *, policy_adapter_name: str = "default"
) -> None:
    """Switch back to the policy (trainable) adapter after a sibling rollout."""
    model.set_adapter(policy_adapter_name)
