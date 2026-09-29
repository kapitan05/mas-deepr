"""Build SFT / DPO training examples from pipeline rollouts.

Cold-start SFT (T3S, ``cold_start.py``) wants role-tagged (prompt,
completion) examples per role. DPO (``dpo.py``) wants (prompt, chosen,
rejected) pairs -- built here by pairing the best- and worst-scored rollout
within a group graded by ``rubric_reward``, discarding groups with zero
score spread (nothing to prefer between).
"""

from dataclasses import dataclass

from mas_deepr.rl.rubric_reward import RolloutOutcome


@dataclass
class SFTExample:
    role: str  # manager | browser | synthesizer
    prompt: str
    completion: str


@dataclass
class DPOExample:
    prompt: str
    chosen: str
    rejected: str


def build_sft_examples(
    role_prompts: list[tuple[str, str, str]],
) -> list[SFTExample]:
    """``role_prompts`` is a list of (role, prompt, completion) triples,
    typically pulled from ``PipelineResult.raw_findings``/role-tagged
    trajectories captured by a milestone eval run."""
    return [SFTExample(role=r, prompt=p, completion=c) for r, p, c in role_prompts]


def build_dpo_pairs(groups: list[list[RolloutOutcome]]) -> list[DPOExample]:
    """One DPO pair per group: best- vs. worst-scored rollout.

    Groups where every rollout scored identically are skipped -- DPO's
    preference loss needs a real chosen/rejected gap, not a tie. This is the
    same "zero variance carries no signal" principle as
    ``rubric_reward.group_advantages``, applied to pairwise preference
    instead of GRPO's group-normalized advantage.
    """
    pairs = []
    for group in groups:
        if len(group) < 2:
            continue
        best = max(group, key=lambda o: o.score)
        worst = min(group, key=lambda o: o.score)
        if best.score <= worst.score:
            continue
        pairs.append(
            DPOExample(
                prompt=best.prompt,
                chosen=best.response,
                rejected=worst.response,
            )
        )
    return pairs
