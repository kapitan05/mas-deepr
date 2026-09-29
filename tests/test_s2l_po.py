"""S2L-PO mixing-fraction schedule: pure logic, no model/backend dependency.

``swap_to_sibling_adapter``/``swap_to_policy_adapter`` need a real
``peft.PeftModel`` with a second adapter loaded and aren't exercised here --
see rl/s2l_po.py's own docstring for that status.
"""

import pytest

from mas_deepr.rl.s2l_po import S2LPOConfig, mixing_fraction, should_use_sibling


def test_mixing_fraction_starts_at_initial_value() -> None:
    config = S2LPOConfig(initial_mixing_fraction=0.5)
    assert mixing_fraction(config, step=0, total_steps=100) == pytest.approx(0.5)


def test_mixing_fraction_anneals_to_zero_at_midpoint() -> None:
    config = S2LPOConfig(
        initial_mixing_fraction=0.5, anneal_over_fraction_of_training=0.5
    )
    assert mixing_fraction(config, step=50, total_steps=100) == pytest.approx(0.0)


def test_mixing_fraction_stays_zero_past_anneal_window() -> None:
    config = S2LPOConfig(
        initial_mixing_fraction=0.5, anneal_over_fraction_of_training=0.5
    )
    assert mixing_fraction(config, step=90, total_steps=100) == pytest.approx(0.0)


def test_mixing_fraction_handles_zero_total_steps() -> None:
    config = S2LPOConfig()
    assert mixing_fraction(config, step=0, total_steps=0) == 0.0


def test_should_use_sibling_respects_draw_against_schedule() -> None:
    config = S2LPOConfig(initial_mixing_fraction=0.5)
    assert should_use_sibling(config, step=0, total_steps=100, draw=0.1) is True
    assert should_use_sibling(config, step=0, total_steps=100, draw=0.9) is False
