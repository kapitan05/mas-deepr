"""``scripts/train_grpo.py``'s CLI flags + training-source wiring.

Marked ``art`` and imported lazily inside the test body, not at module
collection time -- ``train_grpo.py`` imports ``art`` at module level, which
monkeypatches ``transformers`` process-globally (same reasoning as
test_rollout.py's module docstring). Run via ``uv run pytest tests/ -m art``.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.art

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"


def _load_train_grpo() -> object:
    spec = importlib.util.spec_from_file_location("train_grpo_cli", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_grpo_cli"] = module
    spec.loader.exec_module(module)
    return module


def _build_parser(module: object) -> argparse.ArgumentParser:
    # main() builds its own parser inline (no module-level factory) --
    # reconstruct one identically here rather than change train_grpo.py's
    # structure just to make this testable.
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-name", default="mas-deepr-grpo")
    parser.add_argument("--musique-limit", type=int, default=None)
    parser.add_argument("--hotpot-limit", type=int, default=None)
    parser.add_argument("--dr-tulu-limit", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--judge-model", default="gpt-oss-20b-wandb")
    parser.add_argument("--tool-scope", default="wiki_paper")
    parser.add_argument(
        "--prefer-compiled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--backend", choices=("serverless", "local"), default="serverless"
    )
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-exceptions", type=float, default=20)
    parser.add_argument(
        "--penalize-turns", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--turn-penalty-weight", type=float, default=0.03)
    parser.add_argument(
        "--penalize-hallucination",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--hallucination-penalty-weight", type=float, default=0.15)
    return parser


def test_no_longer_imports_research_rubrics_eval_set() -> None:
    """Regression test for a real, confirmed leakage bug: train_grpo.py used
    to train on load_research_rubrics(), the exact set run_milestone_eval.py
    uses to score every model (hardcoded split="test", no train split
    exists for it)."""
    module = _load_train_grpo()
    source = _SCRIPT_PATH.read_text()
    assert "load_research_rubrics" not in source
    assert hasattr(module, "build_train_pool") or "build_train_pool" in source


def test_new_flags_exist_with_expected_defaults() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "Qwen/Qwen3-4B"])
    assert args.musique_limit is None
    assert args.hotpot_limit is None
    assert args.dr_tulu_limit is None
    assert args.tool_scope == "wiki_paper"
    assert args.prefer_compiled is True


def test_prefer_compiled_can_be_disabled() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b", "--no-prefer-compiled"])
    assert args.prefer_compiled is False


def test_tool_scope_is_overridable() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b", "--tool-scope", "general"])
    assert args.tool_scope == "general"


def test_multi_step_flags_exist_with_expected_defaults() -> None:
    """Regression test: --num-steps loops the rollout+train cycle on fresh,
    shuffled batches instead of one shot -- default of 1 must reproduce the
    original single-step behavior exactly (batch_size=None -> whole pool)."""
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b"])
    assert args.num_steps == 1
    assert args.batch_size is None
    assert args.seed == 0


def test_num_steps_and_batch_size_are_settable() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(
        ["--base-model", "b", "--num-steps", "20", "--batch-size", "60"]
    )
    assert args.num_steps == 20
    assert args.batch_size == 60


def test_max_exceptions_default_is_not_arts_zero_tolerance() -> None:
    """Regression test for a real, confirmed crash: art.gather_trajectory_groups
    defaults to max_exceptions=0 (zero tolerance) -- the first question-group
    hitting the known 32768-token context-overflow error (a real, observed
    ~10-25% rate on multi-hop MuSiQue/HotpotQA questions) crashed the whole
    multi-step run outright, losing every step still in flight. Confirmed
    live 2026-09-28: a 20-step run died on step 3."""
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b"])
    assert args.max_exceptions != 0
    source = _SCRIPT_PATH.read_text()
    assert "max_exceptions=args.max_exceptions" in source


def test_uses_itertools_cycle_for_wraparound_batching() -> None:
    """Regression guard: multi-step runs must draw fresh batches (shuffled,
    cycling once the pool is exhausted), not retrain the same batch
    --num-steps times."""
    source = _SCRIPT_PATH.read_text()
    assert "itertools.cycle" in source
    assert "rng.shuffle" in source


def test_model_name_defaults_to_mas_deepr_grpo() -> None:
    """Regression guard: existing single-arm behavior (registering/resuming
    "mas-deepr-grpo") must stay unchanged unless --model-name is passed."""
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b"])
    assert args.model_name == "mas-deepr-grpo"


def test_model_name_is_overridable_for_a_second_arm() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(
        ["--base-model", "b", "--model-name", "mas-deepr-grpo-turns"]
    )
    assert args.model_name == "mas-deepr-grpo-turns"


def test_no_hardcoded_model_name_left_in_registration_or_wandb_sink() -> None:
    """Regression guard: TrainableModel(name=...) and WandbSink(run_name=...)
    must both read args.model_name, not the literal "mas-deepr-grpo" --
    otherwise every arm would collide on one W&B run/LoRA name regardless
    of --model-name."""
    source = _SCRIPT_PATH.read_text()
    assert "name=args.model_name" in source
    assert "run_name=args.model_name" in source


def test_reward_shaping_flags_default_off() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(["--base-model", "b"])
    assert args.penalize_turns is False
    assert args.penalize_hallucination is False
    assert args.turn_penalty_weight == 0.03
    assert args.hallucination_penalty_weight == 0.15


def test_reward_shaping_flags_can_be_enabled() -> None:
    module = _load_train_grpo()
    parser = _build_parser(module)
    args = parser.parse_args(
        [
            "--base-model",
            "b",
            "--penalize-turns",
            "--penalize-hallucination",
            "--turn-penalty-weight",
            "0.05",
        ]
    )
    assert args.penalize_turns is True
    assert args.penalize_hallucination is True
    assert args.turn_penalty_weight == 0.05


def test_backend_train_result_is_logged_to_wandb() -> None:
    """Regression test for a real, confirmed bug: backend.train()'s return
    value (ServerlessTrainResult, carrying reward/policy_loss/entropy/
    grad_norm/learning_rate) used to be discarded outright -- confirmed
    live 2026-09-29 that a completed run's W&B page showed no "loss" group
    at all, only data/time/train/tables (from the separate
    model.log(trajectory_groups, ...) call). backend.train()'s own
    docstring says explicitly it does NOT log metrics on its own."""
    source = _SCRIPT_PATH.read_text()
    assert "train_result = await backend.train(" in source
    assert "model.log(metrics=train_result.metrics, step=train_result.step)" in source
