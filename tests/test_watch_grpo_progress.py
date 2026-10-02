"""``scripts/watch_grpo_progress.py::_load_progress_df`` -- step/arm parsing
from directory names, no network. Imported by path since scripts/ isn't a
package (matches test_run_milestone_eval_warmup.py's pattern).
"""

import importlib.util
import sys
from pathlib import Path

import polars as pl

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "watch_grpo_progress.py"
)
_spec = importlib.util.spec_from_file_location("watch_grpo_progress", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
watch_grpo_progress = importlib.util.module_from_spec(_spec)
sys.modules["watch_grpo_progress"] = watch_grpo_progress
_spec.loader.exec_module(watch_grpo_progress)


def _write_summary(
    root: Path, model_key: str, benchmark: str, score_mean: float
) -> None:
    d = root / model_key
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "milestone": ["post-grpo"],
            "model": [model_key],
            "benchmark": [benchmark],
            "n": [50],
            "n_errors": [0],
            "score_mean": [score_mean],
        }
    ).write_parquet(d / "summary.parquet")


def test_load_progress_df_parses_arm_and_step_from_dir_name(tmp_path: Path) -> None:
    _write_summary(tmp_path, "mas-deepr-grpo-step1", "frames", 0.128)
    _write_summary(tmp_path, "mas-deepr-grpo-step13", "frames", 0.170)

    df = watch_grpo_progress._load_progress_df(tmp_path, ["mas-deepr-grpo"])

    assert set(df["model"].unique().to_list()) == {"mas-deepr-grpo"}
    assert sorted(df["step"].to_list()) == [1, 13]


def test_load_progress_df_only_matches_requested_model_names(tmp_path: Path) -> None:
    _write_summary(tmp_path, "mas-deepr-grpo-step1", "frames", 0.128)
    _write_summary(tmp_path, "mas-deepr-grpo-turns-step1", "frames", 0.140)

    df = watch_grpo_progress._load_progress_df(tmp_path, ["mas-deepr-grpo"])

    assert df["model"].unique().to_list() == ["mas-deepr-grpo"]


def test_load_progress_df_ignores_dirs_without_a_step_suffix(tmp_path: Path) -> None:
    _write_summary(tmp_path, "mas-deepr-grpo-step1", "frames", 0.128)
    (tmp_path / "plots").mkdir()

    df = watch_grpo_progress._load_progress_df(tmp_path, ["mas-deepr-grpo"])

    assert df.height == 1


def test_load_progress_df_raises_when_nothing_matches(tmp_path: Path) -> None:
    _write_summary(tmp_path, "mas-deepr-grpo-step1", "frames", 0.128)
    try:
        watch_grpo_progress._load_progress_df(tmp_path, ["nonexistent-arm"])
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for no matching arms")
