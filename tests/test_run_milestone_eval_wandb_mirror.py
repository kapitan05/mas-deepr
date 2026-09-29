"""``scripts/run_milestone_eval.py::_mirror_to_wandb`` -- per-model
isolation (fixed 2026-09-26). Imported by path since scripts/ isn't a
package, same pattern as test_run_milestone_eval_warmup.py.

Regression test for a real, confirmed bug: one model's WandbSink raising
during ``wandb.init()``'s resume-status check (a live GraphQL timeout,
confirmed in a real run's log) used to abort the whole ``for model_key in
...`` loop, silently skipping the mirror for every OTHER model too, even
ones that would have synced fine. Local results (summary.parquet,
*_results.parquet) were never affected -- only this optional W&B copy.
"""

import importlib.util
import sys
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from mas_deepr.config import Settings

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_milestone_eval.py"
_spec = importlib.util.spec_from_file_location(
    "run_milestone_eval_wandb_mirror", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
run_milestone_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_milestone_eval_wandb_mirror"] = run_milestone_eval
_spec.loader.exec_module(run_milestone_eval)


class _FakeSink:
    """Records which model it was constructed for; raises on __enter__ for
    one designated "flaky" model, mimicking a wandb.init() network failure."""

    entered: ClassVar[list[str]] = []
    flaky_model: ClassVar[str | None] = None

    def __init__(self, *, project: str, run_name: str, config: dict, disabled: bool):
        self.run_name = run_name

    def __enter__(self) -> "_FakeSink":
        if self.flaky_model and self.flaky_model in self.run_name:
            raise TimeoutError("context deadline exceeded")
        _FakeSink.entered.append(self.run_name)
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def log_summary(self, metrics: dict) -> None:
        pass

    def log_table(self, name: str, cols: list, rows: list) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_fake_sink() -> None:
    _FakeSink.entered = []
    _FakeSink.flaky_model = None


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        cache_db=tmp_path / "cache.sqlite3",
    )


def _summary_df(models: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model": models,
            "benchmark": ["frames"] * len(models),
            "score_mean": [0.5] * len(models),
            "cost_usd_mean": [0.01] * len(models),
            "latency_s_mean": [1.0] * len(models),
            "n_errors": [0] * len(models),
        }
    )


def test_one_models_failure_does_not_skip_the_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    monkeypatch.setattr(run_milestone_eval, "WandbSink", _FakeSink)
    _FakeSink.flaky_model = "gemini-2.5-pro"

    settings = _settings(tmp_path)
    df = _summary_df(["llama-3.1-8b-wandb", "gemini-2.5-pro", "deepseek-chat"])

    run_milestone_eval._mirror_to_wandb(df, settings, "baseline", disabled=False)

    # The flaky model's own mirror failed, but the other two still ran --
    # this is the actual bug: before the fix, only models processed BEFORE
    # the flaky one (in iteration order) would ever get entered.
    assert "eval-baseline-llama-3.1-8b-wandb" in _FakeSink.entered
    assert "eval-baseline-deepseek-chat" in _FakeSink.entered
    assert "eval-baseline-gemini-2.5-pro" not in _FakeSink.entered
    assert len(_FakeSink.entered) == 2


def test_no_failures_mirrors_every_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "fake")
    monkeypatch.setattr(run_milestone_eval, "WandbSink", _FakeSink)

    settings = _settings(tmp_path)
    df = _summary_df(["llama-3.1-8b-wandb", "gpt-4.1", "gemini-2.5-pro"])

    run_milestone_eval._mirror_to_wandb(df, settings, "baseline", disabled=False)

    assert len(_FakeSink.entered) == 3
