"""``scripts/run_milestone_eval.py::_warm_up`` -- imported by path since
scripts/ isn't a package.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_milestone_eval.py"
_spec = importlib.util.spec_from_file_location("run_milestone_eval", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
run_milestone_eval = importlib.util.module_from_spec(_spec)
sys.modules["run_milestone_eval"] = run_milestone_eval
_spec.loader.exec_module(run_milestone_eval)


class _FakeAgent:
    def __init__(self, *, raises: bool = False) -> None:
        self._raises = raises
        self.calls = 0

    async def run(self, prompt: str) -> Any:
        self.calls += 1
        if self._raises:
            raise RuntimeError("cold-start endpoint unreachable")
        return "OK"


class _FakePipeline:
    def __init__(self, *, raises: bool = False) -> None:
        self.manager = _FakeAgent(raises=raises)


@pytest.mark.asyncio
async def test_warm_up_calls_manager_once() -> None:
    pipeline = _FakePipeline()
    await run_milestone_eval._warm_up(pipeline)
    assert pipeline.manager.calls == 1


@pytest.mark.asyncio
async def test_warm_up_failure_does_not_raise() -> None:
    """Regression guard: a warm-up failure must not abort the real eval
    run -- the real failure should surface again on the first question."""
    pipeline = _FakePipeline(raises=True)
    await run_milestone_eval._warm_up(pipeline)
    assert pipeline.manager.calls == 1
