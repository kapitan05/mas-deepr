"""End-to-end pipeline smoke test: Manager -> Browser(*) -> Synthesizer wiring,
with fake duck-typed agents standing in for the real MAF ``Agent`` so this
runs with no network/LLM calls. Verifies orchestration order, sub-question
parsing, and that telemetry is recorded for every step.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from agent_framework import Agent

from mas_deepr.agents.topology import ResearchPipeline, run_pipeline
from mas_deepr.config.models import ModelSpec
from mas_deepr.telemetry import TelemetryTracker, read_telemetry

_SPEC = ModelSpec(key="fake-model", model_id="fake/model", family="qwen3")


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_details = {"input_token_count": 10, "output_token_count": 5}


class _FakeAgent:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []

    async def run(self, prompt: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(prompt)
        return _FakeResponse(next(self._responses))


@pytest.mark.asyncio
async def test_run_pipeline_orchestrates_manager_browser_synthesizer(
    tmp_path: Path,
) -> None:
    manager = _FakeAgent(["1. What is X?\n2. What is Y?"])
    browser = _FakeAgent(["Finding for X", "Finding for Y"])
    synthesizer = _FakeAgent(["Final answer combining X and Y"])

    tracker = TelemetryTracker(
        tmp_path / "telemetry.jsonl", run_id="smoke", phase="dev"
    )
    pipeline = ResearchPipeline(
        manager=cast(Agent, manager),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=tracker,
    )

    result = await run_pipeline(pipeline, "original question", question_id="q1")

    assert result.sub_questions == ["What is X?", "What is Y?"]
    assert result.findings == ["Finding for X", "Finding for Y"]
    assert result.final_answer == "Final answer combining X and Y"

    assert manager.calls == ["original question"]
    assert browser.calls == ["What is X?", "What is Y?"]
    assert len(synthesizer.calls) == 1
    assert "original question" in synthesizer.calls[0]
    assert "Finding for X" in synthesizer.calls[0]

    telemetry = read_telemetry(tmp_path / "telemetry.jsonl")
    assert telemetry.height == 4  # 1 manager + 2 browser + 1 synthesizer
    assert sorted(telemetry["role"].to_list()) == [
        "browser",
        "browser",
        "manager",
        "synthesizer",
    ]


@pytest.mark.asyncio
async def test_run_pipeline_single_sub_question(tmp_path: Path) -> None:
    manager = _FakeAgent(["1. Only question?"])
    browser = _FakeAgent(["Only finding"])
    synthesizer = _FakeAgent(["Answer"])
    tracker = TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="s2", phase="dev")
    pipeline = ResearchPipeline(
        manager=cast(Agent, manager),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=tracker,
    )

    result = await run_pipeline(pipeline, "q", question_id="q2")
    assert result.sub_questions == ["Only question?"]
    assert result.final_answer == "Answer"
