"""Memory-strategy regression + behavior tests.

The core contract: ``run_pipeline(..., memory=None)`` and
``run_pipeline(..., memory=StatelessStrategy())`` must produce identical
``PipelineResult`` output and identical telemetry call counts -- this is
what makes the whole memory module a safe, additive abstraction over the
pre-Phase-3 pipeline.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from agent_framework import Agent

from mas_deepr.agents.topology import ResearchPipeline, run_pipeline
from mas_deepr.config.models import ModelSpec
from mas_deepr.memory import (
    ContextFoldingStrategy,
    StatelessStrategy,
    TrajectoryCompressionStrategy,
    get_memory_strategy,
)
from mas_deepr.telemetry import TelemetryTracker, read_telemetry

_SPEC = ModelSpec(key="fake-model", model_id="fake/model", family="qwen3")


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_details = {"input_token_count": 10, "output_token_count": 5}
        self.messages: list[object] = []


class _FakeAgent:
    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []

    async def run(self, prompt: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(prompt)
        return _FakeResponse(next(self._responses))


def _pipeline(
    tmp_path: Path,
    *,
    manager: list[str],
    browser: list[str],
    synthesizer: list[str],
    compressor: list[str] | None = None,
    run_id: str = "mem",
) -> ResearchPipeline:
    tracker = TelemetryTracker(tmp_path / "telemetry.jsonl", run_id=run_id, phase="dev")
    return ResearchPipeline(
        manager=cast(Agent, _FakeAgent(manager)),
        browser=cast(Agent, _FakeAgent(browser)),
        synthesizer=cast(Agent, _FakeAgent(synthesizer)),
        spec=_SPEC,
        tracker=tracker,
        compressor=cast(Agent, _FakeAgent(compressor)) if compressor else None,
    )


@pytest.mark.asyncio
async def test_memory_none_and_stateless_are_byte_identical(tmp_path: Path) -> None:
    def build(run_id: str) -> ResearchPipeline:
        return _pipeline(
            tmp_path,
            manager=["1. What is X?\n2. What is Y?"],
            browser=["Finding for X", "Finding for Y"],
            synthesizer=["Final answer"],
            run_id=run_id,
        )

    pipeline_none = build("none")
    result_none = await run_pipeline(pipeline_none, "q", question_id="q1", memory=None)

    pipeline_stateless = build("stateless")
    result_stateless = await run_pipeline(
        pipeline_stateless, "q", question_id="q1", memory=StatelessStrategy()
    )

    assert result_none.sub_questions == result_stateless.sub_questions
    assert result_none.findings == result_stateless.findings
    assert result_none.final_answer == result_stateless.final_answer
    assert result_none.memory_strategy == "stateless"
    assert result_stateless.memory_strategy == "stateless"

    telemetry_none = read_telemetry(tmp_path / "telemetry.jsonl").filter(
        read_telemetry(tmp_path / "telemetry.jsonl")["run_id"] == "none"
    )
    telemetry_stateless = read_telemetry(tmp_path / "telemetry.jsonl").filter(
        read_telemetry(tmp_path / "telemetry.jsonl")["run_id"] == "stateless"
    )
    assert telemetry_none.height == telemetry_stateless.height == 4


@pytest.mark.asyncio
async def test_manager_direct_fast_path_skips_browser_and_synthesizer(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        manager=["DIRECT: 42"],
        browser=[],
        synthesizer=[],
    )
    result = await run_pipeline(pipeline, "what is 6*7?", question_id="q1")

    assert result.final_answer == "42"
    assert result.sub_questions == []
    telemetry = read_telemetry(tmp_path / "telemetry.jsonl")
    assert telemetry.height == 1
    assert telemetry["role"][0] == "manager"


@pytest.mark.asyncio
async def test_folding_shares_sibling_findings_with_later_branches(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        manager=["1. What is X?\n2. What is Y?"],
        browser=["Finding for X", "Finding for Y"],
        synthesizer=["Final"],
    )
    await run_pipeline(pipeline, "q", question_id="q1", memory=ContextFoldingStrategy())

    browser_agent = cast(_FakeAgent, pipeline.browser)
    # Second branch's prompt should include the first branch's folded finding.
    assert "Finding for X" in browser_agent.calls[1]
    assert "What is X?" in browser_agent.calls[1]
    # First branch has no prior siblings, so no folded context is injected.
    assert browser_agent.calls[0] == "What is X?"


@pytest.mark.asyncio
async def test_retrac_compresses_pass_and_feeds_hypothesis_to_next_pass(
    tmp_path: Path,
) -> None:
    pipeline = _pipeline(
        tmp_path,
        manager=["1. First sub-question?", "1. Second sub-question?"],
        browser=["Finding 1", "Finding 2"],
        synthesizer=["Answer pass 1", "Answer pass 2"],
        compressor=[
            '{"answer": "tentative", "evidence": ["e1"], "analysis": "a", '
            '"uncertainties": [], "failed_attempts": []}',
            '{"answer": "final", "evidence": [], "analysis": "", '
            '"uncertainties": [], "failed_attempts": []}',
        ],
    )
    strategy = TrajectoryCompressionStrategy(max_passes=2)
    result = await run_pipeline(pipeline, "q", question_id="q1", memory=strategy)

    assert result.memory_strategy == "retrac"
    assert len(result.pass_records) == 2
    assert result.pass_records[0].state is not None
    assert result.pass_records[0].state.answer == "tentative"

    manager_agent = cast(_FakeAgent, pipeline.manager)
    # Second pass's manager prompt should carry the first pass's hypothesis.
    assert "Prior-pass hypothesis" in manager_agent.calls[1]
    assert "tentative" in manager_agent.calls[1]


def test_get_memory_strategy_unknown_key_raises_with_known_keys() -> None:
    with pytest.raises(KeyError, match="stateless"):
        get_memory_strategy("nonexistent")


def test_get_memory_strategy_builds_registered_strategies() -> None:
    assert get_memory_strategy("stateless").name == "stateless"
    assert get_memory_strategy("folding").name == "folding"
    assert get_memory_strategy("retrac").name == "retrac"
