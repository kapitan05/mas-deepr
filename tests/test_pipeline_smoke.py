"""End-to-end pipeline smoke test: Manager -> Browser(*) -> Synthesizer wiring,
with fake duck-typed agents standing in for the real MAF ``Agent`` so this
runs with no network/LLM calls. Verifies orchestration order, sub-question
parsing, and that telemetry is recorded for every step.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from agent_framework import Agent

from mas_deepr.agents.parsing import LEAK_FALLBACK_FINDING
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


class _FakeMcpClient:
    """Duck-typed stand-in for MCPToolClient's async-context-manager
    surface -- records enter/exit calls, doesn't touch a real connection."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self) -> "_FakeMcpClient":
        self.enter_count += 1
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.exit_count += 1


@pytest.mark.asyncio
async def test_run_pipeline_holds_mcp_client_open_for_the_whole_run(
    tmp_path: Path,
) -> None:
    """Regression test: run_pipeline used to have no handle on the
    pipeline's MCPToolClient at all, so every tool call reconnected --
    confirms it now opens the connection once for the whole run, not once
    per tool call (there'd be 2 browser calls here if it were per-call)."""
    manager = _FakeAgent(["1. What is X?\n2. What is Y?"])
    browser = _FakeAgent(["Finding for X", "Finding for Y"])
    synthesizer = _FakeAgent(["Final answer combining X and Y"])
    tracker = TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="mcp", phase="dev")
    fake_mcp = _FakeMcpClient()
    pipeline = ResearchPipeline(
        manager=cast(Agent, manager),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=tracker,
        mcp_client=cast(Any, fake_mcp),
    )

    await run_pipeline(pipeline, "original question", question_id="q3")

    assert fake_mcp.enter_count == 1
    assert fake_mcp.exit_count == 1


@pytest.mark.asyncio
async def test_run_pipeline_with_no_mcp_client_is_unaffected(tmp_path: Path) -> None:
    """mcp_client=None (every other test in this file) must keep working
    exactly as before -- no AttributeError, no accidental connection."""
    manager = _FakeAgent(["1. Only question?"])
    browser = _FakeAgent(["Only finding"])
    synthesizer = _FakeAgent(["Answer"])
    tracker = TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="s3", phase="dev")
    pipeline = ResearchPipeline(
        manager=cast(Agent, manager),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=tracker,
    )

    result = await run_pipeline(pipeline, "q", question_id="q4")
    assert result.final_answer == "Answer"


# --- Leaked tool-call text (C1) ---------------------------------------------

_LEAK = '<tool_call>\n{"name": "fetch_page", "arguments": {"url": "https://x.org"}}\n</tool_call>'


class _FakeResultContent:
    """Duck-typed MAF ``function_result`` content (what a tool returned)."""

    def __init__(self, result: str) -> None:
        self.type = "function_result"
        self.result = result


class _FakeToolMessage:
    def __init__(self, result: str) -> None:
        self.role = "tool"
        self.text = ""
        self.contents = [_FakeResultContent(result)]


class _FakeResponseWithMessages(_FakeResponse):
    def __init__(self, text: str, messages: list[object]) -> None:
        super().__init__(text)
        self.messages = messages


class _ScriptedAgent:
    """Like _FakeAgent, but each scripted reply can carry messages."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = iter(responses)
        self.calls: list[str] = []

    async def run(self, prompt: str, **kwargs: Any) -> _FakeResponse:
        self.calls.append(prompt)
        return next(self._responses)


def _leak_pipeline(
    tmp_path: Path,
    *,
    browser_replies: list[_FakeResponse],
    recovery_replies: list[str] | None,
    synth_reply: str = "Final answer",
) -> tuple[ResearchPipeline, _ScriptedAgent, _FakeAgent | None, _FakeAgent]:
    manager = _FakeAgent(["1. What is X?"])
    browser = _ScriptedAgent(browser_replies)
    recovery = _FakeAgent(recovery_replies) if recovery_replies is not None else None
    synthesizer = _FakeAgent([synth_reply])
    tracker = TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="c1", phase="dev")
    pipeline = ResearchPipeline(
        manager=cast(Agent, manager),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=tracker,
        browser_recovery=cast(Agent, recovery) if recovery is not None else None,
    )
    return pipeline, browser, recovery, synthesizer


@pytest.mark.asyncio
async def test_leaked_finding_triggers_one_tool_free_retry(tmp_path: Path) -> None:
    leaked = _FakeResponseWithMessages(
        f"Let me check one more page.\n{_LEAK}",
        [_FakeToolMessage("X was founded in 1889 (source: https://x.org).")],
    )
    pipeline, _browser, recovery, synthesizer = _leak_pipeline(
        tmp_path,
        browser_replies=[leaked],
        recovery_replies=["X was founded in 1889. Source: https://x.org"],
    )

    result = await run_pipeline(pipeline, "original question", question_id="c1a")

    assert recovery is not None
    assert len(recovery.calls) == 1
    # The retry prompt carries the sub-question and the evidence already
    # gathered (each agent.run() is a fresh conversation).
    assert "What is X?" in recovery.calls[0]
    assert "founded in 1889" in recovery.calls[0]
    assert "no tool calls left" in recovery.calls[0]

    assert result.findings == ["X was founded in 1889. Source: https://x.org"]
    assert "<tool_call" not in synthesizer.calls[0]
    assert result.leaked_tool_calls == 1
    assert result.leaks_recovered == 1

    telemetry = read_telemetry(tmp_path / "telemetry.jsonl")
    assert "browser_recovery" in telemetry["role"].to_list()


@pytest.mark.asyncio
async def test_clean_finding_triggers_no_retry(tmp_path: Path) -> None:
    pipeline, _browser, recovery, _synth = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse("X was founded in 1889.")],
        recovery_replies=[],
    )

    result = await run_pipeline(pipeline, "original question", question_id="c1b")

    assert recovery is not None
    assert recovery.calls == []
    assert result.findings == ["X was founded in 1889."]
    assert result.leaked_tool_calls == 0
    assert result.leaks_recovered == 0


@pytest.mark.asyncio
async def test_retry_that_leaks_again_falls_back_to_marker(tmp_path: Path) -> None:
    pipeline, _browser, recovery, synthesizer = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse(_LEAK)],
        recovery_replies=[_LEAK],
    )

    result = await run_pipeline(pipeline, "original question", question_id="c1c")

    assert recovery is not None
    assert len(recovery.calls) == 1  # exactly one retry, never a loop
    assert result.findings == [LEAK_FALLBACK_FINDING]
    assert LEAK_FALLBACK_FINDING in synthesizer.calls[0]
    assert "<tool_call" not in synthesizer.calls[0]
    assert result.leaked_tool_calls == 1
    assert result.leaks_recovered == 0


@pytest.mark.asyncio
async def test_leak_without_recovery_agent_keeps_cleaned_prose(
    tmp_path: Path,
) -> None:
    pipeline, _browser, _recovery, _synth = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse(f"X was founded in 1889.\n{_LEAK}")],
        recovery_replies=None,
    )

    result = await run_pipeline(pipeline, "original question", question_id="c1d")

    assert result.findings == ["X was founded in 1889."]
    assert result.leaked_tool_calls == 1


class _FailingAgent:
    async def run(self, prompt: str, **kwargs: Any) -> _FakeResponse:
        raise RuntimeError("endpoint down")


@pytest.mark.asyncio
async def test_failed_retry_does_not_fail_the_question(tmp_path: Path) -> None:
    pipeline, _browser, _recovery, _synth = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse(f"Partial answer.\n{_LEAK}")],
        recovery_replies=None,
    )
    pipeline.browser_recovery = cast(Agent, _FailingAgent())

    result = await run_pipeline(pipeline, "original question", question_id="c1e")

    assert result.findings == ["Partial answer."]
    assert result.final_answer == "Final answer"


@pytest.mark.asyncio
async def test_leaked_final_answer_is_cleaned(tmp_path: Path) -> None:
    pipeline, _browser, _recovery, _synth = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse("X was founded in 1889.")],
        recovery_replies=[],
        synth_reply=f"X was founded in 1889.\nFINAL ANSWER: 1889\n{_LEAK}",
    )

    result = await run_pipeline(pipeline, "original question", question_id="c1f")

    assert result.final_answer == "X was founded in 1889.\nFINAL ANSWER: 1889"
    assert result.leaked_tool_calls == 1


@pytest.mark.asyncio
async def test_role_turns_keep_original_leaked_text(tmp_path: Path) -> None:
    """GRPO trains on role_turns: the policy's real (leaked) output must be
    preserved there, with the retry as an extra turn -- only the findings
    passed downstream are cleaned."""
    pipeline, _browser, _recovery, _synth = _leak_pipeline(
        tmp_path,
        browser_replies=[_FakeResponse(_LEAK)],
        recovery_replies=["X was founded in 1889."],
    )

    result = await run_pipeline(
        pipeline, "original question", question_id="c1g", capture_logprobs=True
    )

    roles = [role for role, _turn in result.role_turns]
    assert roles == ["manager", "browser", "browser_recovery", "synthesizer"]
    browser_turn = result.role_turns[1][1]
    assert browser_turn.text == _LEAK
