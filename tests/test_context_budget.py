"""B1 token-budget guard (agents/context_budget.py) + its run_pipeline caps.

Includes the MAF contract test: the guard must shorten what a request
*sends* while ``resp.messages`` keeps the full tool results ("new Message
objects, never edit in place" -- see the module docstring). That behavior
comes from agent-framework internals (ChatContext gets a shallow copy of
the loop's message list), so this test is what catches an MAF upgrade or a
refactor that silently breaks it.
"""

from pathlib import Path
from typing import Any, cast

import pytest
from agent_framework import (
    Agent,
    BaseChatClient,
    ChatResponse,
    Content,
    Message,
    tool,
)
from agent_framework._middleware import ChatMiddlewareLayer
from agent_framework._tools import FunctionInvocationLayer

from mas_deepr.agents import build_pipeline
from mas_deepr.agents.context_budget import (
    CHARS_PER_TOKEN,
    ContextBudget,
    ContextBudgetMiddleware,
    ContextBudgetStats,
    current_budget_stats,
    estimate_request_tokens,
    trim_messages,
    truncate_middle,
)
from mas_deepr.agents.topology import ResearchPipeline, run_pipeline
from mas_deepr.config import ModelSpec, Settings, get_model
from mas_deepr.memory import ContextFoldingStrategy
from mas_deepr.telemetry import TelemetryTracker

_SPEC = ModelSpec(key="fake-model", model_id="fake/model", family="qwen3")


# --- truncate_middle ---------------------------------------------------------


def test_truncate_middle_leaves_short_text_alone() -> None:
    assert truncate_middle("short", 100) == "short"


def test_truncate_middle_keeps_head_and_tail_within_limit() -> None:
    text = "HEAD" + "x" * 5000 + "TAIL https://source.org"
    out = truncate_middle(text, 1000)
    assert len(out) <= 1000
    assert out.startswith("HEAD")
    assert out.endswith("https://source.org")
    assert "chars truncated" in out


# --- estimate / trim_messages ------------------------------------------------


def _call(call_id: str) -> Message:
    return Message(
        role="assistant",
        contents=[
            Content.from_function_call(call_id=call_id, name="fetch", arguments={})
        ],
    )


def _result(call_id: str, text: str) -> Message:
    return Message(
        role="tool", contents=[Content.from_function_result(call_id, result=text)]
    )


def _history(*results: str) -> list[Message]:
    msgs: list[Message] = [Message(role="user", contents=["sub-question"])]
    for i, r in enumerate(results):
        msgs += [_call(f"c{i}"), _result(f"c{i}", r)]
    return msgs


def _result_texts(messages: list[Any]) -> list[str]:
    return [
        str(c.result)
        for m in messages
        for c in m.contents
        if c.type == "function_result"
    ]


def test_estimate_counts_instructions_and_tools_from_options() -> None:
    msgs = [Message(role="user", contents=["hi"])]
    bare = estimate_request_tokens(msgs, {})
    with_sys = estimate_request_tokens(msgs, {"instructions": "s" * 3000})
    assert with_sys - bare == 3000 // CHARS_PER_TOKEN


def test_trim_is_noop_under_budget_and_returns_same_objects() -> None:
    msgs = _history("a" * 100, "b" * 100)
    out, report = trim_messages(msgs, {}, ContextBudget())
    assert report.trimmed_results == 0
    assert all(o is m for o, m in zip(out, msgs, strict=True))


def test_trim_shortens_oldest_first_and_spares_newest() -> None:
    old, mid, new = "o" * 6000, "m" * 6000, "n" * 6000
    msgs = _history(old, mid, new)
    # Budget leaves room for roughly two full results.
    budget = ContextBudget(window_tokens=4300, headroom_tokens=0)
    out, report = trim_messages(msgs, {}, budget)

    texts = _result_texts(out)
    assert len(texts[0]) < len(old)  # oldest cut
    assert texts[2] == new  # newest untouched
    assert not report.over_budget
    assert report.estimated_tokens_after <= budget.budget_tokens
    # Pairs intact: same number of messages, same call ids, same order.
    assert [m.role for m in out] == [m.role for m in msgs]
    assert [c.call_id for m in out for c in m.contents if c.call_id] == [
        c.call_id for m in msgs for c in m.contents if c.call_id
    ]


def test_trim_never_mutates_the_input_messages() -> None:
    msgs = _history("o" * 6000, "n" * 6000)
    out, report = trim_messages(
        msgs, {}, ContextBudget(window_tokens=2500, headroom_tokens=0)
    )
    assert report.trimmed_results >= 1
    assert _result_texts(msgs) == ["o" * 6000, "n" * 6000]
    assert out[2] is not msgs[2]  # shortened -> a new Message object
    assert out[0] is msgs[0]  # untouched -> passed through as-is


def test_trim_flags_over_budget_when_floor_reached() -> None:
    msgs = _history("o" * 6000)
    _out, report = trim_messages(
        msgs,
        {"instructions": "s" * 30000},  # system prompt alone is over budget
        ContextBudget(window_tokens=5000, headroom_tokens=0),
    )
    assert report.over_budget


# --- MAF contract test (real Agent + tool loop, fake model) ------------------


class _RawFakeClient(BaseChatClient):
    """Calls the tool twice, then answers; records every request it gets."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.requests: list[list[Message]] = []

    async def _inner_get_response(
        self, *, messages: Any, stream: bool, options: Any, **kwargs: Any
    ) -> ChatResponse:
        self.requests.append(list(messages))
        n = len(self.requests)
        if n <= 2:
            return ChatResponse(
                messages=[
                    Message(
                        role="assistant",
                        contents=[
                            Content.from_function_call(
                                call_id=f"call{n}", name="big_page", arguments={}
                            )
                        ],
                    )
                ]
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


class _FakeClient(FunctionInvocationLayer, ChatMiddlewareLayer, _RawFakeClient):
    """Same layer order as OpenAIChatCompletionClient (agent-framework 1.11)."""


_PAGE = "PAGE-START " + "z" * 9000 + " PAGE-END"


@tool
def big_page() -> str:
    """Return a large page."""
    return _PAGE


@pytest.mark.asyncio
async def test_maf_guard_trims_request_but_resp_messages_keep_full_results() -> None:
    client = _FakeClient()
    # ~1.5 pages fit: the 3rd request (two results) must be trimmed.
    budget = ContextBudget(window_tokens=4600, headroom_tokens=0)
    agent = Agent(
        client,
        instructions="SYS",
        name="browser",
        tools=[big_page],
        middleware=[ContextBudgetMiddleware(budget)],
    )
    stats = ContextBudgetStats()
    token = current_budget_stats.set(stats)
    try:
        resp = await agent.run("sub-question")
    finally:
        current_budget_stats.reset(token)

    assert len(client.requests) == 3  # middleware ran on every loop step
    sent = _result_texts(client.requests[2])
    assert len(sent) == 2
    assert len(sent[0]) < len(_PAGE)  # oldest result shortened in the request
    assert sent[1] == _PAGE  # newest kept whole

    returned = _result_texts(list(resp.messages))
    assert returned == [_PAGE, _PAGE]  # history MAF returns is untouched

    assert stats.calls_checked == 3
    assert stats.trimmed_calls >= 1
    assert stats.chars_trimmed > 0


# --- run_pipeline caps --------------------------------------------------------


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


def _pipeline(
    tmp_path: Path, browser_replies: list[str], budget: ContextBudget | None
) -> tuple[ResearchPipeline, _FakeAgent, _FakeAgent]:
    browser = _FakeAgent(browser_replies)
    synthesizer = _FakeAgent(["Final"])
    pipeline = ResearchPipeline(
        manager=cast(Agent, _FakeAgent(["1. A?\n2. B?"])),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, synthesizer),
        spec=_SPEC,
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="b1", phase="dev"),
        context_budget=budget,
    )
    return pipeline, browser, synthesizer


@pytest.mark.asyncio
async def test_findings_capped_for_synth_but_raw_findings_kept(tmp_path: Path) -> None:
    long_finding = "F" * 20000 + " https://source.org"
    budget = ContextBudget(finding_max_tokens=1000)  # 3000 chars
    pipeline, _b, synthesizer = _pipeline(tmp_path, [long_finding, "short"], budget)

    result = await run_pipeline(pipeline, "q", question_id="b1a")

    assert result.raw_findings[0] == long_finding
    assert len(result.findings[0]) <= 3000
    assert result.findings[0].endswith("https://source.org")
    assert result.findings[1] == "short"
    assert long_finding not in synthesizer.calls[0]
    assert result.context_stats is not None
    assert result.context_stats.findings_capped == 1


@pytest.mark.asyncio
async def test_folding_prefix_capped(tmp_path: Path) -> None:
    budget = ContextBudget(finding_max_tokens=3000, prefix_max_tokens=500)
    pipeline, browser, _s = _pipeline(tmp_path, ["G" * 8000, "short"], budget)

    result = await run_pipeline(
        pipeline, "q", question_id="b1b", memory=ContextFoldingStrategy()
    )

    # Second Browser call carries the folded sibling finding as a prefix.
    assert len(browser.calls[1]) <= 500 * CHARS_PER_TOKEN + len("\n\nB?")
    assert result.context_stats is not None
    assert result.context_stats.prefixes_capped == 1


@pytest.mark.asyncio
async def test_guard_off_leaves_pipeline_unchanged(tmp_path: Path) -> None:
    long_finding = "F" * 20000
    pipeline, _b, synthesizer = _pipeline(tmp_path, [long_finding, "short"], None)

    result = await run_pipeline(pipeline, "q", question_id="b1c")

    assert result.context_stats is None
    assert result.findings[0] == long_finding
    assert long_finding in synthesizer.calls[0]


# --- build_pipeline wiring ---------------------------------------------------


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("MAS_SLM_BASE_URL", raising=False)
    return Settings()


def _has_guard(agent: Agent) -> bool:
    return any(isinstance(m, ContextBudgetMiddleware) for m in (agent.middleware or []))


def test_build_pipeline_attaches_guard_by_default(
    settings: Settings, tmp_path: Path
) -> None:
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"),
        settings=settings,
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="test"),
    )
    assert pipeline.context_budget is not None
    assert pipeline.context_budget.window_tokens == 32768
    assert _has_guard(pipeline.browser)
    assert _has_guard(pipeline.synthesizer)
    assert pipeline.browser_recovery is not None
    assert _has_guard(pipeline.browser_recovery)


def test_build_pipeline_guard_off(settings: Settings, tmp_path: Path) -> None:
    settings.context_guard = False
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"),
        settings=settings,
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="test"),
    )
    assert pipeline.context_budget is None
    assert not _has_guard(pipeline.browser)


def test_model_spec_window_overrides_settings(
    settings: Settings, tmp_path: Path
) -> None:
    spec = get_model("gpt-4.1").model_copy(update={"context_window_tokens": 128000})
    pipeline = build_pipeline(
        spec=spec,
        settings=settings,
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="test"),
    )
    assert pipeline.context_budget is not None
    assert pipeline.context_budget.window_tokens == 128000
