"""Tool-call guard (agents/tool_call_guard.py) + the real tool budget.

Uses a real MAF ``Agent`` with a fake model client (same layer order as
OpenAIChatCompletionClient in agent-framework 1.11), so these tests pin the
library behavior the fix depends on:

- the budget is enforced via the client's function_invocation_configuration;
- the budget-exhausted final call goes out with no tools at all;
- text-form tool calls (Llama bare JSON, Qwen <tool_call> tags) become real,
  executed, budget-counted calls -- and nothing else gets converted.
"""

from collections.abc import Callable
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
from mas_deepr.agents.parsing import has_leaked_tool_call, strip_leaked_tool_calls
from mas_deepr.agents.tool_call_guard import (
    NO_TOOLS_LEFT_INSTRUCTION,
    ToolCallGuardMiddleware,
    ToolGuardStats,
    current_tool_guard_stats,
    parse_text_tool_calls,
)
from mas_deepr.agents.topology import ResearchPipeline, run_pipeline
from mas_deepr.config import ModelSpec, Settings, get_model
from mas_deepr.telemetry import TelemetryTracker

# --- parse_text_tool_calls ----------------------------------------------------

_TOOLS = {"fetch_page", "wikipedia_search"}


def test_parses_llama_multi_call_reply() -> None:
    text = (
        '{"name": "fetch_page", "parameters": {"url": "https://a.org"}}; '
        '{"name": "wikipedia_search", "parameters": {"query": "Jane Eyre"}}'
    )
    assert parse_text_tool_calls(text, _TOOLS) == [
        ("fetch_page", {"url": "https://a.org"}),
        ("wikipedia_search", {"query": "Jane Eyre"}),
    ]


def test_parses_python_tag_type_function_and_string_arguments() -> None:
    text = (
        '<|python_tag|>{"type": "function", "name": "fetch_page", '
        '"arguments": "{\\"url\\": \\"https://a.org\\"}"}'
    )
    assert parse_text_tool_calls(text, _TOOLS) == [
        ("fetch_page", {"url": "https://a.org"})
    ]


def test_parses_qwen_tags_with_surrounding_prose() -> None:
    text = (
        "Let me check one more page.\n<tool_call>\n"
        '{"name": "fetch_page", "arguments": {"url": "https://a.org"}}\n</tool_call>'
    )
    assert parse_text_tool_calls(text, _TOOLS) == [
        ("fetch_page", {"url": "https://a.org"})
    ]


@pytest.mark.parametrize(
    "text",
    [
        "X was founded in 1889. Source: https://x.org",  # an answer
        'Answer: 42. Data: {"name": "fetch_page", "parameters": {}}',  # prose + JSON
        '{"name": "delete_everything", "parameters": {}}',  # unknown tool
        '{"name": "fetch_page", "parameters": {}}; {"name": "nope", "parameters": {}}',
        '{"name": "fetch_page", "parameters": ',  # truncated
        "",
    ],
)
def test_does_not_convert_non_calls(text: str) -> None:
    assert parse_text_tool_calls(text, _TOOLS) is None


# --- C1 safety net: Llama format detection/stripping -------------------------


def test_leak_detection_covers_llama_json_and_strips_it() -> None:
    leaked = (
        'Searching now. {"name": "fetch_page", "parameters": {"url": "https://a"}}; '
        '{"name": "wikipedia_search", "parameters": {"query": "x"}}'
    )
    assert has_leaked_tool_call(leaked)
    assert strip_leaked_tool_calls(leaked) == "Searching now."
    assert not has_leaked_tool_call('The config was {"a": 1}.')


# --- Real Agent + fake model ---------------------------------------------------


class _ScriptedRaw(BaseChatClient):
    """Replies via ``script(n, options)``; records every request it gets."""

    def __init__(self, script: Callable[[int, Any], Message], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.script = script
        self.requests: list[tuple[list[Message], dict[str, Any]]] = []

    async def _inner_get_response(
        self, *, messages: Any, stream: bool, options: Any, **kwargs: Any
    ) -> ChatResponse:
        self.requests.append((list(messages), dict(options or {})))
        return ChatResponse(messages=[self.script(len(self.requests), options)])


class _FakeClient(FunctionInvocationLayer, ChatMiddlewareLayer, _ScriptedRaw):
    """Same layer order as OpenAIChatCompletionClient (agent-framework 1.11)."""


executions: list[str] = []


@tool
def fetch_page(url: str) -> str:
    """Fetch a page."""
    executions.append(url)
    return f"content of {url}"


def _real_call(n: int) -> Message:
    return Message(
        role="assistant",
        contents=[
            Content.from_function_call(
                call_id=f"c{n}", name="fetch_page", arguments={"url": f"https://p{n}"}
            )
        ],
    )


def _text(text: str, raw: Any = None) -> Message:
    return Message(
        role="assistant", contents=[Content.from_text(text, raw_representation=raw)]
    )


async def _run(
    client: _FakeClient, *, single: bool = False
) -> tuple[Any, ToolGuardStats]:
    executions.clear()
    agent = Agent(
        client,
        name="browser",
        instructions="SYS",
        tools=[fetch_page],
        middleware=[ToolCallGuardMiddleware(single_tool_call_per_turn=single)],
    )
    stats = ToolGuardStats()
    token = current_tool_guard_stats.set(stats)
    try:
        resp = await agent.run("sub-question")
    finally:
        current_tool_guard_stats.reset(token)
    return resp, stats


@pytest.mark.asyncio
async def test_budget_enforced_and_final_call_has_no_tools() -> None:
    # A model that would call tools forever, unless no tools are offered.
    def script(n: int, options: Any) -> Message:
        if options.get("tools"):
            return _real_call(n)
        return _text("Final answer from evidence.")

    client = _FakeClient(script=script)
    client.function_invocation_configuration["max_function_calls"] = 2
    resp, stats = await _run(client)

    assert len(executions) == 2  # the budget really stops tool execution
    final_messages, final_options = client.requests[-1]
    assert "tools" not in final_options and "tool_choice" not in final_options
    assert final_messages[-1].text == NO_TOOLS_LEFT_INSTRUCTION
    assert resp.text == "Final answer from evidence."
    assert stats.final_turns_without_tools == 1
    # The nudge is request-only: MAF's returned history doesn't contain it.
    assert all(m.text != NO_TOOLS_LEFT_INSTRUCTION for m in resp.messages)


@pytest.mark.asyncio
async def test_llama_text_calls_are_executed_and_counted() -> None:
    raw = object()  # stands in for the logprob-bearing OpenAI Choice

    def script(n: int, options: Any) -> Message:
        if n == 1:
            return _text(
                '{"name": "fetch_page", "parameters": {"url": "https://a"}}; '
                '{"name": "fetch_page", "parameters": {"url": "https://b"}}',
                raw=raw,
            )
        return _text("Answer based on a and b.")

    client = _FakeClient(script=script)
    client.function_invocation_configuration["max_function_calls"] = 8
    resp, stats = await _run(client)

    assert executions == ["https://a", "https://b"]
    assert resp.text == "Answer based on a and b."
    assert stats.text_tool_calls_converted == 2 and stats.replies_converted == 1
    calls = [c for m in resp.messages for c in m.contents if c.type == "function_call"]
    assert [c.name for c in calls] == ["fetch_page", "fetch_page"]
    assert calls[0].raw_representation is raw  # GRPO keeps the real tokens


@pytest.mark.asyncio
async def test_converted_calls_count_against_the_budget() -> None:
    def script(n: int, options: Any) -> Message:
        if options.get("tools"):
            return _text(f'{{"name": "fetch_page", "parameters": {{"url": "u{n}"}}}}')
        return _text("done")

    client = _FakeClient(script=script)
    client.function_invocation_configuration["max_function_calls"] = 3
    resp, stats = await _run(client)

    assert len(executions) == 3
    assert stats.final_turns_without_tools == 1
    assert resp.text == "done"


@pytest.mark.asyncio
async def test_plain_answer_mentioning_a_tool_is_left_alone() -> None:
    client = _FakeClient(script=lambda n, o: _text("I used fetch_page; X is 42."))
    resp, stats = await _run(client)
    assert executions == []
    assert resp.text == "I used fetch_page; X is 42."
    assert stats.replies_converted == 0


# --- One call per turn (Llama 3.1 chat template) ------------------------------


def _max_calls_per_turn(messages: list[Message]) -> int:
    return max(
        (sum(1 for c in m.contents if c.type == "function_call") for m in messages),
        default=0,
    )


def _native_multi_call(n: int) -> Message:
    return Message(
        role="assistant",
        contents=[
            Content.from_function_call(
                call_id=f"c{n}{i}", name="fetch_page", arguments={"url": f"u{i}"}
            )
            for i in range(3)
        ],
    )


@pytest.mark.asyncio
async def test_single_call_model_runs_only_first_text_call() -> None:
    def script(n: int, options: Any) -> Message:
        if n == 1:
            return _text(
                '{"name": "fetch_page", "parameters": {"url": "https://a"}}; '
                '{"name": "fetch_page", "parameters": {"url": "https://b"}}; '
                '{"name": "fetch_page", "parameters": {"url": "https://c"}}'
            )
        return _text("done")

    client = _FakeClient(script=script)
    resp, stats = await _run(client, single=True)

    assert executions == ["https://a"]
    assert stats.text_tool_calls_converted == 1
    # The request after the call -- the one that 400'd on W&B -- now carries
    # a history the Llama 3.1 template can render.
    assert _max_calls_per_turn(client.requests[1][0]) == 1
    assert resp.text == "done"


@pytest.mark.asyncio
async def test_single_call_model_trims_native_multi_call_reply() -> None:
    client = _FakeClient(
        script=lambda n, o: _native_multi_call(n) if n == 1 else _text("done")
    )
    resp, _stats = await _run(client, single=True)
    assert executions == ["u0"]
    assert _max_calls_per_turn(client.requests[1][0]) == 1
    assert resp.text == "done"


@pytest.mark.asyncio
async def test_multi_call_models_keep_all_calls() -> None:
    client = _FakeClient(
        script=lambda n, o: _native_multi_call(n) if n == 1 else _text("done")
    )
    await _run(client, single=False)
    assert executions == ["u0", "u1", "u2"]


# --- Pipeline wiring ------------------------------------------------------------


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("MAS_SLM_BASE_URL", raising=False)
    return Settings()


def _has(agent: Agent, cls: type) -> bool:
    return any(isinstance(m, cls) for m in (agent.middleware or []))


def _guard(agent: Agent) -> ToolCallGuardMiddleware:
    return next(
        m for m in (agent.middleware or []) if isinstance(m, ToolCallGuardMiddleware)
    )


def test_build_pipeline_sets_single_call_flag_from_spec(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wandb-x")
    tracker = TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="test")
    llama = build_pipeline(
        spec=get_model("llama-3.1-8b-wandb"), settings=settings, tracker=tracker
    )
    other = build_pipeline(
        spec=get_model("gpt-4.1"), settings=settings, tracker=tracker
    )
    assert _guard(llama.browser).single_tool_call_per_turn
    assert not _guard(other.browser).single_tool_call_per_turn


def test_build_pipeline_browser_has_own_limited_client_and_guard(
    settings: Settings, tmp_path: Path
) -> None:
    pipeline = build_pipeline(
        spec=get_model("gpt-4.1"),
        settings=settings,
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="test"),
    )
    browser_client = cast(Any, pipeline.browser).client
    manager_client = cast(Any, pipeline.manager).client
    assert browser_client is not manager_client
    assert (
        browser_client.function_invocation_configuration["max_function_calls"]
        == settings.max_tool_calls_per_query
    )
    assert (
        manager_client.function_invocation_configuration["max_function_calls"] is None
    )
    assert _has(pipeline.browser, ToolCallGuardMiddleware)
    assert _has(pipeline.synthesizer, ToolCallGuardMiddleware)
    assert not _has(pipeline.manager, ToolCallGuardMiddleware)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_details = {"input_token_count": 1, "output_token_count": 1}


class _ClientStub:
    def __init__(self) -> None:
        self.function_invocation_configuration: dict[str, Any] = {
            "max_function_calls": None
        }


class _FakeAgent:
    def __init__(self, responses: list[str], client: Any = None) -> None:
        self._responses = iter(responses)
        self.client = client
        self.kwargs: list[dict[str, Any]] = []

    async def run(self, prompt: str, **kwargs: Any) -> _FakeResponse:
        self.kwargs.append(kwargs)
        return _FakeResponse(next(self._responses))


@pytest.mark.asyncio
async def test_run_pipeline_applies_budget_to_browser_client(tmp_path: Path) -> None:
    stub = _ClientStub()
    browser = _FakeAgent(["finding"], client=stub)
    pipeline = ResearchPipeline(
        manager=cast(Agent, _FakeAgent(["1. A?"])),
        browser=cast(Agent, browser),
        synthesizer=cast(Agent, _FakeAgent(["final"])),
        spec=ModelSpec(key="f", model_id="f", family="qwen3"),
        tracker=TelemetryTracker(tmp_path / "t.jsonl", run_id="t", phase="dev"),
    )
    result = await run_pipeline(
        pipeline, "q", question_id="tg", max_tool_calls_per_query=3
    )
    assert stub.function_invocation_configuration["max_function_calls"] == 3
    # The old, ineffective kwarg is gone.
    assert "function_invocation_kwargs" not in browser.kwargs[0]
    assert result.tool_guard_stats is not None
