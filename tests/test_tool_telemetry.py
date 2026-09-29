"""Tool-call telemetry: ``TelemetryTracker.record_tool_call``/``tool_cost_for``/
``summarize_tool_calls``, and the ``record_tool_call`` helper's failure-marker
detection (see ``tools/tool_telemetry.py``)."""

from pathlib import Path

import pytest

from mas_deepr.telemetry import TelemetryTracker, summarize_tool_calls
from mas_deepr.tools.tool_telemetry import record_tool_call


def _tracker(tmp_path: Path) -> TelemetryTracker:
    return TelemetryTracker(tmp_path / "telemetry.jsonl", run_id="t", phase="dev")


def test_record_tool_call_writes_to_sibling_sink(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    tracker.record_tool_call(
        tool_name="web_search",
        provider="searxng",
        question_id="q1",
        latency_s=0.5,
    )
    assert tracker.tool_sink.exists()
    assert tracker.tool_sink.name == "tool_calls.jsonl"
    assert tracker.tool_sink.parent == tracker.sink.parent


def test_tool_cost_for_accumulates_per_question(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    tracker.record_tool_call(
        tool_name="tavily_search", provider="tavily", question_id="q1", latency_s=0.1
    )
    tracker.record_tool_call(
        tool_name="tavily_search", provider="tavily", question_id="q1", latency_s=0.1
    )
    tracker.record_tool_call(
        tool_name="web_search", provider="searxng", question_id="q1", latency_s=0.1
    )

    assert tracker.tool_cost_for("q1") == pytest.approx(0.01)  # 2 * $0.005
    assert tracker.tool_cost_for("unknown_question") == 0.0


def test_free_providers_record_zero_cost(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    for provider, tool_name in [
        ("searxng", "web_search"),
        ("crawl4ai", "fetch_page"),
        ("semantic_scholar", "semantic_scholar_search"),
        ("pubmed", "pubmed_search"),
    ]:
        tracker.record_tool_call(
            tool_name=tool_name, provider=provider, question_id="q1", latency_s=0.1
        )
    assert tracker.tool_cost_for("q1") == 0.0


def test_summarize_tool_calls_groups_by_provider(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)
    tracker.record_tool_call(
        tool_name="web_search", provider="searxng", question_id="q1", latency_s=0.2
    )
    tracker.record_tool_call(
        tool_name="web_search", provider="searxng", question_id="q2", latency_s=0.4
    )
    tracker.record_tool_call(
        tool_name="tavily_search",
        provider="tavily",
        question_id="q1",
        latency_s=0.1,
        error="[mcp_call_failed: timeout]",
    )

    df = summarize_tool_calls(tracker.tool_sink)
    rows = {r["provider"]: r for r in df.to_dicts()}

    assert rows["searxng"]["calls"] == 2
    assert rows["searxng"]["cost_usd"] == 0.0
    assert rows["searxng"]["errors"] == 0

    assert rows["tavily"]["calls"] == 1
    assert rows["tavily"]["cost_usd"] == pytest.approx(0.005)
    assert rows["tavily"]["errors"] == 1


@pytest.mark.asyncio
async def test_record_tool_call_helper_detects_failure_marker(
    tmp_path: Path,
) -> None:
    tracker = _tracker(tmp_path)

    async def failing_call() -> str:
        return "[mcp_call_failed: ConnectionError: refused]"

    result = await record_tool_call(
        tracker,
        tool_name="web_search",
        provider="searxng",
        question_id="q1",
        call=failing_call,
    )

    assert result.startswith("[mcp_call_failed:")
    df = summarize_tool_calls(tracker.tool_sink)
    row = df.to_dicts()[0]
    assert row["errors"] == 1


@pytest.mark.asyncio
async def test_record_tool_call_helper_success_has_no_error(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path)

    async def successful_call() -> list[dict[str, str]]:
        return [{"title": "A", "url": "http://a.com", "snippet": "s"}]

    result = await record_tool_call(
        tracker,
        tool_name="web_search",
        provider="searxng",
        question_id="q1",
        call=successful_call,
    )

    assert result == [{"title": "A", "url": "http://a.com", "snippet": "s"}]
    df = summarize_tool_calls(tracker.tool_sink)
    row = df.to_dicts()[0]
    assert row["errors"] == 0
