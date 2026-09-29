"""Pure, infra-free tests for the ART rollout wrapper.

``rollout_fn``/``run_rollout_group`` need a live ART backend + served model
and are explicitly not exercised here (see rl/rollout.py's own docstring
for that status) -- this only covers the deterministic conversion logic.

Marked ``art`` (excluded from the default suite by ``addopts`` in
pyproject.toml) for a confirmed reason, not a hypothetical one: importing
``art`` monkeypatches ``transformers.masking_utils`` process-globally, which
breaks plain-``transformers`` forward passes (``test_cold_start.py``,
``test_dpo.py``) for the rest of the interpreter once triggered. Run
ART-touching tests in their own process: ``uv run pytest tests/ -m art``.
"""

import asyncio
import time

import pytest

pytestmark = pytest.mark.art


def test_build_trajectory_from_result_carries_reward_and_metadata() -> None:
    # Imported lazily, inside the test body: importing ``rl.rollout`` (and
    # therefore ``art``) must not happen at collection time, only when this
    # ``art``-marked test actually runs -- see the module docstring above.
    from mas_deepr.agents.topology import PipelineResult
    from mas_deepr.rl.rollout import build_trajectory_from_result

    result = PipelineResult(
        question_id="q1",
        question="What is the capital of France?",
        sub_questions=["Where is Paris?"],
        findings=["Paris is the capital of France."],
        final_answer="Paris",
        memory_strategy="folding",
    )
    trajectory = build_trajectory_from_result(
        question=result.question, result=result, reward=0.85
    )

    assert trajectory.reward == 0.85
    assert trajectory.metadata["question_id"] == "q1"
    assert trajectory.metadata["memory_strategy"] == "folding"
    assert trajectory.metadata["num_sub_questions"] == 1
    assert trajectory.metadata["final_answer"] == "Paris"
    assert trajectory.metrics["num_sub_questions"] == 1.0
    assert trajectory.metrics["num_raw_findings"] == 0.0
    contents = [
        m["content"]  # type: ignore[index]
        for m in trajectory.messages_and_choices
    ]
    assert result.question in contents
    assert result.final_answer in contents


class _FakeContent:
    """Duck-typed stand-in for a MAF ``Content`` item -- deliberately not
    importing ``agent_framework`` here, so this test doesn't couple to its
    exact internals, only to the ``type``/``raw_representation`` attributes
    ``choice_from_message`` reads via ``getattr``."""

    def __init__(self, type_: str, raw_representation: object = None) -> None:
        self.type = type_
        self.raw_representation = raw_representation


class _FakeMessage:
    """Duck-typed stand-in for a MAF ``Message``."""

    def __init__(self, role: str, text: str, contents: list[_FakeContent]) -> None:
        self.role = role
        self.text = text
        self.contents = contents


def _fake_choice(content: str) -> object:
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    return Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content=content),
    )


def test_build_trajectory_from_result_uses_real_role_turns_when_present() -> None:
    from mas_deepr.agents.topology import AgentTurn, PipelineResult
    from mas_deepr.rl.rollout import build_trajectory_from_result

    manager_choice = _fake_choice("1. Where is Paris?")
    manager_turn = AgentTurn(
        text="1. Where is Paris?",
        messages=[
            _FakeMessage("user", "What is the capital of France?", []),
            _FakeMessage(
                "assistant",
                "1. Where is Paris?",
                [_FakeContent("text", raw_representation=manager_choice)],
            ),
        ],
    )
    # A pure tool-call turn (no text content) -- no Choice to be found,
    # per choice_from_message's documented limitation.
    browser_turn = AgentTurn(
        text="",
        messages=[
            _FakeMessage("user", "Where is Paris?", []),
            _FakeMessage(
                "assistant",
                "",
                [_FakeContent("function_call", raw_representation=object())],
            ),
            _FakeMessage("tool", "Paris is in France.", []),
        ],
    )
    synth_choice = _fake_choice("Paris")
    synth_turn = AgentTurn(
        text="Paris",
        messages=[
            _FakeMessage("user", "synthesize", []),
            _FakeMessage(
                "assistant",
                "Paris",
                [_FakeContent("text", raw_representation=synth_choice)],
            ),
        ],
    )

    result = PipelineResult(
        question_id="q1",
        question="What is the capital of France?",
        sub_questions=["Where is Paris?"],
        final_answer="Paris",
        role_turns=[
            ("manager", manager_turn),
            ("browser", browser_turn),
            ("synthesizer", synth_turn),
        ],
    )

    trajectory = build_trajectory_from_result(
        question=result.question, result=result, reward=0.85
    )

    assert trajectory.reward == 0.85
    assert trajectory.metrics["num_role_turns"] == 3.0

    from openai.types.chat.chat_completion import Choice as OpenAIChoice

    real_choices = [
        m for m in trajectory.messages_and_choices if isinstance(m, OpenAIChoice)
    ]
    # Manager's and Synthesizer's text turns became real, trainable Choices;
    # the Browser's pure tool-call turn did not (no text content).
    assert real_choices == [manager_choice, synth_choice]

    # The Browser's tool-call turn is still present, just as untrained
    # plain-dict context, not silently dropped.
    plain_dicts = [
        m for m in trajectory.messages_and_choices if not isinstance(m, OpenAIChoice)
    ]
    assert any(d["content"] == "Paris is in France." for d in plain_dicts)


def test_build_trajectory_from_result_trains_pure_tool_call_when_choice_present() -> (
    None
):
    """With rl/trainable_client.py's TrainableOpenAIChatCompletionClient (a
    real pipeline built with trainable=True), a pure tool-call turn's
    function_call content carries a real Choice -- confirms
    choice_from_message picks it up and the turn becomes a trainable span,
    not untrained context like the sibling test above (which simulates
    MAF's stock client, no Choice on the function_call content)."""
    from mas_deepr.agents.topology import AgentTurn, PipelineResult
    from mas_deepr.rl.rollout import build_trajectory_from_result

    tool_call_choice = _fake_choice("")  # content empty -- pure tool call
    browser_turn = AgentTurn(
        text="",
        messages=[
            _FakeMessage("user", "Where is Paris?", []),
            _FakeMessage(
                "assistant",
                "",
                [_FakeContent("function_call", raw_representation=tool_call_choice)],
            ),
            _FakeMessage("tool", "Paris is in France.", []),
        ],
    )
    result = PipelineResult(
        question_id="q1",
        question="Where is Paris?",
        final_answer="Paris",
        role_turns=[("browser", browser_turn)],
    )
    trajectory = build_trajectory_from_result(
        question=result.question, result=result, reward=0.5
    )
    from openai.types.chat.chat_completion import Choice as OpenAIChoice

    real_choices = [
        m for m in trajectory.messages_and_choices if isinstance(m, OpenAIChoice)
    ]
    assert real_choices == [tool_call_choice]


class _RecordingPipeline:
    """Stands in for the real ``ResearchPipeline`` -- ``run_pipeline`` is
    monkeypatched below, so nothing ever inspects this beyond identity."""


async def _fake_run_pipeline(pipeline: object, prompt: str, **kwargs: object) -> object:
    from mas_deepr.agents.topology import PipelineResult

    return PipelineResult(
        question_id=str(kwargs.get("question_id", "")),
        question=prompt,
        final_answer="FINAL ANSWER: Paris",
    )


def _make_model(monkeypatch: pytest.MonkeyPatch) -> object:
    """A TrainableModel with get_step/get_inference_name faked out --
    both need a real registered backend on the actual class (confirmed
    live: calling either on an unregistered model raises "Model is not
    registered with the Backend"), which none of these tests set up.
    Patched at the class level since pydantic models reject arbitrary
    instance attribute assignment for non-field names.
    """
    import art

    async def _fake_get_step(self: object) -> int:
        return 0

    def _fake_get_inference_name(self: object, step: int | None = None) -> str:
        return "fake-inference-model"

    monkeypatch.setattr(art.TrainableModel, "get_step", _fake_get_step)
    monkeypatch.setattr(
        art.TrainableModel, "get_inference_name", _fake_get_inference_name
    )
    return art.TrainableModel(name="m", project="p", base_model="b")


def _fake_judge_and_tracker() -> tuple[object, object]:
    """rollout_fn's dispatch either never touches these (verifiable path,
    or the no-op tests below) or only passes them through to a
    monkeypatched scorer -- a bare object() stands in fine either way,
    cast past the real JudgeClient/TelemetryTracker types."""
    from typing import cast

    from mas_deepr.evals.judge import JudgeClient
    from mas_deepr.telemetry import TelemetryTracker

    return cast(JudgeClient, object()), cast(TelemetryTracker, object())


@pytest.mark.asyncio
async def test_rollout_fn_sets_tool_scope_on_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: rollout.py used to build its own Settings without
    ever calling get_tool_scope, silently falling through to the "general"
    class default and exposing GRPO rollouts to web_search/SearXNG at
    training volume."""
    from mas_deepr.config import Settings, get_tool_scope
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    captured: dict[str, Settings] = {}

    def fake_build_pipeline(
        *, spec: object, settings: Settings, **kwargs: object
    ) -> object:
        captured["settings"] = settings
        return _RecordingPipeline()

    monkeypatch.setattr(rollout_mod, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(rollout_mod, "run_pipeline", _fake_run_pipeline)

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1", source="musique", split="train", prompt="p", answer="Paris"
    )
    judge, tracker = _fake_judge_and_tracker()

    await rollout_mod.rollout_fn(
        model,  # type: ignore[arg-type]
        question,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
        tool_scope="general",
    )
    assert captured["settings"].mcp_enabled_tools == get_tool_scope("general")


@pytest.mark.asyncio
async def test_rollout_fn_defaults_to_wiki_paper_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.config import Settings, get_tool_scope
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    captured: dict[str, Settings] = {}

    def fake_build_pipeline(
        *, spec: object, settings: Settings, **kwargs: object
    ) -> object:
        captured["settings"] = settings
        return _RecordingPipeline()

    monkeypatch.setattr(rollout_mod, "build_pipeline", fake_build_pipeline)
    monkeypatch.setattr(rollout_mod, "run_pipeline", _fake_run_pipeline)

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1", source="musique", split="train", prompt="p", answer="Paris"
    )
    judge, tracker = _fake_judge_and_tracker()
    await rollout_mod.rollout_fn(
        model,  # type: ignore[arg-type]
        question,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
    )
    assert captured["settings"].mcp_enabled_tools == get_tool_scope("wiki_paper")


@pytest.mark.asyncio
async def test_rollout_fn_dispatches_rubric_question_to_judge_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.data.schema import Question, RubricCriterion
    from mas_deepr.rl import rollout as rollout_mod

    calls: dict[str, bool] = {"judge": False, "verifiable": False}

    async def fake_judge_path(**kwargs: object) -> list[object]:
        calls["judge"] = True
        return [type("O", (), {"score": 1.0})()]

    async def fake_verifiable_path(**kwargs: object) -> list[object]:
        calls["verifiable"] = True
        return [type("O", (), {"score": 1.0})()]

    monkeypatch.setattr(
        rollout_mod, "build_pipeline", lambda **kwargs: _RecordingPipeline()
    )
    monkeypatch.setattr(rollout_mod, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(rollout_mod, "score_rollout_group", fake_judge_path)
    monkeypatch.setattr(
        rollout_mod, "score_rollout_group_verifiable", fake_verifiable_path
    )

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1",
        source="research_rubrics",
        split="test",
        prompt="p",
        rubrics=[RubricCriterion(criterion="c", weight=1.0, axis="a")],
    )
    judge, tracker = _fake_judge_and_tracker()
    await rollout_mod.rollout_fn(
        model,  # type: ignore[arg-type]
        question,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
    )
    assert calls == {"judge": True, "verifiable": False}


@pytest.mark.asyncio
async def test_rollout_fn_dispatches_answer_question_to_verifiable_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    calls: dict[str, bool] = {"judge": False, "verifiable": False}

    async def fake_judge_path(**kwargs: object) -> list[object]:
        calls["judge"] = True
        return [type("O", (), {"score": 1.0})()]

    async def fake_verifiable_path(**kwargs: object) -> list[object]:
        calls["verifiable"] = True
        return [type("O", (), {"score": 1.0})()]

    monkeypatch.setattr(
        rollout_mod, "build_pipeline", lambda **kwargs: _RecordingPipeline()
    )
    monkeypatch.setattr(rollout_mod, "run_pipeline", _fake_run_pipeline)
    monkeypatch.setattr(rollout_mod, "score_rollout_group", fake_judge_path)
    monkeypatch.setattr(
        rollout_mod, "score_rollout_group_verifiable", fake_verifiable_path
    )

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1", source="musique", split="train", prompt="p", answer="Paris"
    )
    judge, tracker = _fake_judge_and_tracker()
    await rollout_mod.rollout_fn(
        model,  # type: ignore[arg-type]
        question,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
    )
    assert calls == {"judge": False, "verifiable": True}


@pytest.mark.asyncio
async def test_rollout_fn_raises_on_question_with_neither_rubrics_nor_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    monkeypatch.setattr(
        rollout_mod, "build_pipeline", lambda **kwargs: _RecordingPipeline()
    )
    monkeypatch.setattr(rollout_mod, "run_pipeline", _fake_run_pipeline)

    model = _make_model(monkeypatch)
    question = Question(question_id="q1", source="mystery", split="train", prompt="p")
    judge, tracker = _fake_judge_and_tracker()
    with pytest.raises(ValueError, match=r"neither \.rubrics nor \.answer"):
        await rollout_mod.rollout_fn(
            model,  # type: ignore[arg-type]
            question,
            judge=judge,  # type: ignore[arg-type]
            tracker=tracker,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_run_rollout_group_runs_rollouts_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: run_rollout_group used to await each rollout one
    at a time in a plain loop, paying group_size x one rollout's latency
    for no reason -- each rollout in a group is independent. Confirms the
    fix actually overlaps them, not just that it doesn't crash."""
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    in_flight = 0
    max_in_flight = 0

    async def fake_rollout_fn(model: object, question: object, **kwargs: object) -> str:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return "trajectory"

    monkeypatch.setattr(rollout_mod, "rollout_fn", fake_rollout_fn)

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1", source="musique", split="train", prompt="p", answer="Paris"
    )
    judge, tracker = _fake_judge_and_tracker()

    start = time.monotonic()
    await rollout_mod.run_rollout_group(
        model,  # type: ignore[arg-type]
        question,
        group_size=4,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
    )
    elapsed = time.monotonic() - start

    assert max_in_flight == 4  # all 4 overlapped, not one at a time
    assert elapsed < 0.15  # well under 4 x 0.05s sequential


@pytest.mark.asyncio
async def test_run_rollout_group_llm_semaphore_bounds_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mas_deepr.data.schema import Question
    from mas_deepr.rl import rollout as rollout_mod

    in_flight = 0
    max_in_flight = 0

    async def fake_rollout_fn(model: object, question: object, **kwargs: object) -> str:
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.03)
        in_flight -= 1
        return "trajectory"

    monkeypatch.setattr(rollout_mod, "rollout_fn", fake_rollout_fn)

    model = _make_model(monkeypatch)
    question = Question(
        question_id="q1", source="musique", split="train", prompt="p", answer="Paris"
    )
    judge, tracker = _fake_judge_and_tracker()
    semaphore = asyncio.Semaphore(2)

    await rollout_mod.run_rollout_group(
        model,  # type: ignore[arg-type]
        question,
        group_size=6,
        judge=judge,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
        llm_semaphore=semaphore,
    )
    assert max_in_flight == 2
