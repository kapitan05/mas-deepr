"""ART rollout function: wraps ``agents.topology.run_pipeline`` for GRPO.

Per docs/plan/plan-v2.md's Phase 4 decision: OpenPipe ART, not veRL/verl-
agent, because it needs only ``Settings.slm_base_url`` repointed at ART's
serving endpoint -- the "designed-but-unused indirection point" from Phase
0 -- rather than reimplementing Manager->Browser->Synthesizer as a
gym-style ``reset``/``step`` loop.

**Status: written against the real ``art`` API (0.5.x), but genuinely
unverified** -- running it needs a live ``art.TrainableModel`` registered
against a real backend. ``scripts/train_grpo.py`` defaults to
``ServerlessBackend`` (training + inference on W&B's servers, no local
GPU; needs W&B Training access + ``WANDB_API_KEY``).
``build_trajectory_from_result`` is pure and unit tested
(``tests/test_rollout.py``); ``rollout_fn``/``run_rollout_group`` need
that live backend before their first real run -- treat that first run,
not this code, as milestone M8's actual starting point.

``build_trajectory_from_result`` now builds the real, concatenated
Manager->Browser*->Synthesizer trajectory (logprobs captured via
``capture_logprobs=True`` on ``run_pipeline``) instead of a fabricated
single-turn stub -- see its own docstring for exactly what is and isn't
trainable in that sequence, and the two specific things (tool-call-turn
logprobs, coarse cross-role credit assignment) still unverified/simplified.
"""

import asyncio
import json
from typing import Any, cast

import art
from art.types import Message as ArtMessage
from openai.types.chat.chat_completion import Choice as OpenAIChoice

from mas_deepr.agents.topology import (
    AgentTurn,
    PipelineResult,
    build_pipeline,
    choice_from_message,
    run_pipeline,
)
from mas_deepr.config import ModelSpec, Settings, get_tool_scope
from mas_deepr.data.schema import Question
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.rl.rubric_reward import (
    score_rollout_group,
    score_rollout_group_verifiable,
)
from mas_deepr.telemetry import TelemetryTracker

_MessageOrChoice = ArtMessage | OpenAIChoice


def _as_art_message(
    role: str,
    text: str,
    *,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> ArtMessage:
    """Cast a plain dict to ART's ``Message`` union.

    ``role`` comes from a MAF message at runtime (one of user/system/
    assistant/tool), which is exactly the set ``art.types.Message``'s
    ``ChatCompletionXMessageParam`` union covers -- a real, checked cast,
    not an escape hatch to ``Any``. A tool-role message needs its
    ``tool_call_id`` for correct chat-template rendering downstream
    (``ChatCompletionToolMessageParam`` requires it); an assistant message
    that called a tool needs ``tool_calls`` for the same reason, so a
    following tool-role message isn't answering a call nobody can see.
    """
    entry: dict[str, Any] = {"role": role, "content": text}
    if tool_call_id is not None:
        entry["tool_call_id"] = tool_call_id
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return cast(ArtMessage, entry)


def _tool_call_id(message: object) -> str | None:
    """The ``call_id`` a tool-result message answers, if MAF captured one
    on its content (``Content.from_function_result(call_id=...)``)."""
    for content in getattr(message, "contents", None) or []:
        call_id = getattr(content, "call_id", None)
        if call_id:
            return str(call_id)
    return None


def _tool_calls(message: object) -> list[dict[str, Any]]:
    """OpenAI-shaped ``tool_calls`` for an assistant message's function-call
    content, if any (``Content.from_function_call(call_id, name, arguments)``).

    Needed on the *fallback* (no-Choice) path specifically: when a Choice
    is captured, ``choice.message.tool_calls`` already carries this; when a
    turn is a pure tool call with no text (no Choice, per
    ``choice_from_message``'s documented limit), this is the only place
    that information survives -- without it, the fallback plain-dict
    assistant message would render as an empty turn preceding a tool-role
    message that answers a call nobody can see.
    """
    calls: list[dict[str, Any]] = []
    for content in getattr(message, "contents", None) or []:
        if getattr(content, "type", None) != "function_call":
            continue
        call_id = getattr(content, "call_id", None)
        name = getattr(content, "name", None)
        if not call_id or not name:
            continue
        arguments = getattr(content, "arguments", None)
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {})
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


def _entries_for_turn(turn: AgentTurn) -> list[_MessageOrChoice]:
    """Flatten one real ``AgentTurn`` into ``messages_and_choices`` entries.

    An assistant message becomes a trainable ``Choice`` when
    ``choice_from_message`` found one (text content, logprobs requested);
    everything else -- system/user/tool messages, and assistant turns that
    were pure tool calls with no text -- is recorded as plain-dict context,
    carrying no gradient, but with its ``tool_calls``/``tool_call_id``
    preserved so the transcript stays coherent even where it isn't
    trainable. See ``choice_from_message``'s docstring in
    ``agents/topology.py`` for exactly why tool-call turns fall in that
    second bucket.
    """
    entries: list[_MessageOrChoice] = []
    for message in turn.messages:
        role = getattr(message, "role", None) or "user"
        text = getattr(message, "text", None) or ""
        if role == "assistant":
            choice = choice_from_message(message)
            entries.append(
                choice
                if choice is not None
                else _as_art_message(role, text, tool_calls=_tool_calls(message))
            )
        elif role == "tool":
            call_id = _tool_call_id(message)
            entries.append(_as_art_message(role, text, tool_call_id=call_id))
        else:
            entries.append(_as_art_message(role, text))
    return entries


def build_trajectory_from_result(
    *, question: str, result: PipelineResult, reward: float
) -> art.Trajectory:
    """Flatten one ``PipelineResult`` into an ``art.Trajectory``.

    Concatenates the *real* Manager -> Browser* -> Synthesizer message
    sequence (``result.role_turns``, populated only when ``run_pipeline``
    was called with ``capture_logprobs=True`` -- see ``rollout_fn`` below)
    into one trajectory, broadcasting the single downstream ``reward``
    across every trainable span in it. This replaces the fabricated
    ``(question, final_answer)`` stub this function used to always return:
    trainable spans now carry the logprobs the policy actually produced
    under each role's real system prompt, instead of teacher-forcing text
    against a context it was never sampled under.

    One thing this does NOT fix, still coarse and a deliberate,
    simplest-correct-fix choice rather than the only possible one: credit
    assignment across roles -- one reward broadcast across the whole real
    sequence, not a role-specific reward.

    A Browser turn's tool-call decisions (which tool, which args) ARE now
    trainable, single-tool-call turns only -- ``rollout_fn`` builds the
    pipeline with ``trainable=True``, which swaps in
    ``rl.trainable_client.TrainableOpenAIChatCompletionClient`` so that
    ``Choice`` survives instead of MAF's stock client discarding it (see
    ``agents/topology.py``'s ``AgentTurn``/``choice_from_message``
    docstrings). Unverified against a live server still (check on the
    first real M8 run), same as everything else in this module.

    Falls back to the old fabricated single-turn shape when
    ``result.role_turns`` is empty (``run_pipeline`` called without
    ``capture_logprobs=True``), so this stays usable for callers/tests that
    don't need the real trace.
    """
    base_metrics: dict[str, float | int | bool] = {
        "num_sub_questions": float(len(result.sub_questions)),
        "num_raw_findings": float(len(result.raw_findings)),
    }
    metadata: dict[str, float | int | str | bool | None] = {
        "question_id": result.question_id,
        "memory_strategy": result.memory_strategy,
        "num_sub_questions": len(result.sub_questions),
        "final_answer": result.final_answer[:2000],
    }

    if not result.role_turns:
        return art.Trajectory(
            messages_and_choices=[
                _as_art_message("user", question),
                _as_art_message("assistant", result.final_answer),
            ],
            reward=reward,
            metrics=base_metrics,
            metadata=metadata,
        )

    messages_and_choices: list[_MessageOrChoice] = []
    for _role, turn in result.role_turns:
        messages_and_choices.extend(_entries_for_turn(turn))

    return art.Trajectory(
        messages_and_choices=messages_and_choices,
        reward=reward,
        metrics={**base_metrics, "num_role_turns": float(len(result.role_turns))},
        metadata=metadata,
    )


async def rollout_fn(
    model: art.TrainableModel,
    question: Question,
    *,
    judge: JudgeClient,
    tracker: TelemetryTracker,
    tool_scope: str = "wiki_paper",
    prefer_compiled: bool = True,
) -> art.Trajectory:
    """One ART rollout: run the MAS pipeline against ``model``'s inference
    endpoint, grade it, return a scored ``Trajectory``.

    ``model.inference_base_url`` is ART's field for the OpenAI-compatible
    endpoint a ``TrainableModel`` exposes during training -- pointing
    ``Settings.slm_base_url`` there is the entire integration surface, per
    the plan's Phase 0 design. The model *name* sent with each request is
    NOT ``model.inference_model_name`` directly (see below) -- that field
    is unversioned and ART's ServerlessBackend rejects it outright.

    ``tool_scope`` defaults to ``"wiki_paper"`` -- without this, rollouts
    would fall through to ``Settings``' class default (general scope minus
    Tavily) and hit ``web_search``/SearXNG at training volume, the exact
    ban risk ``config/tool_scopes.py`` exists to prevent (confirmed no
    scope was ever set here previously). ``prefer_compiled`` defaults to
    ``True`` so a DSPy-compiled prompt (``scripts/compile_dspy.py``) is
    used once one exists; ``prompts/registry.py::load_prompt`` no-ops back
    to the hand-written prompt if it doesn't, so this is safe either way.

    Reward dispatch is on the ``Question`` itself, not a separately-passed
    rubric list: a rubric-bearing question (ResearchRubrics-shaped,
    including DR-Tulu's own rl-data) grades via the judge; a verifiable-
    answer question (MuSiQue/HotpotQA/FRAMES-shaped) grades judge-free via
    exact match -- see ``rl/rubric_reward.py``. A question with neither is
    a caller bug, not a silent zero-reward.
    """
    # model.inference_model_name is UNVERSIONED (no :stepN suffix) --
    # confirmed live 2026-09-28 that ART's ServerlessBackend rejects it
    # outright for inference ("LoRA inference requires an explicit
    # checkpoint... Unversioned references and aliases such as 'latest'
    # are not supported"), even for a model that has never been trained: a
    # fresh registration IS servable, but only via the explicit
    # ``:step0`` reference, which the server auto-provisions as the base
    # weights. get_inference_name(step=...) is the versioned name that
    # actually works -- matches OpenPipe's own reference training loop
    # (`start_step = await model.get_step()` immediately after
    # registration, before any rollout).
    step = await model.get_step()
    settings = Settings(
        slm_base_url=model.inference_base_url or "",
        slm_api_key=model.inference_api_key or "EMPTY",
        slm_model=model.get_inference_name(step=step),
        mcp_enabled_tools=get_tool_scope(tool_scope),
    )
    spec = ModelSpec(key=model.name, model_id=settings.slm_model, family="qwen3")
    pipeline = build_pipeline(
        spec=spec,
        settings=settings,
        tracker=tracker,
        prefer_compiled=prefer_compiled,
        trainable=True,
    )

    result = await run_pipeline(
        pipeline,
        question.prompt,
        question_id=question.question_id,
        capture_logprobs=True,
    )

    if question.rubrics is not None:
        outcomes = await score_rollout_group(
            judge=judge,
            prompt=question.prompt,
            question_id=question.question_id,
            rollouts=[result.final_answer],
            rubrics=question.rubrics,
        )
    elif question.answer is not None:
        outcomes = await score_rollout_group_verifiable(
            prompt=question.prompt,
            question_id=question.question_id,
            gold_answer=question.answer,
            aliases=question.answer_aliases,
            rollouts=[result.final_answer],
        )
    else:
        raise ValueError(
            f"question {question.question_id!r} has neither .rubrics nor "
            ".answer -- nothing to grade against"
        )
    reward = outcomes[0].score if outcomes else 0.0
    return build_trajectory_from_result(
        question=question.prompt, result=result, reward=reward
    )


async def run_rollout_group(
    model: art.TrainableModel,
    question: Question,
    *,
    group_size: int,
    judge: JudgeClient,
    tracker: TelemetryTracker,
    tool_scope: str = "wiki_paper",
    prefer_compiled: bool = True,
    llm_semaphore: asyncio.Semaphore | None = None,
) -> art.TrajectoryGroup:
    """Run ``group_size`` rollouts of the same question concurrently --
    one GRPO group.

    Runs the rollouts via ``asyncio.gather``, not a sequential loop: each
    rollout in a group answers the same question independently, so there
    is nothing to serialize on -- matching DR-Tulu's own concurrency
    principle (docs/dr-tulu-agent-infra.md ss2), "fan out freely at the
    call site, cap hard at the resource layer" rather than rationing
    parallelism at the call site itself. Fixed 2026-09-25: this used to
    `await` each rollout one at a time, paying ``group_size``x one
    rollout's latency for no reason.

    ``llm_semaphore``, when given, bounds how many whole rollouts run
    concurrently against the served model -- construct ONE
    ``asyncio.Semaphore(settings.grpo_llm_concurrency)`` per training run
    (not per group/question) and pass the same instance to every call, so
    it actually bounds total in-flight rollouts across every question,
    not just within one group. ``None`` (tests, ad-hoc scripts) means
    unbounded -- ``scripts/train_grpo.py`` always passes a real one.

    GRPO's group-normalized advantage (``rubric_reward.group_advantages``)
    is computed by ART's own trainer from each trajectory's ``.reward``;
    this just gathers the group, it doesn't compute advantages itself.
    """

    async def _one_rollout() -> art.Trajectory:
        if llm_semaphore is None:
            return await rollout_fn(
                model,
                question,
                judge=judge,
                tracker=tracker,
                tool_scope=tool_scope,
                prefer_compiled=prefer_compiled,
            )
        async with llm_semaphore:
            return await rollout_fn(
                model,
                question,
                judge=judge,
                tracker=tracker,
                tool_scope=tool_scope,
                prefer_compiled=prefer_compiled,
            )

    trajectories: list[Any] = await asyncio.gather(
        *(_one_rollout() for _ in range(group_size))
    )
    return art.TrajectoryGroup(trajectories=trajectories)
