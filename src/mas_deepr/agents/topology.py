"""3-agent MAF pipeline: Manager -> Browser -> Synthesizer.

One shared SLM plays all three roles, distinguished by system-prompt
instructions loaded from the prompt registry -- this is the model GRPO will
later train as a single LoRA on role-tagged trajectories (see plan Phase 4).
The pipeline itself is a plain sequential orchestration over MAF ``Agent``
instances; swap this module for a ``FunctionalWorkflow`` graph later without
touching tools/, prompts/, or telemetry/.

Phase 3 adds an optional ``memory`` strategy (see ``memory/``) that wraps
this same body in a multi-pass loop and injects extra context at the
Manager/Browser prompts. ``memory=None`` (the default) preserves the
pre-Phase-3 single-pass behavior exactly -- see ``tests/test_memory.py``
for the regression contract.
"""

import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from agent_framework import Agent, ChatOptions, FunctionTool
from openai.types.chat.chat_completion import Choice as OpenAIChoice

from mas_deepr.agents.parsing import parse_sub_questions
from mas_deepr.config import ModelSpec, Settings
from mas_deepr.llm import build_chat_client
from mas_deepr.llm.rate_limit import (
    NullTpmLimiter,
    TpmLimiter,
    build_tpm_limiter,
    estimate_tokens,
)
from mas_deepr.memory import MemoryStrategy, PassRecord, StatelessStrategy
from mas_deepr.memory.base import BranchTrajectory, MainThreadState
from mas_deepr.prompts import load_prompt
from mas_deepr.telemetry import TelemetryTracker, Timer, usage_from_response
from mas_deepr.tools import (
    MCPToolClient,
    make_code_exec_tool,
    make_fetch_page_tool,
    make_pubmed_tool,
    make_semantic_scholar_tool,
    make_tavily_tool,
    make_web_search_tool,
    make_wikipedia_tool,
)
from mas_deepr.tools.tool_telemetry import current_question_id

logger = logging.getLogger(__name__)

_DIRECT_PREFIX = "DIRECT:"


@dataclass
class ResearchPipeline:
    """Bundle of the three role agents plus everything needed to run them."""

    manager: Agent
    browser: Agent
    synthesizer: Agent
    spec: ModelSpec
    tracker: TelemetryTracker
    compressor: Agent | None = None
    # Proactive TPM pacing (llm/rate_limit.py) -- a NullTpmLimiter when
    # spec.tpm_limit is unset, so _run_agent never needs to branch on
    # whether one is active.
    tpm_limiter: "TpmLimiter | NullTpmLimiter" = field(
        default_factory=lambda: NullTpmLimiter()
    )
    # None for pipelines built without a real MCP client (unit tests
    # constructing ResearchPipeline directly with fake agents) -- when set,
    # run_pipeline holds one connection open for the whole run instead of
    # every tool call reconnecting (tools/mcp_client.py's module docstring).
    mcp_client: MCPToolClient | None = None


@dataclass
class PipelineResult:
    question_id: str
    question: str
    sub_questions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    final_answer: str = ""
    raw_findings: list[str] = field(default_factory=list)
    pass_records: list[PassRecord] = field(default_factory=list)
    memory_strategy: str = "stateless"
    # Real per-role message traces, one (role, AgentTurn) pair per actual
    # agent.run() call, in the exact order they happened -- a single
    # chronological list, not per-role lists, because a memory strategy can
    # run multiple passes (manager -> browser* -> synthesizer, repeated) and
    # only true call order lets a consumer reconstruct one honest trajectory
    # across passes. Only populated when run_pipeline(...,
    # capture_logprobs=True) is set -- see rl/rollout.py, the only consumer.
    role_turns: list[tuple[str, "AgentTurn"]] = field(default_factory=list)


def build_pipeline(
    *,
    spec: ModelSpec,
    settings: Settings,
    tracker: TelemetryTracker,
    prefer_compiled: bool = False,
    trainable: bool = False,
) -> ResearchPipeline:
    """Construct the Manager/Browser/Synthesizer/Compressor agents for ``spec``.

    ``trainable=True`` (GRPO rollouts only, see ``rl/rollout.py``) swaps in
    ``rl.trainable_client.TrainableOpenAIChatCompletionClient`` so a pure
    tool-call turn's ``Choice`` survives instead of being discarded --
    lazily imported here so eval/baseline pipelines never pull in
    training-only code. False (default) is MAF's stock client, unchanged
    behavior for every non-GRPO caller.
    """
    if not spec.supports_native_tools:
        raise NotImplementedError(
            f"model {spec.key!r} can't do native function-calling; the Browser "
            "role needs the prompted-JSON-tools fallback (plan Stage G, not "
            "built yet). Pick a tool-calling model or implement the fallback."
        )
    if trainable:
        from mas_deepr.rl.trainable_client import TrainableOpenAIChatCompletionClient

        client = build_chat_client(
            spec, settings, client_cls=TrainableOpenAIChatCompletionClient
        )
    else:
        client = build_chat_client(spec, settings)
    # Tool calls proxy through the MCP backend (mcp_backend/, run via
    # scripts/run_mcp_server.py) instead of calling providers in-process --
    # see docs/dr-tulu-agent-infra.md and the MCP-adoption plan.
    mcp_client = MCPToolClient(
        url=settings.mcp_server_url,
        transport=settings.mcp_transport,
        concurrency=settings.mcp_tool_concurrency,
        searxng_concurrency=settings.mcp_searxng_concurrency,
    )
    enabled_tools = settings.mcp_enabled_tools_set()

    manager = Agent(
        client,
        instructions=load_prompt(
            "manager", prefer_compiled=prefer_compiled
        ).instructions,
        name="manager",
    )
    # Single gate for all 6 MCP tools -- settings.mcp_enabled_tools_set().
    # Uniform on purpose: previously web_search/fetch_page were hardcoded
    # on regardless of this set, academic tools used a separate bool, and
    # only Tavily actually respected the set -- three inconsistent
    # mechanisms for what should be one. Now every tool is a plain
    # (name, factory) pair checked against the same set, so e.g. dropping
    # "web_search" from MAS_MCP_ENABLED_TOOLS (keeping wikipedia_search,
    # fetch_page, semantic_scholar_search, pubmed_search) is enough for a
    # genuinely ban-risk-free run on Wikipedia-answerable benchmarks --
    # see the deployment checklist.
    tool_factories: list[tuple[str, Callable[[], FunctionTool]]] = [
        (
            "web_search",
            lambda: make_web_search_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                max_results=settings.search_max_results,
            ),
        ),
        (
            "fetch_page",
            lambda: make_fetch_page_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                timeout_s=settings.fetch_timeout_s,
                max_chars=settings.fetch_max_chars,
            ),
        ),
        (
            "wikipedia_search",
            lambda: make_wikipedia_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                max_results=settings.search_max_results,
            ),
        ),
        (
            "semantic_scholar_search",
            lambda: make_semantic_scholar_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                max_results=settings.search_max_results,
            ),
        ),
        (
            "pubmed_search",
            lambda: make_pubmed_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                max_results=settings.search_max_results,
            ),
        ),
        (
            "tavily_search",
            lambda: make_tavily_tool(
                mcp_client=mcp_client,
                tracker=tracker,
                max_results=settings.search_max_results,
            ),
        ),
    ]
    browser_tools = [
        factory() for name, factory in tool_factories if name in enabled_tools
    ]
    browser = Agent(
        client,
        instructions=load_prompt(
            "browser", prefer_compiled=prefer_compiled
        ).instructions,
        name="browser",
        tools=browser_tools,
    )
    synthesizer = Agent(
        client,
        instructions=load_prompt(
            "synthesizer", prefer_compiled=prefer_compiled
        ).instructions,
        name="synthesizer",
        # Dropped, not just discouraged via prompt, for models that can't
        # reliably use it (ModelSpec.supports_code_exec) -- confirmed live
        # that offering it anyway produces malformed tool-call-shaped text
        # as the "final answer" instead of either a real tool call or a
        # real (if unverified) answer.
        tools=[make_code_exec_tool()] if spec.supports_code_exec else [],
    )
    # Only used by memory strategies that need a compression call (e.g.
    # RE-TRAC); cheap to always build, since an unused Agent costs nothing
    # until ``.run()`` is actually called.
    compressor = Agent(
        client,
        instructions=load_prompt("compressor", prefer_compiled=False).instructions,
        name="compressor",
    )
    return ResearchPipeline(
        manager=manager,
        browser=browser,
        synthesizer=synthesizer,
        spec=spec,
        tracker=tracker,
        compressor=compressor,
        tpm_limiter=build_tpm_limiter(spec.tpm_limit),
        mcp_client=mcp_client,
    )


@dataclass
class AgentTurn:
    """One ``agent.run()`` call's outcome: final text plus its full trajectory.

    ``messages`` are MAF ``Message`` objects, in the shape ``agent.run()``
    returns them (unverified against a live server whether this includes the
    input prompt or only newly-generated turns -- see
    ``rl/rollout.py::build_trajectory_from_result``, the one place that cares).
    When the call was made with ``capture_logprobs=True``, an assistant
    message that produced text content carries the real OpenAI ``Choice``
    (with populated ``.logprobs``) as that content's ``raw_representation``
    -- see ``choice_from_message`` below. A pure tool-call turn (no text)
    carries one too *only* when the pipeline was built with
    ``build_pipeline(..., trainable=True)`` (GRPO rollouts,
    ``rl/rollout.py``) -- MAF's stock client discards that ``Choice`` for
    function-call content (``_parse_tool_calls_from_openai``); the
    training-only ``rl.trainable_client.TrainableOpenAIChatCompletionClient``
    preserves it instead (single tool call per turn only -- see that
    module's docstring for why parallel tool calls fall back to the stock,
    undiscovered-Choice behavior). Eval/baseline pipelines never set
    ``trainable=True``, so "which tool to call" stays untrained context
    there, unchanged from before.
    """

    text: str
    messages: list[object] = field(default_factory=list)


def choice_from_message(message: Any) -> OpenAIChoice | None:
    """Pull the real, logprob-bearing OpenAI ``Choice`` off a MAF assistant
    message, if one was captured (see ``AgentTurn`` docstring for when).

    Checks ``function_call`` content too, not just ``text`` -- with
    ``rl/trainable_client.py``'s GRPO-only client, a pure tool-call turn's
    ``Choice`` (single tool call, no accompanying text) is preserved there
    instead of discarded, so "which tool to call" is now trainable, not
    just "what to write". With MAF's stock client (eval/baseline runs),
    ``function_call`` content never carries a ``Choice``, so this check is
    a no-op there -- unchanged behavior for anything that isn't a GRPO
    rollout.
    """
    for content in getattr(message, "contents", None) or []:
        if getattr(content, "type", None) in ("text", "function_call"):
            raw = getattr(content, "raw_representation", None)
            if isinstance(raw, OpenAIChoice):
                return raw
    return None


async def _run_agent(
    pipeline: ResearchPipeline,
    agent: Agent,
    prompt: str,
    *,
    role: str,
    question_id: str,
    max_function_calls: int | None = None,
    memory_strategy: str | None = None,
    pass_index: int | None = None,
    capture_logprobs: bool = False,
) -> AgentTurn:
    invocation_kwargs = (
        {"max_function_calls": max_function_calls}
        if max_function_calls is not None
        else None
    )
    # Only requested for GRPO rollouts (rl/rollout.py) -- eval/baseline
    # callers never set this, so their inference cost/latency is unaffected.
    # extra_body={"return_token_ids": True} is a real vLLM OpenAI-compatible
    # extension (v0.10.2+, confirmed live 2026-09-28 against ART's own
    # error): without it, ART's serverless trainer rejects the trajectory
    # outright -- "Trainable Choice is missing vLLM prompt_token_ids/
    # token_ids. Use a vLLM endpoint with return_token_ids enabled." --
    # since it needs the exact tokens actually generated (not a
    # re-tokenization of the returned text, which can silently drift from
    # what was really sampled) to compute the training loss correctly.
    run_options: ChatOptions | None = (
        cast(
            ChatOptions,
            {"logprobs": True, "extra_body": {"return_token_ids": True}},
        )
        if capture_logprobs
        else None
    )
    # Proactive TPM pacing -- reserves estimated token budget before the
    # call goes out, so most 429s never happen (vs. llm_max_retries, which
    # only reacts after one already did). A no-op when the spec has no
    # known tpm_limit. See llm/rate_limit.py.
    await pipeline.tpm_limiter.acquire(estimate_tokens(prompt))
    with Timer() as t:
        try:
            resp = await agent.run(
                prompt,
                function_invocation_kwargs=invocation_kwargs,
                options=run_options,
            )
        except Exception as e:
            logger.warning(
                "agent call failed role=%s question_id=%s error=%s: %s",
                role,
                question_id,
                type(e).__name__,
                e,
            )
            pipeline.tracker.record(
                role=role,
                spec=pipeline.spec,
                input_tokens=0,
                output_tokens=0,
                latency_s=t.elapsed_s,
                question_id=question_id,
                error=f"{type(e).__name__}: {e}",
                memory_strategy=memory_strategy,
                pass_index=pass_index,
            )
            raise
    input_tokens, output_tokens = usage_from_response(resp)
    pipeline.tracker.record(
        role=role,
        spec=pipeline.spec,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_s=t.elapsed_s,
        question_id=question_id,
        memory_strategy=memory_strategy,
        pass_index=pass_index,
    )
    logger.debug(
        "agent call ok role=%s question_id=%s latency_s=%.2f tokens_in=%d "
        "tokens_out=%d",
        role,
        question_id,
        t.elapsed_s,
        input_tokens,
        output_tokens,
    )
    messages = list(getattr(resp, "messages", None) or [])
    return AgentTurn(text=resp.text or "", messages=messages)


async def run_pipeline(
    pipeline: ResearchPipeline,
    question: str,
    *,
    question_id: str,
    max_sub_queries: int = 4,
    max_tool_calls_per_query: int = 8,
    memory: MemoryStrategy | None = None,
    capture_logprobs: bool = False,
) -> PipelineResult:
    """Run one question end-to-end through Manager -> Browser* -> Synthesizer.

    ``memory=None`` runs exactly one pass with no extra injected context --
    byte-identical to the pre-Phase-3 pipeline. Passing a ``MemoryStrategy``
    wraps this same body in up to ``strategy.max_passes`` passes, injecting
    ``prepare_pass``/``prepare_branch`` text and calling
    ``fold_branch``/``finalize_pass`` at the corresponding points.

    ``capture_logprobs=True`` (only set by ``rl/rollout.py``) additionally
    requests logprobs on every role call (manager, browser, synthesizer,
    and -- if a memory strategy uses one -- compressor) and records each
    real ``AgentTurn`` onto ``result.role_turns``, for GRPO to train on
    instead of a fabricated (question, final_answer) pair. No effect on
    eval/baseline callers, which never set it.

    Holds ``pipeline.mcp_client`` open for this whole run (via
    ``AsyncExitStack``, skipped when ``None``) so every Browser tool call
    inside it reuses one connection instead of reconnecting per call --
    see ``tools/mcp_client.py``'s module docstring. Safe when multiple
    questions share one pipeline and run concurrently (evals/runner.py's
    ``asyncio.gather``): fastmcp's ``Client`` is an explicitly reentrant,
    reference-counted context manager, so overlapping ``run_pipeline``
    calls just keep one connection alive as long as any of them are
    still using it, each still bounded by ``MCPToolClient``'s own
    per-provider semaphores.
    """
    # Tool closures (built once per model in build_pipeline) are shared
    # across every concurrently-running question in evals/runner.py's
    # asyncio.gather -- a tool has no other way to know which question
    # invoked it. Each gather()ed coroutine gets its own copy of this
    # context, so concurrent questions don't collide; reset in `finally`
    # so the var doesn't leak into whatever reuses this task/thread next.
    token = current_question_id.set(question_id)
    try:
        async with contextlib.AsyncExitStack() as stack:
            # Hold one MCP connection open for the whole run instead of
            # every tool call reconnecting (tools/mcp_client.py's module
            # docstring) -- None for pipelines built without a real MCP
            # client (unit tests with fake agents).
            if pipeline.mcp_client is not None:
                await stack.enter_async_context(pipeline.mcp_client)
            strategy: MemoryStrategy = memory or StatelessStrategy()
            result = PipelineResult(
                question_id=question_id,
                question=question,
                memory_strategy=strategy.name,
            )
            main_thread = MainThreadState(question=question)
            prior_passes: list[PassRecord] = []

            async def _compress(prompt: str) -> str:
                if pipeline.compressor is None:
                    return ""
                turn = await _run_agent(
                    pipeline,
                    pipeline.compressor,
                    prompt,
                    role="compressor",
                    question_id=question_id,
                    memory_strategy=strategy.name,
                    capture_logprobs=capture_logprobs,
                )
                result.role_turns.append(("compressor", turn))
                return turn.text

            for pass_index in range(strategy.max_passes):
                pass_prefix = await strategy.prepare_pass(
                    question=question, prior_passes=prior_passes, pass_index=pass_index
                )
                manager_prompt = (
                    f"{pass_prefix}\n\n{question}" if pass_prefix else question
                )

                manager_turn = await _run_agent(
                    pipeline,
                    pipeline.manager,
                    manager_prompt,
                    role="manager",
                    question_id=question_id,
                    memory_strategy=strategy.name,
                    pass_index=pass_index,
                    capture_logprobs=capture_logprobs,
                )
                result.role_turns.append(("manager", manager_turn))
                manager_out = manager_turn.text

                if manager_out.strip().startswith(_DIRECT_PREFIX):
                    logger.info(
                        "manager DIRECT fast-path question_id=%s pass=%d",
                        question_id,
                        pass_index,
                    )
                    result.final_answer = manager_out.strip()[
                        len(_DIRECT_PREFIX) :
                    ].strip()
                    return result

                sub_questions = parse_sub_questions(
                    manager_out, max_sub_queries=max_sub_queries
                )
                logger.debug(
                    "manager produced %d sub-questions question_id=%s pass=%d",
                    len(sub_questions),
                    question_id,
                    pass_index,
                )
                pass_findings: list[str] = []

                for sub_q in sub_questions:
                    branch_prefix = await strategy.prepare_branch(
                        sub_question=sub_q, main_thread=main_thread
                    )
                    browser_prompt = (
                        f"{branch_prefix}\n\n{sub_q}" if branch_prefix else sub_q
                    )

                    browser_turn = await _run_agent(
                        pipeline,
                        pipeline.browser,
                        browser_prompt,
                        role="browser",
                        question_id=question_id,
                        max_function_calls=max_tool_calls_per_query,
                        memory_strategy=strategy.name,
                        pass_index=pass_index,
                        capture_logprobs=capture_logprobs,
                    )
                    result.role_turns.append(("browser", browser_turn))
                    raw_finding = browser_turn.text
                    main_thread.branches.append(
                        BranchTrajectory(
                            sub_question=sub_q,
                            raw_finding=raw_finding,
                            messages=browser_turn.messages,
                        )
                    )
                    result.raw_findings.append(raw_finding)

                    folded = await strategy.fold_branch(
                        sub_question=sub_q,
                        raw_finding=raw_finding,
                        main_thread=main_thread,
                    )
                    pass_findings.append(folded)

                synth_prompt = f"Original question: {question}\n\n" + "\n\n".join(
                    f"Sub-question: {sq}\nFindings: {f}"
                    for sq, f in zip(sub_questions, pass_findings, strict=True)
                )
                synth_turn = await _run_agent(
                    pipeline,
                    pipeline.synthesizer,
                    synth_prompt,
                    role="synthesizer",
                    question_id=question_id,
                    memory_strategy=strategy.name,
                    pass_index=pass_index,
                    capture_logprobs=capture_logprobs,
                )
                result.role_turns.append(("synthesizer", synth_turn))
                final_answer = synth_turn.text

                pass_record = await strategy.finalize_pass(
                    final_answer=final_answer,
                    sub_questions=sub_questions,
                    findings=pass_findings,
                    main_thread=main_thread,
                    pass_index=pass_index,
                    compress=_compress,
                )
                pass_record.pass_index = pass_index
                prior_passes.append(pass_record)

                result.sub_questions = sub_questions
                result.findings = pass_findings
                result.final_answer = final_answer

            result.pass_records = prior_passes
            return result
    finally:
        current_question_id.reset(token)
