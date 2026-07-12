"""3-agent MAF pipeline: Manager -> Browser -> Synthesizer.

One shared SLM plays all three roles, distinguished by system-prompt
instructions loaded from the prompt registry -- this is the model GRPO will
later train as a single LoRA on role-tagged trajectories (see plan Phase 4).
The pipeline itself is a plain sequential orchestration over MAF ``Agent``
instances; swap this module for a ``FunctionalWorkflow`` graph later without
touching tools/, prompts/, or telemetry/.
"""

from dataclasses import dataclass, field

from agent_framework import Agent

from mas_deepr.agents.parsing import parse_sub_questions
from mas_deepr.config import ModelSpec, Settings
from mas_deepr.llm import build_chat_client
from mas_deepr.prompts import load_prompt
from mas_deepr.telemetry import TelemetryTracker, Timer, usage_from_response
from mas_deepr.tools import (
    WebCache,
    make_code_exec_tool,
    make_fetch_page_tool,
    make_web_search_tool,
)


@dataclass
class ResearchPipeline:
    """Bundle of the three role agents plus everything needed to run them."""

    manager: Agent
    browser: Agent
    synthesizer: Agent
    spec: ModelSpec
    tracker: TelemetryTracker


@dataclass
class PipelineResult:
    question_id: str
    question: str
    sub_questions: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    final_answer: str = ""


def build_pipeline(
    *,
    spec: ModelSpec,
    settings: Settings,
    tracker: TelemetryTracker,
    prefer_compiled: bool = False,
) -> ResearchPipeline:
    """Construct the Manager/Browser/Synthesizer agents for ``spec``."""
    client = build_chat_client(spec, settings)
    cache = WebCache(settings.cache_db)

    manager = Agent(
        client,
        instructions=load_prompt(
            "manager", prefer_compiled=prefer_compiled
        ).instructions,
        name="manager",
    )
    browser = Agent(
        client,
        instructions=load_prompt(
            "browser", prefer_compiled=prefer_compiled
        ).instructions,
        name="browser",
        tools=[
            make_web_search_tool(
                api_key=settings.tavily_api_key,
                cache=cache,
                max_results=settings.search_max_results,
            ),
            make_fetch_page_tool(
                timeout_s=settings.fetch_timeout_s,
                max_chars=settings.fetch_max_chars,
                cache=cache,
            ),
        ],
    )
    synthesizer = Agent(
        client,
        instructions=load_prompt(
            "synthesizer", prefer_compiled=prefer_compiled
        ).instructions,
        name="synthesizer",
        tools=[make_code_exec_tool()],
    )
    return ResearchPipeline(
        manager=manager,
        browser=browser,
        synthesizer=synthesizer,
        spec=spec,
        tracker=tracker,
    )


async def _run_agent(
    pipeline: ResearchPipeline,
    agent: Agent,
    prompt: str,
    *,
    role: str,
    question_id: str,
    max_function_calls: int | None = None,
) -> str:
    invocation_kwargs = (
        {"max_function_calls": max_function_calls}
        if max_function_calls is not None
        else None
    )
    with Timer() as t:
        try:
            resp = await agent.run(prompt, function_invocation_kwargs=invocation_kwargs)
        except Exception as e:
            pipeline.tracker.record(
                role=role,
                spec=pipeline.spec,
                input_tokens=0,
                output_tokens=0,
                latency_s=t.elapsed_s,
                question_id=question_id,
                error=f"{type(e).__name__}: {e}",
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
    )
    return resp.text or ""


async def run_pipeline(
    pipeline: ResearchPipeline,
    question: str,
    *,
    question_id: str,
    max_sub_queries: int = 4,
    max_tool_calls_per_query: int = 8,
) -> PipelineResult:
    """Run one question end-to-end through Manager -> Browser* -> Synthesizer."""
    result = PipelineResult(question_id=question_id, question=question)

    manager_out = await _run_agent(
        pipeline, pipeline.manager, question, role="manager", question_id=question_id
    )
    result.sub_questions = parse_sub_questions(
        manager_out, max_sub_queries=max_sub_queries
    )

    for sub_q in result.sub_questions:
        finding = await _run_agent(
            pipeline,
            pipeline.browser,
            sub_q,
            role="browser",
            question_id=question_id,
            max_function_calls=max_tool_calls_per_query,
        )
        result.findings.append(finding)

    synth_prompt = f"Original question: {question}\n\n" + "\n\n".join(
        f"Sub-question: {sq}\nFindings: {f}"
        for sq, f in zip(result.sub_questions, result.findings, strict=True)
    )
    result.final_answer = await _run_agent(
        pipeline,
        pipeline.synthesizer,
        synth_prompt,
        role="synthesizer",
        question_id=question_id,
    )
    return result
