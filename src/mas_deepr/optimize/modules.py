"""DSPy program mirroring the MAF Manager -> Browser -> Synthesizer pipeline.

The Browser role does NOT use ``dspy.ReAct``: ReAct's compiled instructions
bake in DSPy's own tool-selection text protocol (next_thought/next_tool_name/
next_tool_args), which has no equivalent in MAF's native function-calling
``Agent`` and would not port back to the prompt registry as a usable system
prompt. Instead, retrieval is a plain synchronous step -- the same
``web_search``/``fetch_page`` MCP tools MAF's browser calls, just invoked
imperatively here -- and DSPy only optimizes the "reason over gathered
evidence" signature, which *is* directly portable.
"""

import asyncio
from collections.abc import Callable

import dspy

from mas_deepr.agents.parsing import parse_sub_questions
from mas_deepr.config import Settings
from mas_deepr.optimize.signatures import (
    BrowserSignature,
    ManagerSignature,
    SynthesizerSignature,
)
from mas_deepr.tools import MCPToolClient

Retriever = Callable[[str], str]


def make_retriever(
    *, settings: Settings, top_k: int = 3, fetch_top: int = 1
) -> Retriever:
    """Search + fetch a sub-question's evidence, synchronously.

    Calls the same MCP tools MAF's browser calls (via ``MCPToolClient``, not
    in-process), blocking via ``asyncio.run`` since DSPy's compile loop
    (MIPROv2) evaluates the program synchronously.

    Two real bugs, both confirmed live 2026-09-26 against a real MCP
    server, fixed here:

    1. This used to hardcode ``"web_search"`` regardless of
       ``settings.mcp_enabled_tools`` -- ``compile_dspy.py``'s wiki_paper
       scope (dropping ``web_search`` specifically to avoid hammering
       SearXNG at compile-time rollout volume, see
       ``config/tool_scopes.py``) was silently never honored here. Now
       picks whichever of ``web_search``/``wikipedia_search`` is actually
       enabled -- ``web_search`` preferred when available (unchanged
       behavior for general-scope callers), ``wikipedia_search`` as the
       wiki_paper-scope fallback. Raises clearly if neither is enabled
       rather than silently returning empty results.
    2. This used to accept one shared ``MCPToolClient`` and call
       ``asyncio.run(mcp_client.call(...))`` separately for the search AND
       the fetch -- two separate ``asyncio.run`` calls, each creating a
       fresh event loop, reusing the SAME client (and therefore the same
       ``asyncio.Semaphore``s, bound to whichever loop existed when they
       were constructed). MIPROv2 evaluates examples across multiple
       threads/loops in parallel, so this raised "RuntimeError: ... is
       bound to a different event loop" on effectively every call --
       confirmed live: 6,728 failures, 100% of search calls in one real
       compile run, meaning every sub-question silently retrieved "No
       search results found" for that entire run. Fixed by doing exactly
       one ``asyncio.run`` per ``_retrieve`` call, constructing a fresh
       ``MCPToolClient`` inside that same async function -- its semaphores
       now always bind to the loop they're actually used in, and search +
       fetch share one loop instead of each getting a different one.
    """
    enabled = settings.mcp_enabled_tools_set()
    if "web_search" in enabled:
        search_tool = "web_search"
    elif "wikipedia_search" in enabled:
        search_tool = "wikipedia_search"
    else:
        raise ValueError(
            "make_retriever needs 'web_search' or 'wikipedia_search' enabled "
            f"(mcp_enabled_tools={settings.mcp_enabled_tools!r}) -- neither is "
            "available, nothing this retriever can search with."
        )

    def _retrieve(query: str) -> str:
        async def _do_retrieve() -> str:
            mcp_client = MCPToolClient(
                url=settings.mcp_server_url,
                transport=settings.mcp_transport,
                concurrency=settings.mcp_tool_concurrency,
                searxng_concurrency=settings.mcp_searxng_concurrency,
            )
            hits = await mcp_client.call(search_tool, query=query, max_results=top_k)
            if not hits or isinstance(hits, str):  # str = an inline failure marker
                return "No search results found."

            blocks = []
            for i, hit in enumerate(hits):
                block = f"[{i + 1}] {hit['title']} ({hit['url']})\n{hit['snippet']}"
                if i < fetch_top:
                    page_text = await mcp_client.call(
                        "fetch_page",
                        url=hit["url"],
                        timeout_s=settings.fetch_timeout_s,
                        max_chars=settings.fetch_max_chars,
                    )
                    block += f"\nFetched content: {page_text}"
                blocks.append(block)
            return "\n\n".join(blocks)

        return asyncio.run(_do_retrieve())

    return _retrieve


class ResearchProgram(dspy.Module):
    """DSPy counterpart to ``agents.topology.run_pipeline``.

    Named submodules ``manager``, ``browser``, ``synthesizer`` are exactly
    what ``optimize.compile`` extracts optimized instructions/demos from
    after ``MIPROv2.compile`` -- keep these names in sync with
    ``optimize.render``'s field-label maps if the signatures change.
    """

    def __init__(self, *, retrieve: Retriever, max_sub_queries: int = 4) -> None:
        super().__init__()
        self.manager = dspy.Predict(ManagerSignature)
        self.browser = dspy.Predict(BrowserSignature)
        self.synthesizer = dspy.Predict(SynthesizerSignature)
        self._retrieve = retrieve
        self.max_sub_queries = max_sub_queries

    def forward(self, question: str) -> dspy.Prediction:
        manager_out = self.manager(question=question)
        sub_questions = parse_sub_questions(
            manager_out.sub_questions, max_sub_queries=self.max_sub_queries
        )

        findings = []
        for sub_q in sub_questions:
            context = self._retrieve(sub_q)
            browser_out = self.browser(sub_question=sub_q, context=context)
            findings.append(browser_out.finding)

        findings_text = "\n\n".join(
            f"Sub-question: {sq}\nFindings: {f}"
            for sq, f in zip(sub_questions, findings, strict=True)
        )
        synth_out = self.synthesizer(question=question, findings=findings_text)

        return dspy.Prediction(
            final_answer=synth_out.final_answer,
            sub_questions=sub_questions,
            findings=findings,
        )
