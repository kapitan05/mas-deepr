"""DSPy program mirroring the MAF Manager -> Browser -> Synthesizer pipeline.

The Browser role does NOT use ``dspy.ReAct``: ReAct's compiled instructions
bake in DSPy's own tool-selection text protocol (next_thought/next_tool_name/
next_tool_args), which has no equivalent in MAF's native function-calling
``Agent`` and would not port back to the prompt registry as a usable system
prompt. Instead, retrieval is a plain synchronous step -- the same
``web_search``/``fetch_page`` tools MAF's browser calls, just invoked
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
from mas_deepr.tools import WebCache, fetch_page, web_search

Retriever = Callable[[str], str]


def make_retriever(
    *, settings: Settings, cache: WebCache, top_k: int = 3, fetch_top: int = 1
) -> Retriever:
    """Search + fetch a sub-question's evidence, synchronously.

    Runs the same cached ``web_search``/``fetch_page`` coroutines MAF's
    browser tools use, blocking via ``asyncio.run`` since DSPy's compile
    loop (MIPROv2) evaluates the program synchronously.
    """

    def _retrieve(query: str) -> str:
        hits = asyncio.run(
            web_search(
                query,
                max_results=top_k,
                api_key=settings.tavily_api_key,
                cache=cache,
            )
        )
        if not hits:
            return "No search results found."

        blocks = []
        for i, hit in enumerate(hits):
            block = f"[{i + 1}] {hit['title']} ({hit['url']})\n{hit['snippet']}"
            if i < fetch_top:
                page_text = asyncio.run(
                    fetch_page(
                        hit["url"],
                        timeout_s=settings.fetch_timeout_s,
                        max_chars=settings.fetch_max_chars,
                        cache=cache,
                    )
                )
                block += f"\nFetched content: {page_text}"
            blocks.append(block)
        return "\n\n".join(blocks)

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
