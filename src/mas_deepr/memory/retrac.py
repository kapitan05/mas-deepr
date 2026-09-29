"""RE-TRAC: recursive trajectory -> structured-state compression across passes.

Training-free mode: after each pass, the ``compressor`` role condenses the
full trajectory into a ``StructuredState``; the next pass's Manager prompt is
prefixed with that state rendered as a "treat as hypothesis, verify or
override" instruction. Most expensive strategy to *run* (N x LLM calls per
question, via ``max_passes``) but needs zero training.
"""

from mas_deepr.memory.base import CompressFn, MainThreadState, PassRecord
from mas_deepr.memory.state import parse_structured_state

_COMPRESS_PROMPT = """
Question: {question}

This pass's sub-questions and findings:
{trajectory}

This pass's final answer: {final_answer}

Compress the above into a JSON object with keys: answer (string), evidence
(list of short strings), analysis (string), uncertainties (list of short
strings), failed_attempts (list of short strings). Respond with ONLY the
JSON object, no prose before or after.
""".strip()


class TrajectoryCompressionStrategy:
    name = "retrac"

    def __init__(self, max_passes: int = 4) -> None:
        self.max_passes = max_passes

    async def prepare_pass(
        self, *, question: str, prior_passes: list[PassRecord], pass_index: int
    ) -> str:
        if not prior_passes:
            return ""
        state = prior_passes[-1].state
        return state.render() if state else ""

    async def prepare_branch(
        self, *, sub_question: str, main_thread: MainThreadState
    ) -> str:
        return ""

    async def fold_branch(
        self, *, sub_question: str, raw_finding: str, main_thread: MainThreadState
    ) -> str:
        return raw_finding

    async def finalize_pass(
        self,
        *,
        final_answer: str,
        sub_questions: list[str],
        findings: list[str],
        main_thread: MainThreadState,
        pass_index: int,
        compress: CompressFn,
    ) -> PassRecord:
        trajectory = (
            "\n".join(
                f"Sub-question: {sq}\nFinding: {f}"
                for sq, f in zip(sub_questions, findings, strict=True)
            )
            or "(no sub-questions this pass)"
        )
        prompt = _COMPRESS_PROMPT.format(
            question=main_thread.question,
            trajectory=trajectory,
            final_answer=final_answer,
        )
        raw = await compress(prompt)
        state = parse_structured_state(raw)
        return PassRecord(
            pass_index=pass_index,
            sub_questions=sub_questions,
            findings=findings,
            final_answer=final_answer,
            state=state,
        )
