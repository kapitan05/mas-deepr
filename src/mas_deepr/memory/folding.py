"""Context-Folding: sibling sub-questions see each other's findings.

Today's Manager->Browser* loop is already accidentally isomorphic to
Context-Folding's branch()/return() tools (each sub-question is a branch) --
just blind, since siblings never see each other. The only behavioral delta
this strategy adds is ``prepare_branch``, which renders prior sub-question ->
finding pairs as extra Browser context. v1 ``fold_branch`` is identity (no
new LLM call) -- cheapest strategy to build and run.
"""

from mas_deepr.memory.base import CompressFn, MainThreadState, PassRecord


class ContextFoldingStrategy:
    name = "folding"
    max_passes = 1

    async def prepare_pass(
        self, *, question: str, prior_passes: list[PassRecord], pass_index: int
    ) -> str:
        return ""

    async def prepare_branch(
        self, *, sub_question: str, main_thread: MainThreadState
    ) -> str:
        if not main_thread.folded:
            return ""
        lines = ["Findings from sibling sub-questions already answered:"]
        lines += [f"- Q: {sq}\n  A: {finding}" for sq, finding in main_thread.folded]
        return "\n".join(lines)

    async def fold_branch(
        self, *, sub_question: str, raw_finding: str, main_thread: MainThreadState
    ) -> str:
        main_thread.folded.append((sub_question, raw_finding))
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
        return PassRecord(
            pass_index=pass_index,
            sub_questions=sub_questions,
            findings=findings,
            final_answer=final_answer,
        )
