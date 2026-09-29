"""Explicit no-op strategy -- byte-identical to pre-Phase-3 behavior.

This is the regression target for the whole memory module:
``run_pipeline(..., memory=None)`` and ``run_pipeline(..., memory=
StatelessStrategy())`` must produce identical ``PipelineResult`` output and
identical telemetry call counts. See ``tests/test_memory.py``.
"""

from mas_deepr.memory.base import CompressFn, MainThreadState, PassRecord


class StatelessStrategy:
    name = "stateless"
    max_passes = 1

    async def prepare_pass(
        self, *, question: str, prior_passes: list[PassRecord], pass_index: int
    ) -> str:
        return ""

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
        return PassRecord(
            pass_index=pass_index,
            sub_questions=sub_questions,
            findings=findings,
            final_answer=final_answer,
        )
