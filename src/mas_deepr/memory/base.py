"""Memory/context-strategy protocol and shared data shapes.

Two hook scopes on one protocol, not two interfaces: RE-TRAC-style
strategies operate across N full ``run_pipeline`` passes (outer loop);
Context-Folding-style strategies operate across the Browser's per-sub-
question branches within one pass (inner loop). A strategy that needed both
scopes at once (e.g. a causal-graph strategy updating on every branch return
while also consulting/pruning across passes) would be forced into two
unrelated classes under a two-interface split -- one Protocol with
sensible no-op defaults avoids that. ``StatelessStrategy``
(memory/baseline.py) implements every hook as an identity/no-op and is the
regression test that this abstraction is lossless: ``run_pipeline`` with
``memory=None`` and with ``memory=StatelessStrategy()`` must produce
byte-identical ``PipelineResult`` output.

Deliberately import-free of ``agents.topology``/``agent_framework`` --
strategies never touch ``Agent``/telemetry directly. Any strategy that needs
an extra LLM call (RE-TRAC's compressor) is handed a plain
``prompt -> response`` async callback (``CompressFn``) by ``run_pipeline``,
which does the actual agent/telemetry wiring. This keeps memory/ testable
with no MAF or network dependency, matching the rest of the registries.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mas_deepr.memory.state import StructuredState

CompressFn = Callable[[str], Awaitable[str]]


@dataclass
class BranchTrajectory:
    """Everything captured from one Browser call for one sub-question."""

    sub_question: str
    raw_finding: str
    messages: list[object] = field(default_factory=list)


@dataclass
class MainThreadState:
    """State visible across sibling branches within one pass.

    ``folded`` accumulates (sub_question, folded_finding) pairs as branches
    complete -- this is the concrete thing Context-Folding adds that today's
    pipeline lacks (siblings currently never see each other).
    """

    question: str
    folded: list[tuple[str, str]] = field(default_factory=list)
    branches: list[BranchTrajectory] = field(default_factory=list)


@dataclass
class PassRecord:
    """One completed pass's outcome, handed to the next pass as context.

    ``pass_index`` is set by ``run_pipeline`` after ``finalize_pass``
    returns, not by the strategy itself, so strategies don't need to track
    their own position in the loop.
    """

    pass_index: int
    sub_questions: list[str]
    findings: list[str]
    final_answer: str
    state: "StructuredState | None" = None


class MemoryStrategy(Protocol):
    """Pluggable context/memory strategy hooking into ``run_pipeline``.

    ``max_passes`` drives the outer pass loop (1 for strategies that only
    touch the inner branch loop, e.g. Folding; 4-8 for RE-TRAC). Every hook
    below has a no-op-shaped implementation in ``StatelessStrategy`` -- a
    strategy overrides only what it needs.
    """

    name: str
    max_passes: int

    async def prepare_pass(
        self, *, question: str, prior_passes: list[PassRecord], pass_index: int
    ) -> str:
        """Extra context prepended to the Manager prompt for this pass."""
        ...

    async def prepare_branch(
        self, *, sub_question: str, main_thread: MainThreadState
    ) -> str:
        """Extra context prepended to the Browser prompt for one sub-question."""
        ...

    async def fold_branch(
        self, *, sub_question: str, raw_finding: str, main_thread: MainThreadState
    ) -> str:
        """Post-process one branch's raw finding before it feeds the synthesizer."""
        ...

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
        """Build this pass's ``PassRecord`` once Manager->Browser*->Synthesizer
        finishes."""
        ...
