"""Token-budget guard for the 32k context window (B1).

Hard multi-hop questions are exactly the ones whose Browser tool loop runs
longest -- and every ``fetch_page``/search result stays in MAF's running
history, so somewhere around tool call 6-8 the request crosses the served
model's context window and vLLM rejects it. The question then dies (eval:
``metric="error"``, score 0; GRPO: an exception that drops the rollout),
which is the "hard trajectories never get signal" failure mode.

Two mechanisms, both off when ``ResearchPipeline.context_budget is None``:

1. ``ContextBudgetMiddleware`` -- a MAF ``ChatMiddleware``. MAF's
   ``FunctionInvocationLayer`` wraps ``ChatMiddlewareLayer``, so this runs
   before *every* model call inside the tool loop, not once per
   ``agent.run()`` (confirmed against agent-framework-core 1.11). When the
   estimated request exceeds the budget it shortens tool results, oldest
   first, keeping each one's head and tail around a ``[... N chars
   truncated ...]`` marker. Results are processed strictly oldest-first and
   each is cut down to ``min_result_chars`` before the next is touched, so
   the newest results (what the model is about to read) are shortened only
   as a last resort.
   Tool-call/tool-result pairs are never removed, so the chat template
   stays valid.

   RULE -- new ``Message`` objects, never edit in place: ``ChatContext``
   holds a *shallow copy* of the loop's message list, i.e. the very same
   ``Message``/``Content`` objects MAF keeps in its history and later
   returns as ``resp.messages``. Mutating them would permanently erase the
   full evidence that C1 recovery (``_collect_tool_evidence``), folding
   memory (``BranchTrajectory.messages``) and debugging read afterwards,
   and would compound across loop steps. Replacing them with shortened
   copies changes only what this one request sends. Guarded by
   ``tests/test_context_budget.py``.

   GRPO is unaffected by the trimming: ART 0.5.20 trains each ``Choice`` on
   vLLM's own ``prompt_token_ids`` and starts a new training sequence
   whenever a prompt doesn't extend the previous one
   (``art/preprocessing/tokenize.py::assemble_vllm_training_sequences``),
   so every completion is trained on exactly the context it was sampled
   under. Guarded by an ``art``-marked test.

2. ``truncate_middle`` caps, applied by ``run_pipeline`` to each Browser
   finding (before folding/the Synthesizer see it) and to memory-strategy
   prefixes -- user-message text the middleware deliberately never touches.

Token counts are estimates (``CHARS_PER_TOKEN`` = 3, deliberately more
conservative than ``llm/rate_limit.py``'s 4, since URLs, numbers and
non-English text tokenize worse than prose); ``headroom_tokens`` absorbs
the response plus estimation error.
"""

import contextvars
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agent_framework import ChatContext, ChatMiddleware, Content, Message

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 3
# Rough per-message chat-template overhead (role tags, separators).
_PER_MESSAGE_OVERHEAD_TOKENS = 8


def chars_to_tokens(n_chars: int) -> int:
    return -(-max(0, n_chars) // CHARS_PER_TOKEN)


def tokens_to_chars(n_tokens: int) -> int:
    return max(0, n_tokens) * CHARS_PER_TOKEN


def truncate_middle(text: str, max_chars: int, *, head_fraction: float = 0.7) -> str:
    """Shorten ``text`` to at most ``max_chars``, keeping its head and tail.

    Head-heavy by default (titles/lead paragraphs carry most of a page's
    answer), but a tail is kept too since findings tend to end with their
    source URLs. The marker counts toward ``max_chars``.
    """
    if len(text) <= max_chars:
        return text
    omitted_guess = len(text) - max_chars
    marker = f"\n[... {omitted_guess} chars truncated ...]\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return text[: max(0, max_chars)]
    head = int(keep * head_fraction)
    tail = keep - head
    omitted = len(text) - head - tail
    marker = f"\n[... {omitted} chars truncated ...]\n"
    return text[:head] + marker + (text[len(text) - tail :] if tail > 0 else "")


@dataclass
class ContextBudget:
    """Budget settings; built from ``Settings`` in ``build_pipeline``."""

    window_tokens: int = 32768
    headroom_tokens: int = 4096
    min_result_chars: int = 600
    finding_max_tokens: int = 3000
    prefix_max_tokens: int = 6000

    @property
    def budget_tokens(self) -> int:
        return max(0, self.window_tokens - self.headroom_tokens)


@dataclass
class ContextBudgetStats:
    """Per-question counters (one per ``run_pipeline`` call, shared by every
    agent call inside it via ``current_budget_stats``)."""

    calls_checked: int = 0
    trimmed_calls: int = 0
    trimmed_results: int = 0
    chars_trimmed: int = 0
    # Still estimated over budget after trimming everything allowed -- the
    # call goes out anyway (and may still be rejected); a non-zero value
    # means the budget knobs need adjusting, not that the guard failed
    # silently.
    over_budget_calls: int = 0
    max_estimated_tokens: int = 0
    findings_capped: int = 0
    prefixes_capped: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


# Middleware instances are built once per pipeline and shared by every
# concurrently-running question, so per-question counters travel via a
# ContextVar set in run_pipeline (same pattern as tool_telemetry's
# current_question_id).
current_budget_stats: contextvars.ContextVar[ContextBudgetStats | None] = (
    contextvars.ContextVar("current_budget_stats", default=None)
)


@dataclass
class TrimReport:
    estimated_tokens_before: int
    estimated_tokens_after: int
    trimmed_results: int
    chars_trimmed: int
    over_budget: bool


def _result_text(content: Any) -> str:
    items = getattr(content, "items", None)
    if items:
        return "\n".join(
            getattr(i, "text", None) or ""
            for i in items
            if getattr(i, "type", None) == "text"
        )
    result = getattr(content, "result", None)
    return "" if result is None else str(result)


def _content_chars(content: Any) -> int:
    ctype = getattr(content, "type", None)
    if ctype == "function_result":
        return len(_result_text(content))
    if ctype == "function_call":
        return len(str(getattr(content, "name", "") or "")) + len(
            str(getattr(content, "arguments", "") or "")
        )
    return len(str(getattr(content, "text", "") or ""))


def _tool_chars(tool: Any) -> int:
    spec = getattr(tool, "to_json_schema_spec", None)
    if callable(spec):
        try:
            return len(json.dumps(spec()))
        except Exception:
            pass
    return len(str(getattr(tool, "name", ""))) + len(
        str(getattr(tool, "description", ""))
    )


def estimate_request_tokens(
    messages: Sequence[Any], options: Mapping[str, Any] | None
) -> int:
    """Conservative token estimate of one chat request: system instructions
    and tool schemas (which MAF passes in ``options``, not ``messages``)
    plus every message's contents."""
    opts = options or {}
    chars = len(str(opts.get("instructions") or ""))
    chars += sum(_tool_chars(t) for t in (opts.get("tools") or []))
    chars += sum(
        _content_chars(c)
        for m in messages
        for c in (getattr(m, "contents", None) or [])
    )
    return chars_to_tokens(chars) + _PER_MESSAGE_OVERHEAD_TOKENS * len(messages)


def _copy_with_contents(message: Any, contents: list[Any]) -> Message:
    return Message(
        role=message.role,
        contents=contents,
        author_name=getattr(message, "author_name", None),
        message_id=getattr(message, "message_id", None),
        additional_properties=getattr(message, "additional_properties", None),
        raw_representation=getattr(message, "raw_representation", None),
    )


def trim_messages(
    messages: Sequence[Any],
    options: Mapping[str, Any] | None,
    budget: ContextBudget,
) -> tuple[list[Any], TrimReport]:
    """Return a request-ready message list that fits ``budget``, if possible.

    Never mutates ``messages`` or anything inside it: shortened tool results
    go into *new* ``Message``/``Content`` objects; untouched messages are
    passed through as the same objects.
    """
    before = estimate_request_tokens(messages, options)
    if before <= budget.budget_tokens:
        return list(messages), TrimReport(before, before, 0, 0, False)

    excess = tokens_to_chars(before - budget.budget_tokens)
    positions = [
        (mi, ci)
        for mi, m in enumerate(messages)
        for ci, c in enumerate(getattr(m, "contents", None) or [])
        if getattr(c, "type", None) == "function_result"
    ]
    # Chronological = oldest first; each result goes all the way to the
    # floor before the next is touched, so the newest are cut last.
    replacements: dict[tuple[int, int], str] = {}
    for mi, ci in positions:
        if excess <= 0:
            break
        text = _result_text(messages[mi].contents[ci])
        if len(text) <= budget.min_result_chars:
            continue
        target = max(budget.min_result_chars, len(text) - excess)
        shortened = truncate_middle(text, target)
        saved = len(text) - len(shortened)
        if saved <= 0:
            continue
        replacements[(mi, ci)] = shortened
        excess -= saved

    if not replacements:
        return list(messages), TrimReport(before, before, 0, 0, True)

    new_messages: list[Any] = []
    for mi, m in enumerate(messages):
        if not any(k[0] == mi for k in replacements):
            new_messages.append(m)
            continue
        new_contents: list[Any] = []
        for ci, c in enumerate(m.contents):
            if (mi, ci) in replacements:
                new_contents.append(
                    Content.from_function_result(
                        c.call_id,
                        result=replacements[(mi, ci)],
                        exception=getattr(c, "exception", None),
                        additional_properties=getattr(c, "additional_properties", None),
                    )
                )
            else:
                new_contents.append(c)
        new_messages.append(_copy_with_contents(m, new_contents))

    after = estimate_request_tokens(new_messages, options)
    chars_trimmed = sum(
        len(_result_text(messages[mi].contents[ci])) - len(t)
        for (mi, ci), t in replacements.items()
    )
    return new_messages, TrimReport(
        before, after, len(replacements), chars_trimmed, after > budget.budget_tokens
    )


class ContextBudgetMiddleware(ChatMiddleware):
    """Keeps every model request under ``budget`` -- see module docstring."""

    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget

    async def process(self, context: ChatContext, call_next: Any) -> None:
        new_messages, report = trim_messages(
            list(context.messages), context.options, self.budget
        )
        if report.trimmed_results:
            context.messages = new_messages
            logger.info(
                "context guard trimmed %d tool result(s), %d chars: est %d -> "
                "%d tokens (budget %d)",
                report.trimmed_results,
                report.chars_trimmed,
                report.estimated_tokens_before,
                report.estimated_tokens_after,
                self.budget.budget_tokens,
            )
        if report.over_budget:
            logger.warning(
                "context guard: request still over budget after trimming "
                "(est %d tokens, budget %d)",
                report.estimated_tokens_after,
                self.budget.budget_tokens,
            )
        stats = current_budget_stats.get()
        if stats is not None:
            stats.calls_checked += 1
            stats.max_estimated_tokens = max(
                stats.max_estimated_tokens, report.estimated_tokens_before
            )
            if report.trimmed_results:
                stats.trimmed_calls += 1
                stats.trimmed_results += report.trimmed_results
                stats.chars_trimmed += report.chars_trimmed
            if report.over_budget:
                stats.over_budget_calls += 1
        await call_next()
