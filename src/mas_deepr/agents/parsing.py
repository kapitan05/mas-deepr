"""Shared parsing for the Manager role's sub-question output.

Both the MAF pipeline (agents/topology.py) and the DSPy-compiled pipeline
(optimize/modules.py) produce the same plain-text numbered-list format for
sub-questions, so they share this parser rather than each reimplementing it.
"""

import re

_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)]|-)\s*(.+)$")


def parse_sub_questions(text: str, *, max_sub_queries: int) -> list[str]:
    lines = [_NUMBERED_LINE.match(line) for line in text.strip().splitlines()]
    parsed = [m.group(1).strip() for m in lines if m]
    if not parsed:
        parsed = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return parsed[:max_sub_queries] or [text.strip()]


# --- Leaked tool-call text (C1) ---------------------------------------------
#
# Qwen-family models emit tool calls as ``<tool_call>{json}</tool_call>``.
# When the serving endpoint's tool parser doesn't run on a response -- most
# commonly MAF's final ``tool_choice="none"`` call after the Browser has used
# up ``max_function_calls`` (see ``_disable_tools_at_function_call_limit`` /
# the equivalent branch in ``agent_framework/_tools.py``) -- that markup comes
# back as plain assistant *text* instead of a real tool call, and would
# otherwise flow into findings/final answers verbatim.

_TOOL_CALL_OPEN = "<tool_call"
_CLOSED_TOOL_CALL = re.compile(r"<tool_call\b[^>]*>.*?</tool_call>", re.DOTALL)
_UNCLOSED_TOOL_CALL = re.compile(r"<tool_call\b.*\Z", re.DOTALL)
_STRAY_CLOSING_TAG = re.compile(r"</tool_call>")

# Passed downstream instead of an empty string when a leaked finding had
# nothing left after cleanup -- tells the Synthesizer explicitly that this
# sub-question's evidence is missing, rather than handing it silence.
LEAK_FALLBACK_FINDING = (
    "No answer found for this sub-question (the tool-call budget ran out "
    "before an answer was written)."
)


def has_leaked_tool_call(text: str) -> bool:
    """True if ``text`` contains raw tool-call markup (``<tool_call...``)."""
    return _TOOL_CALL_OPEN in text


def strip_leaked_tool_calls(text: str) -> str:
    """Remove raw ``<tool_call>...</tool_call>`` markup from ``text``.

    Handles closed blocks (one or several), a trailing unclosed block (a
    truncated generation -- everything from the opening tag on is dropped)
    and stray closing tags. Surrounding prose is kept. Returns ``""`` when
    nothing but markup was present; callers decide what to substitute.
    """
    cleaned = _CLOSED_TOOL_CALL.sub("", text)
    cleaned = _UNCLOSED_TOOL_CALL.sub("", cleaned)
    cleaned = _STRAY_CLOSING_TAG.sub("", cleaned)
    # Collapse the blank-line gaps a removed block leaves behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
