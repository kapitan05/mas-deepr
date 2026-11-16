"""Shared parsing for the Manager role's sub-question output.

Both the MAF pipeline (agents/topology.py) and the DSPy-compiled pipeline
(optimize/modules.py) produce the same plain-text numbered-list format for
sub-questions, so they share this parser rather than each reimplementing it.
"""

import json
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

# Llama-3.x-family models write tool calls as bare JSON objects instead --
# ``{"name": "fetch_page", "parameters": {...}}`` (optionally preceded by
# ``"type": "function"`` and/or the ``<|python_tag|>`` token), several in a
# row separated by ``;`` -- confirmed live with Llama 3.1 8B on W&B
# Inference. Normally agents/tool_call_guard.py turns these into real tool
# calls before they ever reach a finding; this is the safety net for
# whatever still slips through (e.g. a call to a tool that isn't attached).
_JSON_TOOL_CALL_START = re.compile(
    r'\{\s*(?:"type"\s*:\s*"function"\s*,\s*)?"name"\s*:\s*"[^"\n]{1,100}"\s*,'
    r'\s*"(?:parameters|arguments)"\s*:'
)
_PYTHON_TAG = "<|python_tag|>"

# Passed downstream instead of an empty string when a leaked finding had
# nothing left after cleanup -- tells the Synthesizer explicitly that this
# sub-question's evidence is missing, rather than handing it silence.
LEAK_FALLBACK_FINDING = (
    "No answer found for this sub-question (the tool-call budget ran out "
    "before an answer was written)."
)


def has_leaked_tool_call(text: str) -> bool:
    """True if ``text`` contains raw tool-call markup: Qwen ``<tool_call...``
    tags or Llama-style bare-JSON calls (see ``_JSON_TOOL_CALL_START``)."""
    return (
        _TOOL_CALL_OPEN in text
        or _PYTHON_TAG in text
        or _JSON_TOOL_CALL_START.search(text) is not None
    )


def _strip_json_tool_calls(text: str) -> str:
    """Remove Llama-style JSON tool-call objects (and the ``;`` between
    them); a truncated trailing object drops everything from its start."""
    decoder = json.JSONDecoder()
    out: list[str] = []
    i = 0
    while (m := _JSON_TOOL_CALL_START.search(text, i)) is not None:
        out.append(text[i : m.start()])
        try:
            _obj, end = decoder.raw_decode(text, m.start())
        except json.JSONDecodeError:
            end = len(text)
        while end < len(text) and text[end] in " \t;,":
            end += 1
        i = end
    out.append(text[i:])
    return "".join(out)


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
    cleaned = _strip_json_tool_calls(cleaned.replace(_PYTHON_TAG, ""))
    # Collapse the blank-line gaps a removed block leaves behind.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
