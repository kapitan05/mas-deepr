"""Tool-call guard: make tool calls either really happen or not happen at all.

Replaces "strip the leaked markup afterwards" (C1) as the primary fix. Two
distinct failure modes, both confirmed against agent-framework-core 1.11:

1. Budget exhausted. Once ``function_invocation_configuration
   ["max_function_calls"]`` is reached, MAF makes one last model call with
   ``tool_choice="none"`` -- but keeps the tools in the request, so the chat
   template still shows them and the model writes a tool call as plain text
   anyway (the original Qwen ``<tool_call>`` leak). MAF never executes it,
   but the text becomes the "answer". Fix: on that turn this middleware
   removes ``tools``/``tool_choice`` from the request entirely and appends
   one instruction to answer now. No tool is offered, so none can be
   requested, and MAF executes nothing on that turn regardless.

2. Unparsed tool calls. A model can emit perfectly valid tool calls as text
   that the serving endpoint's parser doesn't recognise -- confirmed live
   with Llama 3.1 8B on W&B Inference: ``{"name": "fetch_page",
   "parameters": {...}}; {"name": ...}`` on the Browser's very first call,
   so nothing was searched and the JSON became the finding. Fix: when tools
   *are* offered and the reply is nothing but such calls to tools that are
   actually attached, rewrite them into real ``function_call`` contents.
   MAF's FunctionInvocationLayer (which wraps this middleware) then executes
   them and counts them against the budget like any other call. The
   original text content's ``raw_representation`` (the logprob/token-id
   bearing OpenAI ``Choice``) is carried onto the first converted call, so
   GRPO still trains on exactly the tokens the policy generated.

Formats recognised: Qwen ``<tool_call>{json}</tool_call>`` blocks (prose
around them allowed -- the tags are unambiguous) and Llama bare JSON
objects ``{"name": ..., "parameters"|"arguments": {...}}`` separated by
``;``/whitespace, optionally prefixed by ``<|python_tag|>`` (the whole reply
must be calls -- an answer that merely contains JSON is never converted).
Conversion is all-or-nothing: one unknown tool name or malformed object and
the reply is left untouched for the C1 safety net in ``run_pipeline``.

3. One call per turn (``single_tool_call_per_turn=True``, set per model via
   ``ModelSpec``). Some chat templates -- Llama 3.1's official one -- can't
   render an assistant turn with several tool calls: the request *after*
   such a turn fails with a 400. For those models any multi-call reply
   (text-converted or natively parsed) keeps only its first call. The model
   sees that call's result next turn and asks for the next one itself --
   the one-call-per-turn rhythm the template was built for.

History rule (same as context_budget.py): request-side changes go into new
lists/dicts, never into objects MAF keeps in its loop history.
"""

import contextvars
import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from agent_framework import ChatContext, ChatMiddleware, Content, Message

logger = logging.getLogger(__name__)

NO_TOOLS_LEFT_INSTRUCTION = (
    "You have no tool calls left. Do not call or mention any tools. Write your "
    "answer now using only the evidence above, and list the source URLs you "
    "relied on."
)

_QWEN_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_PYTHON_TAG = "<|python_tag|>"
_SEPARATORS = " \t\r\n;,"


@dataclass
class ToolGuardStats:
    """Per-question counters, shared via ``current_tool_guard_stats``."""

    final_turns_without_tools: int = 0
    text_tool_calls_converted: int = 0
    replies_converted: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


current_tool_guard_stats: contextvars.ContextVar[ToolGuardStats | None] = (
    contextvars.ContextVar("current_tool_guard_stats", default=None)
)


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, Mapping):
        fn = tool.get("function")
        name = fn.get("name") if isinstance(fn, Mapping) else tool.get("name")
        return str(name) if name else None
    name = getattr(tool, "name", None)
    return str(name) if name else None


def _call_from_obj(obj: Any, tool_names: set[str]) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or name not in tool_names:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return name, args


def parse_text_tool_calls(
    text: str, tool_names: set[str]
) -> list[tuple[str, dict[str, Any]]] | None:
    """Parse a reply that consists of text-form tool calls.

    Returns the ``(name, arguments)`` list, or ``None`` if the text isn't
    (entirely) valid calls to tools in ``tool_names``.
    """
    if not text.strip() or not tool_names:
        return None

    blocks = _QWEN_BLOCK.findall(text)
    if blocks:
        calls = []
        for block in blocks:
            try:
                call = _call_from_obj(json.loads(block), tool_names)
            except json.JSONDecodeError:
                return None
            if call is None:
                return None
            calls.append(call)
        return calls

    body = text.strip()
    if body.startswith(_PYTHON_TAG):
        body = body[len(_PYTHON_TAG) :]
    decoder = json.JSONDecoder()
    calls = []
    i = 0
    while True:
        while i < len(body) and body[i] in _SEPARATORS:
            i += 1
        if i >= len(body):
            break
        if body[i] != "{":
            return None  # prose -- not a pure tool-call reply
        try:
            obj, end = decoder.raw_decode(body, i)
        except json.JSONDecodeError:
            return None
        call = _call_from_obj(obj, tool_names)
        if call is None:
            return None
        calls.append(call)
        i = end
    return calls or None


def _copy_message(message: Any, contents: list[Any]) -> Message:
    return Message(
        role="assistant",
        contents=contents,
        author_name=getattr(message, "author_name", None),
        message_id=getattr(message, "message_id", None),
        additional_properties=getattr(message, "additional_properties", None),
        raw_representation=getattr(message, "raw_representation", None),
    )


def _keep_first_call(message: Any) -> Message | None:
    """Copy of ``message`` with only its first ``function_call`` content,
    or ``None`` if it has at most one."""
    contents = list(getattr(message, "contents", None) or [])
    calls = [c for c in contents if getattr(c, "type", None) == "function_call"]
    if len(calls) <= 1:
        return None
    first = calls[0]
    kept = [
        c for c in contents if getattr(c, "type", None) != "function_call" or c is first
    ]
    return _copy_message(message, kept)


def _convert_reply(
    message: Any, tool_names: set[str], *, max_calls: int | None = None
) -> tuple[Message, int] | None:
    contents = list(getattr(message, "contents", None) or [])
    if any(getattr(c, "type", None) == "function_call" for c in contents):
        return None  # the endpoint already parsed real calls
    text_contents = [c for c in contents if getattr(c, "type", None) == "text"]
    if not text_contents:
        return None
    text = "".join(str(getattr(c, "text", "") or "") for c in text_contents)
    calls = parse_text_tool_calls(text, tool_names)
    if not calls:
        return None
    if max_calls is not None:
        calls = calls[:max_calls]
    raw = next(
        (
            getattr(c, "raw_representation", None)
            for c in text_contents
            if getattr(c, "raw_representation", None) is not None
        ),
        None,
    )
    new_contents = [
        Content.from_function_call(
            call_id=f"textcall_{uuid.uuid4().hex[:12]}",
            name=name,
            arguments=args,
            raw_representation=raw if i == 0 else None,
        )
        for i, (name, args) in enumerate(calls)
    ]
    # Keep any non-text contents (e.g. reasoning) the reply carried.
    new_contents = [
        c for c in contents if getattr(c, "type", None) not in ("text",)
    ] + new_contents
    return _copy_message(message, new_contents), len(calls)


class ToolCallGuardMiddleware(ChatMiddleware):
    """See module docstring. Attach to every role that has tools."""

    def __init__(self, *, single_tool_call_per_turn: bool = False) -> None:
        self.single_tool_call_per_turn = single_tool_call_per_turn

    async def process(self, context: ChatContext, call_next: Any) -> None:
        options: Mapping[str, Any] = context.options or {}
        tools: Sequence[Any] = options.get("tools") or []
        stats = current_tool_guard_stats.get()

        if tools and options.get("tool_choice") == "none":
            # 1. Budget used up: send no tools at all, plus an explicit nudge.
            new_options = {
                k: v for k, v in options.items() if k not in ("tools", "tool_choice")
            }
            context.options = new_options
            context.messages = [
                *context.messages,
                Message(role="user", contents=[NO_TOOLS_LEFT_INSTRUCTION]),
            ]
            if stats is not None:
                stats.final_turns_without_tools += 1
            logger.info("tool budget used up: final call sent with no tools offered")
            await call_next()
            return

        await call_next()

        # 2. Tools were offered: turn unparsed text-form calls into real ones.
        response = context.result
        messages = getattr(response, "messages", None)
        if not tools or not messages:
            return
        tool_names = {n for n in (_tool_name(t) for t in tools) if n}
        max_calls = 1 if self.single_tool_call_per_turn else None
        # A fresh response object from the inner client, not loop history --
        # replacing its last message is safe.
        converted = _convert_reply(messages[-1], tool_names, max_calls=max_calls)
        if converted is not None:
            new_message, n_calls = converted
            messages[-1] = new_message
            if stats is not None:
                stats.replies_converted += 1
                stats.text_tool_calls_converted += n_calls
            logger.info("converted %d text-form tool call(s) into real calls", n_calls)
        elif self.single_tool_call_per_turn:
            # 3. Natively parsed multi-call reply on a one-call-per-turn model.
            trimmed = _keep_first_call(messages[-1])
            if trimmed is not None:
                messages[-1] = trimmed
