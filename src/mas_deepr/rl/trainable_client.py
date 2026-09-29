"""GRPO-only chat client: preserves tool-call logprobs and vLLM token IDs
MAF's stock client discards.

Confirmed directly against ``agent_framework_openai``'s source
(``_chat_completion_client.py``): ``_parse_text_from_openai`` attaches the
full OpenAI ``Choice`` (with ``.logprobs``) to text content;
``_parse_tool_calls_from_openai`` reads the exact same ``choice`` object but
keeps only ``{name, arguments}``, discarding the ``Choice``. For a pure
tool-call turn (no text at all -- the Browser's typical tool-selection
step, confirmed live: content="" + one function_call), nothing ends up
carrying that ``Choice`` forward, so "which tool to call" gets no gradient
even though ``choice.logprobs.content`` already holds the exact per-token
logprobs for the tool-call-arguments tokens the model generated. This is
not an API limitation -- OpenAI's ``logprobs`` field is one flat span over
everything generated in that turn, tool-call JSON included -- it's a
client-parsing choice, made once, here, undone for training use only.

Guarded to exactly one tool call per choice: a ``Choice``'s logprobs are
one shared span across everything generated in that turn, so attaching the
same ``Choice`` to two separately-modeled ``function_call`` ``Content``
items (parallel tool calls) would misattribute tokens between them --
falls back to the stock (undiscovered-Choice) behavior for that turn
instead, same as before this fix, never wrong, just not yet improved for
that turn.

Also confirmed live 2026-09-28 against a real ART ServerlessBackend
training run: ART rejects every trajectory with "Trainable Choice is
missing vLLM prompt_token_ids/token_ids" unless the vLLM endpoint was
called with ``extra_body={"return_token_ids": True}`` (a real vLLM
OpenAI-compatible extension, v0.10.2+ -- see ``agents/topology.py``'s
``_run_agent``, which sets this when ``capture_logprobs=True``) AND the
resulting ``Choice`` carries a ``prompt_token_ids`` attribute. The
response DOES include both fields (confirmed via a raw API call), but at
different nesting levels: ``token_ids`` is on each ``choice`` (survives
into ``raw_representation`` unmodified, since ``Choice``'s pydantic
config is ``extra="allow"``), while ``prompt_token_ids`` is a TOP-LEVEL
field on the ``ChatCompletion`` response, never copied down onto the
``Choice`` objects anywhere in MAF's own parsing -- so it was silently
lost before ever reaching a stored trajectory, regardless of the
tool-call fix above. ``_parse_response_from_openai`` below copies it onto
every choice before MAF's stock parsing runs, so it travels along
wherever ``raw_representation=choice`` already does (both the stock
text-content path and the ``_parse_tool_calls_from_openai`` override
above use the same object reference).

Used only by ``rl/rollout.py``'s GRPO rollouts (via
``build_pipeline(..., trainable=True)``) -- eval/baseline pipelines have no
use for trainable tool-call spans and keep using MAF's stock
``OpenAIChatCompletionClient`` via ``llm/factory.py``.
"""

from collections.abc import Mapping
from typing import Any

from agent_framework import ChatResponse, Content
from agent_framework.openai import OpenAIChatCompletionClient
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice


class TrainableOpenAIChatCompletionClient(OpenAIChatCompletionClient):
    """Identical to ``OpenAIChatCompletionClient``, except a pure-tool-call
    turn's ``Choice`` (real per-token logprobs included) is preserved on
    the resulting ``function_call`` content instead of discarded, and every
    ``Choice`` carries the response's ``prompt_token_ids`` -- see module
    docstring for why both are needed for ART training to accept a
    trajectory at all."""

    def _parse_response_from_openai(
        self, response: ChatCompletion, options: Mapping[str, Any]
    ) -> ChatResponse:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            for choice in response.choices:
                choice.prompt_token_ids = prompt_token_ids  # type: ignore[attr-defined]
        return super()._parse_response_from_openai(response, options)

    def _parse_tool_calls_from_openai(
        self, choice: "Choice | ChunkChoice"
    ) -> list[Content]:
        contents = super()._parse_tool_calls_from_openai(choice)
        if len(contents) != 1 or not isinstance(choice, Choice):
            # 0 tool calls: nothing to attach. >1 (parallel calls) or a
            # streaming ChunkChoice: ambiguous/incomplete span -- see
            # module docstring. Leave MAF's stock behavior unchanged.
            return contents
        call = contents[0]
        return [
            Content.from_function_call(
                call.call_id or "",
                call.name or "",
                arguments=call.arguments,
                raw_representation=choice,
            )
        ]
