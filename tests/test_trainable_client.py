"""``TrainableOpenAIChatCompletionClient``'s tool-call ``Choice`` preservation
and ``prompt_token_ids`` propagation.

Pure parsing-logic tests, no network -- constructs real OpenAI response
objects (``Choice``/``ChatCompletionMessage``/tool calls) and checks what
``_parse_tool_calls_from_openai``/``_parse_response_from_openai`` attach.
"""

import time

from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from mas_deepr.rl.trainable_client import TrainableOpenAIChatCompletionClient


def _tool_call_choice(*, n: int) -> Choice:
    calls: list[
        ChatCompletionMessageFunctionToolCall | ChatCompletionMessageCustomToolCall
    ] = [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{i}",
            type="function",
            function=Function(name="wikipedia_search", arguments='{"query": "x"}'),
        )
        for i in range(n)
    ]
    message = ChatCompletionMessage(role="assistant", content=None, tool_calls=calls)
    return Choice(finish_reason="tool_calls", index=0, message=message)


def test_single_tool_call_gets_the_choice_attached() -> None:
    client = object.__new__(TrainableOpenAIChatCompletionClient)
    choice = _tool_call_choice(n=1)
    contents = TrainableOpenAIChatCompletionClient._parse_tool_calls_from_openai(
        client, choice
    )
    assert len(contents) == 1
    assert contents[0].raw_representation is choice
    assert contents[0].name == "wikipedia_search"
    assert contents[0].call_id == "call_0"


def test_parallel_tool_calls_fall_back_to_stock_behavior() -> None:
    """Ambiguous which tokens belong to which call -- see module docstring.
    Falls back to MAF's stock (undiscovered-Choice) behavior, not a crash
    or a wrong attribution."""
    client = object.__new__(TrainableOpenAIChatCompletionClient)
    choice = _tool_call_choice(n=2)
    contents = TrainableOpenAIChatCompletionClient._parse_tool_calls_from_openai(
        client, choice
    )
    assert len(contents) == 2
    assert all(c.raw_representation is not choice for c in contents)


def test_no_tool_calls_returns_empty() -> None:
    client = object.__new__(TrainableOpenAIChatCompletionClient)
    message = ChatCompletionMessage(role="assistant", content="hi", tool_calls=None)
    choice = Choice(finish_reason="stop", index=0, message=message)
    contents = TrainableOpenAIChatCompletionClient._parse_tool_calls_from_openai(
        client, choice
    )
    assert contents == []


def _chat_completion(*, prompt_token_ids: list[int] | None) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content="hi")
    choice = Choice(finish_reason="stop", index=0, message=message)
    kwargs: dict[str, object] = {}
    if prompt_token_ids is not None:
        kwargs["prompt_token_ids"] = prompt_token_ids
    return ChatCompletion(
        id="x",
        choices=[choice],
        created=int(time.time()),
        model="m",
        object="chat.completion",
        **kwargs,  # type: ignore[arg-type]
    )


def test_parse_response_copies_prompt_token_ids_onto_the_choice() -> None:
    """Regression test for a real, confirmed bug: ART's ServerlessBackend
    rejects every trajectory with "Trainable Choice is missing vLLM
    prompt_token_ids/token_ids" because MAF's stock parsing never copies
    the response-level prompt_token_ids field down onto the per-choice
    object it actually stores as raw_representation -- confirmed live
    against a real training run and a real API response (prompt_token_ids
    is top-level on ChatCompletion, token_ids is per-choice; only the
    latter survived MAF's own parsing)."""
    client = object.__new__(TrainableOpenAIChatCompletionClient)
    response = _chat_completion(prompt_token_ids=[1, 2, 3])

    chat_response = TrainableOpenAIChatCompletionClient._parse_response_from_openai(
        client, response, {}
    )

    found = False
    for message in chat_response.messages:
        for content in message.contents:
            raw = getattr(content, "raw_representation", None)
            if isinstance(raw, Choice):
                assert raw.prompt_token_ids == [1, 2, 3]  # type: ignore[attr-defined]
                found = True
    assert found, "expected a Choice-carrying content in the parsed response"


def test_parse_response_no_prompt_token_ids_does_not_crash() -> None:
    """A response without this field (e.g. a non-vLLM provider, or
    capture_logprobs=False callers that never request it) must not raise --
    this client is also used wherever the stock one would be."""
    client = object.__new__(TrainableOpenAIChatCompletionClient)
    response = _chat_completion(prompt_token_ids=None)

    chat_response = TrainableOpenAIChatCompletionClient._parse_response_from_openai(
        client, response, {}
    )
    assert len(chat_response.messages) == 1
