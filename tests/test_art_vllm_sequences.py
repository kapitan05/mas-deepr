"""ART contract test for the B1 token-budget guard.

Trimming tool results mid-loop (agents/context_budget.py) is safe for GRPO
only because ART trains each ``Choice`` on vLLM's own ``prompt_token_ids``
and starts a NEW training sequence whenever a choice's prompt doesn't
extend the running one -- so a completion sampled under a trimmed context
is trained under exactly that context, never under the untrimmed history.
Confirmed in openpipe-art 0.5.20,
``art/preprocessing/tokenize.py::assemble_vllm_training_sequences``.

If a future ART version re-tokenized ``messages`` instead, nothing would
error -- training quality would just silently drop. This test pins the
behavior. Run with: ``uv run pytest tests/test_art_vllm_sequences.py -m art``
(needs the rl-art extra).
"""

from typing import Any, cast

import pytest

# importorskip, not bare imports -- see test_cold_start.py: art's tokenize
# module imports torch, which only the rl-art extra installs.
pytest.importorskip("torch")
art = pytest.importorskip("art")

from art.preprocessing.tokenize import (  # noqa: E402
    assemble_vllm_training_sequences,
)
from art.trajectories import LegacyHistory  # noqa: E402
from openai.types.chat import ChatCompletionMessage  # noqa: E402
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs  # noqa: E402
from openai.types.chat.chat_completion_token_logprob import (  # noqa: E402
    ChatCompletionTokenLogprob,
)

pytestmark = [pytest.mark.art]


def _choice(prompt_ids: list[int], completion_ids: list[int]) -> Choice:
    choice = Choice(
        finish_reason="stop",
        index=0,
        message=ChatCompletionMessage(role="assistant", content="x"),
        logprobs=ChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(
                    token=str(t), logprob=-0.5, bytes=None, top_logprobs=[]
                )
                for t in completion_ids
            ]
        ),
    )
    # Exactly where TrainableOpenAIChatCompletionClient / vLLM's
    # return_token_ids put them.
    cast(dict[str, Any], choice.model_extra)["prompt_token_ids"] = prompt_ids
    cast(dict[str, Any], choice.model_extra)["token_ids"] = completion_ids
    return choice


def _assemble(choices: list[Choice]) -> list[Any]:
    trajectory = art.Trajectory(messages_and_choices=list(choices), reward=1.0)
    return assemble_vllm_training_sequences(
        tokenizer=cast(Any, None),  # only stored on the result, never called
        histories=[LegacyHistory(messages_and_choices=list(choices), tools=None)],
        advantage=1.0,
        allow_training_without_logprobs=False,
        trajectory=trajectory,
    )


def test_extending_prompts_share_one_sequence() -> None:
    # Normal tool loop: each prompt = previous prompt + completion + tool result.
    results = _assemble([_choice([1, 2, 3], [4, 5]), _choice([1, 2, 3, 4, 5, 6], [7])])
    assert len(results) == 1
    assert results[0].token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert results[0].assistant_mask == [0, 0, 0, 1, 1, 0, 1]


def test_trimmed_prompt_starts_new_sequence_with_exact_sampled_context() -> None:
    # Third call's history was trimmed by the guard ([3, 4, 5, 6] -> [9]),
    # so its prompt no longer extends the running sequence.
    results = _assemble(
        [
            _choice([1, 2, 3], [4, 5]),
            _choice([1, 2, 3, 4, 5, 6], [7]),
            _choice([1, 2, 9], [10]),
        ]
    )
    assert len(results) == 2
    assert results[0].token_ids == [1, 2, 3, 4, 5, 6, 7]
    # Trained under exactly the (trimmed) prompt vLLM saw.
    assert results[1].token_ids == [1, 2, 9, 10]
    assert results[1].assistant_mask == [0, 0, 0, 1]
