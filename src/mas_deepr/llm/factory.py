"""Chat-client factory. Single place that decides which endpoint serves a model.

Self-hosted SLMs go through ``settings.slm_base_url`` -- point that at local
vLLM for dev, at the Polar proxy for trajectory capture, or at an ART server
during GRPO training. Frontier models carry their own ``api_base_url`` /
``api_key_env`` on the ``ModelSpec``; a frontier spec without those (e.g. the
``gpt-5-mini`` judge) falls back to the ``judge_*`` settings.

Also the single place HTTP resilience (timeout + retry) gets applied --
every model built here shares it, since every caller (agents/topology.py,
evals/judge.py) goes through this one function.
"""

import httpx
from agent_framework.openai import OpenAIChatCompletionClient
from openai import AsyncOpenAI

from mas_deepr.config import ModelSpec, Settings


def _build_async_client(
    *, api_key: str, base_url: str | None, settings: Settings
) -> AsyncOpenAI:
    """One ``AsyncOpenAI`` with a real read timeout and retry budget.

    Confirmed live: a request to api.inference.wandb.ai opened a connection
    and never sent a response back, hanging an entire milestone eval
    indefinitely on one question -- there was no timeout to force it to
    give up. ``max_retries`` reuses the SDK's own exponential-backoff retry
    for timeouts/connection errors/429/5xx; not layering tenacity on top of
    this, since that would multiply attempts (SDK retries x tenacity
    retries) rather than add resilience -- see tools/mcp_client.py for
    where tenacity *is* the right tool (no such built-in retry exists on
    that transport).
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(settings.llm_request_timeout_s, connect=10.0),
        max_retries=settings.llm_max_retries,
    )


def build_chat_client(
    spec: ModelSpec,
    settings: Settings,
    *,
    client_cls: type[OpenAIChatCompletionClient] = OpenAIChatCompletionClient,
) -> OpenAIChatCompletionClient:
    """Return a MAF chat client for ``spec``.

    ``client_cls`` defaults to MAF's stock client; ``rl/rollout.py`` passes
    ``rl.trainable_client.TrainableOpenAIChatCompletionClient`` for GRPO
    rollouts so tool-call turns keep their logprob-bearing ``Choice``
    instead of MAF discarding it (see that module's docstring) -- same
    endpoint/key resolution either way, only the response-parsing class
    differs.
    """
    if spec.self_hosted:
        async_client = _build_async_client(
            api_key=settings.slm_api_key,
            base_url=settings.slm_base_url,
            settings=settings,
        )
        return client_cls(model=spec.model_id, async_client=async_client)

    if spec.api_key_env is not None:
        # A spec that names its own key env var must actually resolve it --
        # silently falling back to the judge's (different provider's) key
        # here previously sent the wrong secret to the wrong endpoint
        # (confirmed live: GEMINI_API_KEY unset -> OPENAI_API_KEY sent to
        # Google, rejected with "Please pass a valid API key"). Fail loudly
        # instead.
        api_key = spec.resolved_api_key()
        if not api_key:
            raise ValueError(
                f"Model {spec.key!r} needs {spec.api_key_env} set in the "
                f"environment, but it's empty/unset. A key only present in "
                f".env is not enough unless something has loaded .env into "
                f"the process environment -- see docs/running-phase-1.md."
            )
    else:
        # No per-spec key configured (e.g. gpt-5-mini, the judge) -> use
        # the judge's own settings by design.
        api_key = settings.judge_api_key

    base_url = spec.api_base_url or settings.judge_base_url
    async_client = _build_async_client(
        api_key=api_key, base_url=base_url, settings=settings
    )
    return client_cls(model=spec.model_id, async_client=async_client)
