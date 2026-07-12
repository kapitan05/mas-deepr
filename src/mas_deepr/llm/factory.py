"""Chat-client factory. Single place that decides which endpoint serves a model.

Self-hosted SLMs go through ``settings.slm_base_url`` -- point that at local
vLLM for dev, at the Polar proxy for trajectory capture, or at an ART server
during GRPO training. Frontier models go to their native API.
"""

from agent_framework.openai import OpenAIChatCompletionClient

from mas_deepr.config import ModelSpec, Settings


def build_chat_client(
    spec: ModelSpec, settings: Settings
) -> OpenAIChatCompletionClient:
    """Return a MAF chat client for ``spec``."""
    if spec.self_hosted:
        return OpenAIChatCompletionClient(
            model=spec.model_id,
            api_key=settings.slm_api_key,
            base_url=settings.slm_base_url,
        )
    return OpenAIChatCompletionClient(
        model=spec.model_id,
        api_key=settings.judge_api_key,
        base_url=settings.judge_base_url,
    )
