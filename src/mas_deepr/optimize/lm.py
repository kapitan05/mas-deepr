"""DSPy LM factory -- mirrors llm/factory.py's self-hosted vs. frontier split.

Kept separate from ``mas_deepr.llm`` because it returns a ``dspy.LM``
(litellm-backed) rather than a MAF chat client; the two frameworks talk to
the same endpoints through different client types.
"""

import dspy

from mas_deepr.config import ModelSpec, Settings


def build_dspy_lm(spec: ModelSpec, settings: Settings) -> dspy.LM:
    if spec.self_hosted:
        return dspy.LM(
            f"openai/{spec.model_id}",
            api_base=settings.slm_base_url,
            api_key=settings.slm_api_key,
        )
    return dspy.LM(
        spec.model_id,
        api_base=settings.judge_base_url,
        api_key=settings.judge_api_key,
    )
