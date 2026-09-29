"""DSPy LM factory -- mirrors llm/factory.py's self-hosted vs. frontier split.

Kept separate from ``mas_deepr.llm`` because it returns a ``dspy.LM``
(litellm-backed) rather than a MAF chat client; the two frameworks talk to
the same endpoints through different client types.
"""

import dspy

from mas_deepr.config import ModelSpec, Settings


def build_dspy_lm(spec: ModelSpec, settings: Settings) -> dspy.LM:
    """Return a DSPy LM for ``spec``.

    Confirmed live 2026-09-26 (first-ever ``compile_dspy.py`` run against a
    non-self-hosted spec, ``qwen3-14b-instruct-wandb``): the frontier
    branch here was missing BOTH the ``openai/`` provider prefix litellm
    requires for a custom model string (bare ``spec.model_id`` raised
    ``litellm.BadRequestError: LLM Provider NOT provided``) AND
    ``spec.api_base_url``/``spec.api_key_env`` -- it silently used the
    *judge's* endpoint/key for every frontier spec instead of the model's
    own, the exact bug class already fixed in ``llm/factory.py::
    build_chat_client`` (GEMINI_API_KEY unset -> OPENAI_API_KEY sent to
    Google) but never mirrored here since no prior compile run had ever
    exercised this branch.
    """
    if spec.self_hosted:
        return dspy.LM(
            f"openai/{spec.model_id}",
            api_base=settings.slm_base_url,
            api_key=settings.slm_api_key,
        )

    if spec.api_key_env is not None:
        api_key = spec.resolved_api_key()
        if not api_key:
            raise ValueError(
                f"Model {spec.key!r} needs {spec.api_key_env} set in the "
                f"environment, but it's empty/unset. A key only present in "
                f".env is not enough unless something has loaded .env into "
                f"the process environment -- see docs/running-phase-1.md."
            )
    else:
        api_key = settings.judge_api_key

    base_url = spec.api_base_url or settings.judge_base_url
    return dspy.LM(f"openai/{spec.model_id}", api_base=base_url, api_key=api_key)
