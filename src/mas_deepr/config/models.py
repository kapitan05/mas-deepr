"""Model registry: every model the thesis matrix touches, with pricing.

Adding a model = adding one ``ModelSpec``. Prices are $ per 1M tokens; for
self-hosted SLMs use the effective GPU-rental cost estimate so the
accuracy-vs-cost plot stays honest. Frontier prices below are public
rate-card estimates as of 2026-09 -- re-check before quoting them in the
thesis.
"""

import os

from pydantic import BaseModel


class ModelSpec(BaseModel):
    key: str  # registry key used in configs / CLI
    model_id: str  # id sent to the OpenAI-compatible endpoint
    family: str  # qwen3 | gpt-oss | frontier
    params_b: float | None = None
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0
    self_hosted: bool = True
    # Frontier-model endpoint overrides. ``None`` => the factory falls back
    # to the judge endpoint (backwards compatible with gpt-5-mini).
    api_base_url: str | None = None
    api_key_env: str | None = None  # name of the env var holding the API key
    # False => this model can't do OpenAI-style function calling; the Browser
    # role needs the prompted-JSON-tools fallback (agents/topology.py).
    supports_native_tools: bool = True
    # False => the Synthesizer's run_python tool gets dropped for this model
    # instead of attached (agents/topology.py). Narrower than
    # supports_native_tools above: some models handle web_search/fetch_page
    # fine but garble run_python specifically -- confirmed live on
    # llama-3.1-8b-wandb, whose FRAMES answers (which need arithmetic/date
    # verification) came back as malformed pseudo-JSON/Python text instead
    # of a real tool call or a real answer, every single time, while its
    # browsecomp/research_rubrics answers (no run_python needed) were fine.
    supports_code_exec: bool = True
    # Account-tier TPM (tokens/minute) ceiling for this model's endpoint --
    # None => no known cap, build_pipeline attaches a no-op NullTpmLimiter
    # (llm/rate_limit.py) instead of pacing calls. This is a property of
    # *your account's tier* on that provider, not the model itself -- a
    # different OpenAI org/tier has a different number. Only set where
    # confirmed live; don't guess a number for the sake of filling the
    # field in, an absent limiter is strictly safer than a wrong one.
    tpm_limit: int | None = None

    def resolved_api_key(self) -> str | None:
        """The API key for this spec, read from ``api_key_env`` if set."""
        if self.api_key_env is None:
            return None
        return os.environ.get(self.api_key_env)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec(
            key="qwen3-4b",
            model_id="Qwen/Qwen3-4B",
            family="qwen3",
            params_b=4,
            input_price_per_mtok=0.03,
            output_price_per_mtok=0.09,
        ),
        ModelSpec(
            key="qwen3-8b",
            model_id="Qwen/Qwen3-8B",
            family="qwen3",
            params_b=8,
            input_price_per_mtok=0.05,
            output_price_per_mtok=0.15,
        ),
        ModelSpec(
            key="qwen3-14b",
            model_id="Qwen/Qwen3-14B",
            family="qwen3",
            params_b=14,
            input_price_per_mtok=0.08,
            output_price_per_mtok=0.24,
        ),
        ModelSpec(
            key="gpt-oss-20b",
            model_id="openai/gpt-oss-20b",
            family="gpt-oss",
            params_b=20,
            input_price_per_mtok=0.10,
            output_price_per_mtok=0.30,
        ),
        ModelSpec(
            key="gpt-5-mini",
            model_id="gpt-5-mini",
            family="frontier",
            input_price_per_mtok=0.25,
            output_price_per_mtok=2.00,
            self_hosted=False,
        ),
        # --- Frontier comparison baselines (run through the same pipeline) ---
        ModelSpec(
            key="gpt-4.1",
            model_id="gpt-4.1",
            family="frontier",
            input_price_per_mtok=2.00,
            output_price_per_mtok=8.00,
            self_hosted=False,
            api_key_env="OPENAI_API_KEY",
            # Confirmed live 2026-09-25: "Rate limit reached for gpt-4.1 in
            # organization ... on tokens per min (TPM): Limit 30000" --
            # this account's tier ceiling, not a gpt-4.1-wide constant.
            # Re-check platform.openai.com/account/rate-limits if this org
            # upgrades tier, and update this number accordingly.
            tpm_limit=30_000,
        ),
        ModelSpec(
            # Registry key kept stable (what you type on --models) even
            # though the live model_id has moved twice now:
            # 1. confirmed live 2026-09-11: "gemini-2.5-pro is no longer
            #    available to new users" (404) -> tried gemini-3.1-pro-preview.
            # 2. gemini-3.1-pro-preview (a "thinking" model) then failed
            #    tool-calling with "Function call is missing a
            #    thought_signature" -- a real, currently-open upstream bug
            #    in Gemini 3.x's OpenAI-compatibility layer, not our code
            #    (see docs/gemini-thought-signature-issue.md). Switched to
            #    gemini-2.5-flash, a non-"thinking-by-default" tier, as a
            #    quick mitigation -- not guaranteed to dodge the same bug,
            #    confirm with the Stage A smoke-test checkpoint before
            #    trusting it for a real run.
            # Google's model lineup moves fast; if this 404s again, check
            # https://ai.google.dev/gemini-api/docs/models for the current
            # id and swap it in here, key unchanged. Price below is carried
            # over from 2.5-pro, NOT re-verified for gemini-2.5-flash.
            key="gemini-2.5-pro",
            model_id="gemini-2.5-flash",
            family="frontier",
            input_price_per_mtok=1.25,
            output_price_per_mtok=10.00,
            self_hosted=False,
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
        ),
        # Cheap judge model -- text-only grading (BrowseComp/ResearchRubrics),
        # no tool calling, so the gemini-2.5-pro entry's thought_signature
        # tool-calling caveat above doesn't apply here. Confirmed "Current"
        # (not deprecated) on ai.google.dev/gemini-api/docs/pricing,
        # 2026-09-13: $0.10/$0.40 per 1M tokens, well under gpt-5-mini
        # ($0.25/$2.00) -- point Settings.judge_model / MAS_JUDGE_MODEL at
        # this key to get judge grading off the OpenAI account entirely
        # (see the shared-judge single-point-of-failure note in
        # docs/eval-architecture.md: an exhausted OPENAI_API_KEY otherwise
        # zeroes browsecomp/research_rubrics for every model under test,
        # not just OpenAI ones). Superseded as the *default* judge by
        # gpt-oss-20b-wandb below (cheaper, and doesn't touch a personal
        # provider key at all) -- kept registered as a fallback/alternative.
        ModelSpec(
            key="gemini-2.5-flash-lite",
            model_id="gemini-2.5-flash-lite",
            family="frontier",
            input_price_per_mtok=0.10,
            output_price_per_mtok=0.40,
            self_hosted=False,
            api_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
        ),
        # Default judge as of 2026-09-14: OpenAI's own open-weight model,
        # hosted on W&B Inference -- billed on WANDB_API_KEY, not a personal
        # OpenAI/Google API key, so judge grading no longer touches either
        # of those quotas at all. Cheapest option checked (confirmed live
        # against wandb.ai/site/pricing/inference): $0.03/$0.13 per 1M
        # tokens, versus $0.10/$0.40 for gemini-2.5-flash-lite above.
        # Deliberately not llama-3.1-8b-wandb/granite-4.2-8b-wandb -- those
        # are comparison *targets* in the frontier matrix, and judging a
        # model with itself risks self-preference bias.
        ModelSpec(
            key="gpt-oss-20b-wandb",
            model_id="openai/gpt-oss-20b",
            family="frontier",
            params_b=20,
            input_price_per_mtok=0.03,
            output_price_per_mtok=0.13,
            self_hosted=False,
            api_base_url="https://api.inference.wandb.ai/v1",
            api_key_env="WANDB_API_KEY",
        ),
        ModelSpec(
            key="deepseek-chat",
            model_id="deepseek-chat",
            family="frontier",
            input_price_per_mtok=0.27,
            output_price_per_mtok=1.10,
            self_hosted=False,
            api_base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
        ),
        # --- W&B Inference (CoreWeave-hosted, OpenAI-compatible) ---
        # Same WANDB_API_KEY as the monitoring layer -- one account, one key.
        # Both also serve as LoRA-hot-swap targets: a PEFT adapter trained by
        # rl/cold_start.py or rl/dpo.py (use_lora=True) and uploaded as a W&B
        # "lora"-type artifact can be served by pointing model_id at
        # "wandb-artifact:///{entity}/{project}/{name}:latest" instead -- see
        # docs/wandb-inference-and-lora.md.
        ModelSpec(
            key="llama-3.1-8b-wandb",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            family="frontier",
            params_b=8,
            input_price_per_mtok=0.22,
            output_price_per_mtok=0.22,
            self_hosted=False,
            api_base_url="https://api.inference.wandb.ai/v1",
            api_key_env="WANDB_API_KEY",
            # Confirmed live 2026-09-13: every FRAMES answer (needs run_python
            # for arithmetic/date verification) came back as garbled
            # pseudo-tool-call text instead of a real answer -- web_search/
            # fetch_page tool calls work fine for this model, just not
            # run_python. See supports_code_exec's docstring above.
            supports_code_exec=False,
        ),
        ModelSpec(
            key="granite-4.2-8b-wandb",
            model_id="ibm-granite/granite-4.2-8b",
            family="frontier",
            params_b=8,
            # Confirmed 2026-09-13 directly against wandb.ai/site/pricing/inference
            # (was a stale 4.1-sourced placeholder: $0.05/$0.10).
            input_price_per_mtok=0.10,
            output_price_per_mtok=0.15,
            self_hosted=False,
            api_base_url="https://api.inference.wandb.ai/v1",
            api_key_env="WANDB_API_KEY",
        ),
        # The GRPO pre-training baseline for scripts/train_grpo.py's default
        # ART ServerlessBackend target -- confirmed live 2026-09-26 that
        # ServerlessBackend currently supports exactly two base models
        # (OpenPipe/Qwen3-14B-Instruct, Qwen/Qwen3-30B-A3B; see
        # art.openpipe.ai/resources/models), and that a freshly-registered
        # TrainableModel's inference_base_url can NOT be queried before at
        # least one real training step exists ("LoRA inference requires an
        # explicit checkpoint... Unversioned references and aliases such as
        # 'latest' are not supported" -- a live 400 from
        # api.training.wandb.ai). There is no "step 0" to eval through ART
        # itself. This spec is the workaround: the identical base weights,
        # unmodified (LoRA training only adds an adapter on top), served as
        # a stable, always-on W&B Inference endpoint instead -- confirmed
        # present via a live GET to api.inference.wandb.ai/v1/models with
        # the exact same model id ART's base_model expects. Eval this
        # BEFORE training as the real "pre-GRPO" number; after training,
        # compare against the trained LoRA checkpoint (a separate
        # model_id="wandb-artifact:///..." spec, registered per run).
        ModelSpec(
            key="qwen3-14b-instruct-wandb",
            model_id="OpenPipe/Qwen3-14B-Instruct",
            family="frontier",
            params_b=14,
            # Confirmed live 2026-09-26 against wandb.ai/site/pricing/inference.
            input_price_per_mtok=0.05,
            output_price_per_mtok=0.22,
            self_hosted=False,
            api_base_url="https://api.inference.wandb.ai/v1",
            api_key_env="WANDB_API_KEY",
        ),
    ]
}


def get_model(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as e:
        raise KeyError(f"Unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}") from e


def cost_usd(spec: ModelSpec, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * spec.input_price_per_mtok
        + output_tokens * spec.output_price_per_mtok
    ) / 1_000_000
