"""Model registry: every model the thesis matrix touches, with pricing.

Adding a model = adding one ``ModelSpec``. Prices are $ per 1M tokens; for
self-hosted SLMs use the effective GPU-rental cost estimate so the
accuracy-vs-cost plot stays honest.
"""

from pydantic import BaseModel


class ModelSpec(BaseModel):
    key: str  # registry key used in configs / CLI
    model_id: str  # id sent to the OpenAI-compatible endpoint
    family: str  # qwen3 | gpt-oss | frontier
    params_b: float | None = None
    input_price_per_mtok: float = 0.0
    output_price_per_mtok: float = 0.0
    self_hosted: bool = True


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
