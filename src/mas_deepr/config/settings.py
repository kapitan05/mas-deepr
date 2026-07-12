"""Runtime configuration. All secrets come from env vars / .env, never hardcoded."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Global settings, loaded from environment / .env.

    ``slm_base_url`` is the single indirection point for inference: point it at
    local vLLM, the Polar proxy, or an ART training server without code changes.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MAS_", extra="ignore"
    )

    # Policy SLM (OpenAI-compatible endpoint: vLLM / Polar proxy / ART)
    slm_base_url: str = "http://localhost:8000/v1"
    slm_api_key: str = "EMPTY"
    slm_model: str = "Qwen/Qwen3-8B"

    # Frontier model used as judge / comparison baseline
    judge_base_url: str | None = None
    judge_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    judge_model: str = "gpt-5-mini"

    # Tools
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    search_max_results: int = 5
    fetch_timeout_s: float = 20.0
    fetch_max_chars: int = 8000

    # Agent loop limits
    max_sub_queries: int = 4
    max_tool_calls_per_query: int = 8

    # Paths
    data_dir: Path = PROJECT_ROOT / "assets" / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    cache_db: Path = PROJECT_ROOT / "assets" / "web_cache.sqlite3"

    # Eval
    eval_concurrency: int = 4

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.runs_dir, self.cache_db.parent):
            p.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
