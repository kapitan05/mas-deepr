"""Runtime configuration. All secrets come from env vars / .env, never hardcoded."""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# pydantic-settings' own env_file=".env" support (below) only populates
# fields declared on Settings -- it never exports .env into os.environ.
# Anything read by raw name elsewhere (ModelSpec.api_key_env for the
# frontier/W&B-Inference specs, WANDB_API_KEY for the monitoring layer)
# needs the actual process environment, not just Settings' own view of
# .env. Confirmed live: GEMINI_API_KEY sitting only in .env was silently
# invisible to os.environ.get(...), which is what caused
# llm/factory.py::build_chat_client to fall back onto the wrong provider's
# key. override=False -- real shell exports still win over .env.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class Settings(BaseSettings):
    """Global settings, loaded from environment / .env.

    ``slm_base_url`` is the single indirection point for inference: point it at
    local vLLM, the Polar proxy, or an ART training server without code changes.
    """

    model_config = SettingsConfigDict(
        # Absolute path -- ".env" alone is relative to cwd, so a script run
        # from outside the repo root would silently miss it while the
        # load_dotenv() call above (also absolute) still found it.
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="MAS_",
        extra="ignore",
    )

    # Policy SLM (OpenAI-compatible endpoint: vLLM / Polar proxy / ART)
    slm_base_url: str = "http://localhost:8000/v1"
    slm_api_key: str = "EMPTY"
    slm_model: str = "Qwen/Qwen3-8B"

    # Frontier model used as judge / comparison baseline
    # judge_base_url/judge_api_key are only a fallback for a judge_model spec
    # that declares no api_key_env of its own (see llm/factory.py) -- the
    # default below (gemini-2.5-flash-lite) carries its own GEMINI_API_KEY/
    # base_url and never touches these two. Kept for anyone overriding
    # MAS_JUDGE_MODEL back to a spec without its own endpoint (e.g. gpt-5-mini).
    judge_base_url: str | None = None
    judge_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    # Cheap, tool-call-free judge hosted on W&B Inference -- $0.03/$0.13 per
    # 1M tokens, billed on WANDB_API_KEY rather than a personal OpenAI/
    # Google key, so judge grading can't zero out every model's accuracy on
    # BrowseComp/ResearchRubrics just because one provider's account ran dry
    # (see config/models.py's gpt-oss-20b-wandb entry and
    # docs/eval-architecture.md). Override via MAS_JUDGE_MODEL.
    judge_model: str = "gpt-oss-20b-wandb"

    # LLM client resilience (llm/factory.py -- one place, applies to every
    # model: self-hosted, frontier, judge). Confirmed live 2026-09-20: a
    # request to api.inference.wandb.ai opened a connection and then never
    # sent a response, and with no read timeout the whole milestone run
    # hung indefinitely on one question, not just that one call. max_retries
    # reuses the OpenAI SDK's own exponential-backoff retry (timeouts,
    # connection errors, 429/5xx) -- deliberately not a second tenacity
    # layer on top, which would multiply attempts instead of adding
    # resilience.
    llm_request_timeout_s: float = 60.0
    llm_max_retries: int = 3

    # Tools -- served through the MCP tool backend (mcp_backend/), not
    # called in-process. SearXNG (search) + Crawl4AI (fetch) are the
    # default $0-marginal-cost providers; Tavily is kept as an opt-in
    # fallback, excluded from mcp_enabled_tools by default -- see
    # docs/dr-tulu-agent-infra.md and the MCP-adoption plan.
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")
    search_max_results: int = 5
    fetch_timeout_s: float = 20.0
    fetch_max_chars: int = 8000

    # MCP server connection (the Browser's tools proxy through this, not
    # in-process calls -- see mcp_backend/server.py + tools/mcp_client.py)
    mcp_transport: str = "http"  # "http" | "stdio"
    mcp_server_host: str = "127.0.0.1"
    mcp_server_port: int = 8765
    mcp_server_url: str = "http://127.0.0.1:8765/mcp"
    # Client-side concurrency cap, independent of eval_concurrency
    # (pipeline/question-level). A secondary safety net only -- the
    # authoritative limits against upstream bans are the server-side rate
    # limiters below (mcp_backend/rate_limit.py), since the server is the
    # one process actually making outbound calls, regardless of how many
    # client processes (eval script, GRPO training, ...) are asking it to.
    mcp_tool_concurrency: int = 8
    mcp_searxng_concurrency: int = 2
    # Comma-separated tool names the Browser is allowed to call -- the
    # single source of truth for which of the 6 MCP tools get attached to
    # the Browser (build_pipeline gates every one of them off this set,
    # uniformly; no separate per-category booleans). Default excludes only
    # the opt-in Tavily fallback. Override via MAS_MCP_ENABLED_TOOLS to add
    # "tavily_search", or to DROP "web_search" for a genuinely
    # ban-risk-free run on Wikipedia-answerable benchmarks (FRAMES/
    # MuSiQue/HotpotQA) -- see the deployment checklist's "RL-scale ban
    # risk" section. wikipedia_search is a direct MediaWiki API call, NOT
    # routed through SearXNG, so it carries none of web_search's
    # scraped-engine ban risk.
    mcp_enabled_tools: str = (
        "web_search,fetch_page,wikipedia_search,semantic_scholar_search,pubmed_search"
    )
    searxng_base_url: str = Field(default="", validation_alias="SEARXNG_BASE_URL")
    semantic_scholar_api_key: str = Field(
        default="", validation_alias="SEMANTIC_SCHOLAR_API_KEY"
    )

    # Server-side rate limits (mcp_backend/rate_limit.py) -- these are the
    # numbers that actually prevent a ban, tuned against each provider's
    # documented ceiling with real margin, not just "seems reasonable":
    #   - PubMed/NCBI: hard-documented 3 req/s without a key, 10 req/s
    #     with one (eutilities.github.io/site/API_Key). Default leaves
    #     ~35% margin below the no-key ceiling.
    #   - Semantic Scholar: confirmed directly by their API team (email,
    #     2026-09-20) -- keyed rate limit is 1 req/s, CUMULATIVE ACROSS ALL
    #     ENDPOINTS, and they explicitly ask callers to set below that
    #     threshold, not AT it, to avoid rejected requests. Unauthenticated
    #     callers share one pool worldwide with an unpredictable effective
    #     rate regardless. Default leaves margin under the keyed ceiling
    #     either way.
    #   - SearXNG: NOT primarily a rate problem -- Google can fingerprint
    #     and block a fresh instance after as few as ~5 rapid searches
    #     (TLS/HTTP2 fingerprinting, not a request-count threshold a rate
    #     limiter alone fixes). This default paces requests to reduce
    #     burst risk; it is not sufficient on its own -- see the
    #     deployment checklist for the engine-configuration mitigation
    #     that actually matters (multi-engine, deprioritize Google).
    pubmed_rate_per_sec: float = 2.0
    semantic_scholar_rate_per_sec: float = 0.9
    searxng_rate_per_min: float = 20.0
    # Wikipedia's REST API is a sanctioned, non-adversarial service (not
    # scraping) -- documented etiquette asks for serial requests, well
    # under ~5 req/s (mediawiki.org/wiki/API:Etiquette). This holds true
    # regardless of how many GRPO rollouts/parallel questions are queued
    # behind it: a token-bucket limiter caps sustained *throughput*, not
    # concurrency, so queue depth adds latency, not ban risk. Raised from
    # 3.0 after empirically probing the real MCP server (scripts/
    # probe_rate_limit.py) live 2026-09-29: 0 failures across 20/40/60-call
    # bursts at 3, 8, and 15 req/s respectively -- 10 leaves real margin
    # below the tested-clean ceiling while meaningfully cutting rollout
    # stall time. (Semantic Scholar's own 0.9 above was tested the same
    # way and NOT raised -- see its comment: 0.9 is a deliberately-reasoned
    # value from a direct provider confirmation, not a guess, and the probe
    # found consistent ~5-7% failure rates at 0.9/1.0/1.2 alike, i.e.
    # already at real capacity, not scope for improvement via this knob.)
    wikipedia_rate_per_sec: float = 10.0
    # Tavily's documented Development-tier ceiling is 100 req/min; this
    # leaves margin and matters less in practice since it's off by
    # default -- included for parity/completeness, not because it's the
    # binding constraint here.
    tavily_rate_per_min: float = 60.0
    # Crawl4AI is a different resource profile entirely: no external
    # rate-limit/ban risk (it fetches whatever URL it's given, not one
    # rate-limited API), but a real LOCAL resource cost -- each call
    # launches a headless Chromium instance. Bounded by its own
    # concurrency limit, not a time-based rate, sized conservatively for
    # a single Apple Silicon laptop (see the deployment checklist).
    crawl4ai_max_concurrent: int = 3

    # Agent loop limits
    max_sub_queries: int = 4
    max_tool_calls_per_query: int = 8

    # Paths
    data_dir: Path = PROJECT_ROOT / "assets" / "data"
    runs_dir: Path = PROJECT_ROOT / "runs"
    cache_db: Path = PROJECT_ROOT / "assets" / "web_cache.sqlite3"

    # Eval
    eval_concurrency: int = 4

    # Monitoring -- one W&B project for eval + training runs alike.
    wandb_project: str = "mas-deepr"

    # GRPO rollout concurrency (rl/rollout.py) -- caps how many whole
    # rollouts (one full Manager->Browser*->Synthesizer pipeline each) run
    # concurrently against the in-training model's served endpoint. This
    # is DR-Tulu's outermost "rollout/batch" layer specifically
    # (docs/dr-tulu-agent-infra.md ss2's table: "how many *agent
    # invocations* run at once"), not their finer-grained per-completion-
    # call semaphore -- wrapping every individual chat-completion inside
    # MAF's own multi-turn agent loop would need deeper surgery than this
    # round's scope. Still a real, separate layer from MCP tool-call
    # concurrency (mcp_tool_concurrency above, tools/mcp_client.py) --
    # DR-Tulu's core principle is never one shared bound across resource
    # types, since an LLM-call burst and a tool-call burst have different
    # cost/failure profiles and shouldn't starve each other. Needed now
    # that run_rollout_group's rollouts run concurrently (fixed
    # 2026-09-25 -- they used to run sequentially, one full rollout's
    # latency at a time, for no reason: each rollout in a GRPO group is
    # independent) -- without this, group_size x concurrent-questions
    # fan-out (already concurrent via art.gather_trajectory_groups) could
    # send unbounded simultaneous requests at one served model endpoint.
    grpo_llm_concurrency: int = 8

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.runs_dir, self.cache_db.parent):
            p.mkdir(parents=True, exist_ok=True)

    def mcp_enabled_tools_set(self) -> set[str]:
        """Parsed ``mcp_enabled_tools`` -- the set of MCP tool names the
        Browser is allowed to call."""
        return {t.strip() for t in self.mcp_enabled_tools.split(",") if t.strip()}


def get_settings() -> Settings:
    return Settings()
