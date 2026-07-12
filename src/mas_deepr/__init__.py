"""Multi-agent SLM deep-research stack.

Modules:
    config     -- pydantic-settings runtime config + model registry
    prompts    -- YAML prompt registry (DSPy writes compiled prompts here)
    agents     -- MAF topology (Manager -> Browser -> Synthesizer)
    tools      -- search / fetch / code-exec behind a cache
    llm        -- OpenAI-compatible chat-client factory (vLLM / Polar / frontier)
    data       -- benchmark loaders (eval-only) + train-pool builders
    evals      -- graders and benchmark runners
    optimize   -- DSPy adapter + compile jobs (Phase 2)
    rl         -- ART rollout + reward functions (Phase 4)
    telemetry  -- per-call token/latency/cost logging
"""
