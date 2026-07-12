# Phase 0 & Phase 1: Environment Baseline and MAF Topology

A from-scratch scaffold of the multi-agent SLM deep-research stack — runtime config through a working three-agent research pipeline, benchmark loaders, and a graded eval harness.

**Status:** 30 source files, 1,828 lines (src+scripts), 50 tests passing, ruff clean, mypy strict clean.

---

## Phase 0: Environment Baseline

A `uv`-managed project with a strict lint/type baseline, and three foundational modules everything downstream depends on: settings, a model/pricing registry, and telemetry. Nothing here is benchmark- or agent-specific — it's the plumbing that has to be right before anything built on top of it can be trusted.

### `pyproject.toml` — config

uv-managed src-layout project, Python 3.12. Ruff (line-length 88, `E F W I UP B SIM RUF`) and mypy in strict-ish mode (`disallow_untyped_defs`, `disallow_incomplete_defs`, `check_untyped_defs`) run across `src`, `scripts`, and `tests` alike.

- **Why it matters:** The CLAUDE.md build contract requires `uv run mypy .` to pass with zero errors and strict typing on every signature — that only holds long-term if tests are checked too, otherwise type drift hides in the one place nobody re-reads.
- **Why this decision:** One per-file ruff ignore exists, for `evals/judge.py`'s E501 — it holds OpenAI's published grading prompt verbatim, and reflowing that text would make it silently diverge from the source it's cited against.

### `config/settings.py` — pydantic-settings

`Settings` loads from env vars / `.env` under an `MAS_` prefix. The single field that matters most: `slm_base_url` — every self-hosted model call goes through this one indirection point.

- **Why it matters:** The plan's Phase 3–4 route inference through three different backends over time — local vLLM now, the Polar capture proxy next, an ART training server later — without touching agent or tool code.
- **Why this decision:** Secrets (Tavily key, judge API key) load only from env, never hardcoded, per the MLOps constraint in CLAUDE.md. No secret is logged anywhere, including telemetry.

### `config/models.py` — registry

A `ModelSpec` registry — key, model id, family, param count, $/1M-token input and output price — covering Qwen3-4B/8B/14B, gpt-oss-20b, and gpt-5-mini (judge).

- **Why it matters:** The thesis's headline deliverable is an Accuracy-vs-Cost plot. That plot is only as honest as the price table it's computed from, so pricing lives in one typed place instead of scattered magic numbers in eval scripts.
- **Why this decision:** Adding a model to the thesis matrix — a new SLM size, a different frontier baseline — is a five-line dict entry, not a code change, matching the "every component modular" requirement.

### `telemetry/tracker.py` — observability

Every LLM call — manager, browser, synthesizer, judge — emits one `LLMCallRecord` (tokens, latency, cost, role, phase) to a thread-safe JSONL sink. `summarize()` rolls it up via polars.

- **Why it matters:** CLAUDE.md requires token usage, latency, and cost logged for every LLM call — this is that requirement, wired through the pipeline from the first agent call onward rather than bolted on later.
- **Why this decision:** JSONL append-only, not a database — Polar's Phase-3 prefix-merging works over exactly this kind of log, so the format was picked to not need rewriting when that phase starts.

### `llm/factory.py` — indirection

`build_chat_client(spec, settings)` — self-hosted specs go to `slm_base_url`, frontier specs go to their native API. One function, one branch.

- **Why it matters:** This is the seam the whole 5-phase plan pivots on: swapping dev inference for the Polar proxy, or the Polar proxy for an ART training endpoint, is a settings change, not a refactor.
- **Why this decision:** Built directly on MAF's own `OpenAIChatCompletionClient` rather than a custom wrapper — anything OpenAI-compatible (vLLM, Polar, ART) works for free.

---

## Phase 1: MAF Topology, Tools, Data, Evals

The working baseline: three agents that can actually research a question, the tools they call, loaders for the three benchmarks plus the training pool, and a harness to grade and score runs. This is what `scripts/run_baseline.py` and `scripts/run_milestone_eval.py` drive.

### Tools

#### `tools/cache.py` — SQLite

A key-value cache over SQLite, keyed by a stable hash of `{kind, **params}` — shared by search and fetch.

- **Why it matters:** BrowseComp answers must not drift as the live web changes between the baseline / post-DSPy / post-GRPO milestones. Caching every outbound call makes a milestone re-run reproducible, and keeps the dev loop cheap.
- **Why this decision:** One table, one process — no cache-invalidation policy needed because entries never need to expire for this project's timeline; simplicity over generality.

#### `tools/search.py` + `tools/fetch.py` — Tavily / trafilatura

Cached, `tenacity`-retried search (Tavily) and page fetch (httpx + trafilatura readability extraction), each exposed to MAF via `@tool`. Fetch failures return a marked empty string instead of raising.

- **Why it matters:** A dead link or a paywall hitting mid-loop must not abort the whole research trajectory — one bad URL degrades gracefully instead of losing an entire GRPO rollout.
- **Why this decision:** Logic lives in plain, directly-testable coroutines (`web_search`, `fetch_page`); the `@tool`-decorated wrapper is a thin shell around each. Tests exercise the coroutine directly with a monkeypatched backend, not the MAF tool-call plumbing.

#### `tools/code_exec.py` — placeholder

Sandboxed Python execution via an isolated subprocess (`python -I`, hard timeout, output cap) — not the real micro-VM the plan names.

- **Why it matters:** The synthesizer needs to verify arithmetic (dates, unit conversions, counts) rather than eyeball it — this is where CodeAct sits in the architecture.
- **Why this decision:** Explicitly labeled placeholder: callers only depend on `run_python` / `make_code_exec_tool`, so swapping in a real firecracker/MAF `SupportsCodeInterpreterTool` backend later touches one file.

### Prompts & agents

#### `prompts/registry.py` + `templates/*.yaml` — YAML

Hand-written Manager/Browser/Synthesizer prompts in YAML. `load_prompt(role, prefer_compiled=...)` prefers a `*.compiled.yaml` variant when one exists.

- **Why it matters:** This is the exact seam Phase 2 (DSPy/MIPROv2) needs — `save_compiled_prompt()` already exists, unused until then. Milestone eval toggles hand-written vs. compiled with one boolean, not a code change.
- **Why this decision:** Prompt isolation from execution logic, per the CLAUDE.md LLM-architecture rule — prompts are data, not string literals buried in `agents/`.

#### `agents/topology.py` — MAF Agent

Manager → Browser → Synthesizer, sequential. One shared SLM plays all three roles via role-specific system instructions; the manager's numbered-list output is parsed into sub-questions, each answered by the browser (tools, `max_function_calls` capped), then synthesized into a final answer.

- **Why it matters:** One shared model across roles is what GRPO trains later — a single LoRA on role-tagged trajectories, per the plan's Phase 4. Building the baseline this way now means Phase 4 doesn't need a re-architecture.
- **Why this decision:** Plain sequential orchestration over MAF `Agent` instances, not a `FunctionalWorkflow` graph — deliberately simple for the baseline. Swapping to a graph topology later only touches this module; tools, prompts, telemetry are untouched by that swap.

### Benchmark & training data

#### `data/schema.py`, `frames.py`, `browsecomp.py`, `research_rubrics.py`, `train_pool.py`, `manifest.py` — loaders

A unified `Question` record (verifiable `answer` or graded `rubrics`, never both) backs five loaders: FRAMES, BrowseComp, ResearchRubrics — all eval-only — plus MuSiQue and HotpotQA for the train/val pool. `manifest.py` commits which question ids landed in which split.

- **Why it matters:** This is the leakage-prevention design from the plan: benchmarks are never split into train/val, only evaluated at milestones. The manifest gives an auditable paper trail proving that.
- **Why this decision:** Every loader's schema and URL was confirmed against the live source before writing code, rather than guessed from memory or the paper text.

**Verified against real sources, not memory.** Dataset identifiers, file names, and column schemas were confirmed live (HF API, GitHub source) before any loader code was written. Two things this caught: BrowseComp's answer set isn't on Hugging Face at all — it's a CSV on OpenAI's blob storage — and its per-row XOR decrypt scheme has an exact, publicly documented implementation worth reproducing verbatim rather than approximating.

| Source | Confirmed | Where |
|---|---|---|
| FRAMES | TSV columns: `Prompt`, `Answer`, 11× wiki-link cols, `reasoning_types` | huggingface.co/datasets/google/frames-benchmark |
| BrowseComp | CSV cols `problem, answer, problem_topic, canary`; SHA256-derived XOR cipher | github.com/openai/simple-evals/browsecomp_eval.py |
| ResearchRubrics | JSONL: `sample_id, prompt, domain, rubrics[{criterion,weight,axis}]` | huggingface.co/datasets/ScaleAI/researchrubrics |
| MuSiQue | JSONL: `id, question, answer, answer_aliases, answerable` | huggingface.co/datasets/dgslibisey/MuSiQue |
| HotpotQA | Parquet (distractor config): `id, question, answer, type, level` | huggingface.co/datasets/hotpotqa/hotpot_qa |

### Eval harness & scripts

#### `evals/graders.py`, `judge.py`, `stats.py`, `runner.py` — grading

Normalized exact-match + token-F1 for verifiable sources (FRAMES/MuSiQue/HotpotQA). An LLM-judge for BrowseComp — OpenAI's published grader prompt, unmodified — and for ResearchRubrics, which scores each rubric criterion independently and reports a weighted-compliance score. `run_benchmark()` dispatches by source, bounded by a concurrency semaphore. Bootstrap 95% CIs wrap every reported score.

- **Why it matters:** These test sets are small (hundreds of questions) — a point-estimate accuracy alone overstates precision. The CI is what makes a "post-GRPO improved by 3 points" claim defensible in the thesis rather than noise.
- **Why this decision:** BrowseComp's grader prompt is copied verbatim from OpenAI's own eval code so scored accuracy is comparable to published baselines, not a homegrown approximation.

#### `scripts/run_baseline.py` vs. `run_milestone_eval.py` — CLI

Two entry points with different blast radii: `run_baseline.py` for fast dev iteration on any sample size, no gate. `run_milestone_eval.py` requires an explicit `--milestone {baseline,post-dspy,post-grpo}` flag, writes a split manifest per run, and is the only script meant to touch a full benchmark test set.

- **Why it matters:** This is the plan's leakage-prevention promise made mechanical rather than a rule someone has to remember — you cannot "peek" at test performance without typing the milestone name.
- **Why this decision:** A `--smoke-limit` flag exists for wiring checks, but prints an explicit warning that the run doesn't count as an official number — the script itself won't let a shortcut masquerade as a real result.

**Bug found while writing tests, fixed.** `run_benchmark()`'s per-question `try/except` originally wrapped only the pipeline call, not the grading step. A misconfigured judge (or any grading-time exception) would propagate up through `asyncio.gather` and crash the entire benchmark run — one bad question taking down a 200-question milestone eval. Caught by a test asserting that a missing judge produces an `error`-metric record for that one question rather than an unhandled exception. Fixed by widening the try block to cover grading too; the test now asserts on that behavior directly.

---

## Verification

```
uv run ruff check . && uv run mypy . && uv run pytest tests/
```

All green.

**Next:** Phase 2 ports this MAF loop into a DSPy program and compiles prompts via MIPROv2 against the train-pool split, evaluating the frozen result at the `post-dspy` milestone.
