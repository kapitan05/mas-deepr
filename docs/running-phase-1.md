# Running Phase 1: the MAS research pipeline

Phase 1 is the working baseline: Manager → Browser → Synthesizer agents on
one shared SLM, backed by cached search/fetch/code-exec tools, scored
against FRAMES, BrowseComp, and ResearchRubrics.

## 1. Prerequisites

```bash
uv sync --dev
```

### An OpenAI-compatible inference endpoint

The pipeline talks to whatever `MAS_SLM_BASE_URL` points at — it doesn't
care whether that's local vLLM, Ollama, or a remote box, as long as it
speaks the OpenAI chat-completions API. For dev, the simplest option is
vLLM serving a Qwen3 model locally:

```bash
vllm serve Qwen/Qwen3-8B --port 8000
```

### Environment variables

Copy these into a `.env` file at the repo root (all `MAS_`-prefixed fields
are in `src/mas_deepr/config/settings.py`):

```bash
# Policy SLM endpoint (vLLM/Ollama/etc. — must be OpenAI-compatible)
MAS_SLM_BASE_URL=http://localhost:8000/v1
MAS_SLM_API_KEY=EMPTY
MAS_SLM_MODEL=Qwen/Qwen3-8B

# Judge model, used to grade BrowseComp and ResearchRubrics
MAS_JUDGE_MODEL=gpt-5-mini
MAS_JUDGE_BASE_URL=          # leave unset to hit OpenAI's default endpoint
OPENAI_API_KEY=sk-...        # NOT MAS_-prefixed — see note below

# Web search tool
TAVILY_API_KEY=tvly-...      # NOT MAS_-prefixed — see note below
```

`OPENAI_API_KEY` and `TAVILY_API_KEY` are read under their own conventional
names rather than `MAS_JUDGE_API_KEY` / `MAS_TAVILY_API_KEY`, so existing
`OPENAI_API_KEY`/`TAVILY_API_KEY` exports from your shell are picked up
automatically without duplicating them.

You only need `OPENAI_API_KEY` if you're running the `browsecomp` or
`research_rubrics` benchmarks (both require a judge). `frames` alone needs
neither the judge nor Tavily's paid tier headroom, since it's graded by
exact match.

## 2. Fast dev-loop iteration

`scripts/run_baseline.py` — no gating, any sample size, for iterating on
prompts/tools without burning a full benchmark run:

```bash
uv run python scripts/run_baseline.py \
  --model qwen3-8b \
  --benchmark frames \
  --limit 20
```

| Flag | Required | Notes |
|---|---|---|
| `--model` | yes | Registry key: `qwen3-4b`, `qwen3-8b`, `qwen3-14b`, `gpt-oss-20b`, `gpt-5-mini` |
| `--benchmark` | yes | `frames` \| `browsecomp` \| `research_rubrics` |
| `--limit` | no (default 20) | Question count pulled from the benchmark |
| `--judge-model` | no (default `gpt-5-mini`) | Only used for `browsecomp`/`research_rubrics` |

Output: a score with a 95% bootstrap CI printed to stdout, plus
`runs/dev-<id>/{telemetry.jsonl,<benchmark>_results.parquet}`.

## 3. Official milestone runs

`scripts/run_milestone_eval.py` is the **only** script that should touch a
full benchmark test set — it requires an explicit `--milestone` flag and
writes a split manifest per run, so there's no accidental "peeking" at test
performance.

```bash
uv run python scripts/run_milestone_eval.py \
  --milestone baseline \
  --models qwen3-4b,qwen3-8b,qwen3-14b,gpt-oss-20b
```

| Flag | Required | Notes |
|---|---|---|
| `--milestone` | yes | `baseline` \| `post-dspy` \| `post-grpo` |
| `--models` | yes | Comma-separated registry keys |
| `--benchmarks` | no (default: all three) | Comma-separated benchmark names |
| `--smoke-limit` | no | Caps questions/benchmark for wiring checks — **not** a valid thesis number; the script prints a warning when set |

`--milestone baseline` runs the hand-written prompts. `post-dspy` and
`post-grpo` automatically switch to whichever compiled prompt variant
exists in the registry (see `docs/running-phase-2.md`) — no code change
needed, just run the milestone once Phase 2/4 output exists.

Output, per `{milestone}/{model}/`:
- `telemetry.jsonl` — every LLM call's tokens/latency/cost
- `{benchmark}_manifest.json` — exact question ids evaluated (audit trail)
- `{benchmark}_results.parquet` — per-question scores
- `summary.parquet` (one level up) — mean score + 95% CI per model × benchmark

## 4. Inspecting results

```python
import polars as pl
from mas_deepr.telemetry import summarize

pl.read_parquet("runs/milestones/baseline/summary.parquet")
summarize(Path("runs/milestones/baseline/qwen3-8b/telemetry.jsonl"))  # cost/latency rollup
```

## Notes

- All web search/fetch calls are cached in `assets/web_cache.sqlite3`
  (`MAS_CACHE_DB` if you want a different path) — reruns of the same
  question are cheap and reproducible, which matters for BrowseComp since
  live-web answers could otherwise drift between milestones.
- A dead link or paywall never aborts a run: `fetch_page` returns a marked
  empty string instead of raising.
- `--milestone` runs pull the **entire** benchmark unless you pass
  `--smoke-limit`, so a first-time `research_rubrics`/`browsecomp` run will
  make one judge call per question — budget for that.
