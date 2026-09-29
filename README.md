# mas-deepr

**Implementation of a modular framework for optimization and evaluation of
small-language-model-based multi-agent systems** (engineering thesis,
Centre for Credible AI, Warsaw University of Technology, 2026/2027 —
Maksim Razantsau, Oleksii Vinichenko; thesis document kept local, not
tracked in this repo).

Multi-agent SLMs (4B-20B) doing deep research/web browsing, optimized via
DSPy prompt compilation and RL/alignment (DPO/GRPO), aiming to approach
frontier-LLM accuracy on research benchmarks at a fraction of the
inference cost — with automated evaluation and cost/latency profiling
against commercial baselines at every step.

## Framework structure (thesis elements 1-4)

| Element | Module | Status |
|---|---|---|
| 1. Multi-agent system | `agents/`, `tools/`, `prompts/` | Done — Manager→Browser→Synthesizer over one shared SLM, cached/retried search+fetch tools |
| — memory/context strategies (extension) | `memory/` | Done — pluggable strategy registry (stateless/folding/RE-TRAC) wrapping the pipeline in a multi-pass loop |
| 2. Pipeline optimization | `optimize/` | Done — DSPy/MIPROv2 prompt compilation against a train-only pool |
| 3. RL/Alignment module | `rl/` | DPO + T3S cold-start SFT: real, **verified** training runs (`tests/test_dpo.py`, `tests/test_cold_start.py`). GRPO via OpenPipe ART, default backend `ServerlessBackend` (trains on W&B infra, no local GPU) with `--backend local` fallback; **unverified** — no W&B Training account here |
| 4. Evaluation & profiling | `evals/`, `telemetry/` | Done — EM/F1 + LLM-judge grading, bootstrap CIs + paired A/B, per-question latency/cost, frontier-model comparison (GPT-4.1 / Gemini 2.5 Pro / DeepSeek through the same pipeline), accuracy + cost/latency charts, optional W&B mirror |

See `docs/plan/plan-v2.md` for the full design rationale and milestone
table, and `docs/phase-0-1-report.md` for why Phase 0/1 is built the way
it is.

## Quickstart

```bash
uv sync --dev              # core stack (agents, DSPy, DPO/cold-start SFT)
uv sync --dev --extra rl-art  # + OpenPipe ART, for GRPO (needs GPU infra to run)
uv run ruff check . && uv run mypy . && uv run pytest tests/
```

See `docs/running-phase-1.md` (inference endpoint, API keys, `.env`) and
`docs/running-phase-2.md` (DSPy compilation) before running any script
against live models.

### Running the pipeline

```bash
# Dev-loop iteration, any sample size, no gating:
uv run python scripts/run_baseline.py --model qwen3-8b --benchmark frames \
    --limit 20 --memory folding

# Official milestone eval (gated, writes a train/val manifest):
uv run python scripts/run_milestone_eval.py --milestone baseline \
    --models qwen3-8b --memory retrac
```

`--memory {stateless,folding,retrac}` selects a `memory/` strategy;
default (`stateless`) is byte-identical to the pre-Phase-3 pipeline.

### Baseline vs. frontier models + plots

```bash
# needs OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY in .env
uv run python scripts/run_milestone_eval.py --milestone baseline \
    --models qwen3-8b,gpt-4.1,gemini-2.5-pro,deepseek-chat

uv run python scripts/plot_milestone.py --milestone baseline
# -> runs/milestones/baseline/plots/{accuracy_comparison.png,
#    cost_latency.png, comparison_table.{md,html}}
```

The frontier models run the *same* Manager->Browser->Synthesizer
pipeline (swapped `ModelSpec`), so the comparison isolates the model
choice from the framework. See `docs/eval-architecture.md`.

### RL/Alignment

```bash
# T3S anchor-masked SFT cold start (verified — runs on CPU/GPU, no serving infra):
uv run python scripts/train_cold_start.py --model-name Qwen/Qwen3-4B \
    --examples examples.jsonl --output-dir runs/cold_start/run1

# DPO from offline preference pairs (verified, same way):
uv run python scripts/train_dpo.py --model-name Qwen/Qwen3-4B \
    --pairs pairs.jsonl --output-dir runs/dpo/run1

# GRPO via ART -- default runs training + inference on W&B infra (no local GPU;
# needs WANDB_API_KEY + W&B Training access). --backend local for your own GPU.
uv run python scripts/train_grpo.py --base-model Qwen/Qwen3-4B
```

## Observability

- **Telemetry** (`telemetry/`): every LLM call — manager/browser/
  synthesizer/judge/compressor — logs tokens, latency, and cost to a JSONL
  ledger. `telemetry.summarize()` for one run; `telemetry.load_runs(root)`
  for a cached cross-run DataFrame. Never logs prompt/response content
  (privacy). This is the canonical record.
- **W&B** (optional mirror): set `WANDB_API_KEY` and eval + training runs
  land in one `mas-deepr` project. A no-op without the key. Never a source
  of truth. See `docs/wandb-art-integration.md`, `docs/eval-architecture.md`.
- **Logging** (`logging_config.py`): human-readable progress/debug output
  (cache hits, retries, role start/end) separate from telemetry's
  permanent cost ledger. Call `configure_logging()` once per script;
  every entry point under `scripts/` already does.

## Layout

```
src/mas_deepr/
  agents/      Manager -> Browser -> Synthesizer MAF pipeline
  memory/      pluggable context/memory strategies (Phase 3)
  optimize/    DSPy/MIPROv2 prompt compilation (Phase 2)
  rl/          T3S cold-start SFT, DPO (TRL), GRPO (ART) (Phase 4)
  tools/       cached+retried search/fetch, sandboxed code exec
  data/        FRAMES/BrowseComp/ResearchRubrics/MuSiQue/HotpotQA loaders
  evals/       EM/F1 + LLM-judge grading, bootstrap CIs, benchmark runner
  telemetry/   per-call cost/latency/token ledger
  config/      settings, model registry
  prompts/     hand-written + DSPy-compiled prompt templates
scripts/       CLI entry points (baseline/milestone eval, compile, train_*)
docs/          plan + phase reports + runbooks
tests/         one test module per source module; `-m art` isolates the
               OpenPipe ART test (see pyproject.toml for why)
```
