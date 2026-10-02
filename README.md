<div align="center">
<img src="assets/mas-deepr-logo.jpg" alt="mas-deepr" width="420"/>

# mas-deepr

**A modular framework for optimizing and evaluating small-language-model
multi-agent systems on deep-research tasks**

[**Thesis Report**](https://img.shields.io/badge/thesis_report-WIP-yellow) • [**Models**](https://img.shields.io/badge/models-coming_soon-lightgrey) • [**Blog**](https://img.shields.io/badge/blog-coming_soon-lightgrey) • [**Interactive Demo**](https://img.shields.io/badge/demo-coming_soon-lightgre)
</div>

</div>

---

## Release Notes

- **2026-09-30** — GRPO training run completed: 340+ gradient steps on
  Qwen3-14B via OpenPipe ART
  (see Results below).

---

## Overview

This repository contains four main modules:

- **[`agents/`](src/mas_deepr/agents/) + [`tools/`](src/mas_deepr/tools/) + [`memory/`](src/mas_deepr/memory/)** —
  the multi-agent system: a Manager→Browser→Synthesizer pipeline over one
  shared SLM, an MCP-based tool backend, and a pluggable memory/context
  strategy layer.
- **[`optimize/`](src/mas_deepr/optimize/)** — DSPy/MIPROv2 prompt
  compilation, tuning the pipeline's instructions against a train-only
  question pool.
- **[`rl/`](src/mas_deepr/rl/)** — RL/Alignment: GRPO (OpenPipe ART), DPO
  and T3S cold-start SFT (Hugging Face TRL).
- **[`evals/`](src/mas_deepr/evals/) + [`telemetry/`](src/mas_deepr/telemetry/)** —
  automated evaluation (LLM-as-judge + deterministic grading) and
  cost/latency/accuracy profiling against commercial LLMs.

---

## Key Features

- **MCP tool backend** (FastMCP server, 6 tools): web search, page
  fetch, Wikipedia, Semantic Scholar, PubMed, Tavily — rate-limited and
  cached server-side.
- **Pluggable memory/context strategies** (`stateless` / `folding` /
  `retrac`) — swap how much cross-branch or cross-pass context the
  agents see, with zero other code changes.
- **DSPy/MIPROv2 prompt compilation** — automatic instruction tuning
  against a held-out train pool.
- **RL** — recipe scales with model size:
  - *LoRA + GRPO* (done, 14B): group-relative advantage, one shared
    adapter across all roles, OpenPipe ART (serverless/local).
  - *SFT → DPO/GRPO* (planned, full-parameter, 8B/14B on HPC): T3S
    anchor-masked cold-start SFT (TRL), DPO/GRPO refinement.
  - *LoRA/QLoRA + GRPO* (planned, 70B+): same recipe, adapter-only once
    full-parameter is infeasible at scale.
  - Reward = LLM-judge rubric score ± opt-in turn-count/hallucination
    penalties, for any of the above.
- **Evaluation**: EM + LLM-judge grading, bootstrap CIs, paired A/B,
  cost/latency per question — vs. GPT-4.1/Gemini/DeepSeek on the same
  pipeline.

---

## Architecture

```
                 ┌─────────────────────────┐
  question  ───▶ │        Manager          │
                 └────────────┬────────────┘
                               │ sub-questions
                 ┌─────────────▼────────────┐
                 │   memory/ strategy layer  │  (stateless / folding / retrac)
                 └─────────────┬────────────┘
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        ┌─────────┐      ┌─────────┐       ┌─────────┐
        │ Browser │      │ Browser │  ...  │ Browser │   ◀── MCP tools
        └────┬────┘      └────┬────┘       └────┬────┘      (FastMCP)
             └─────────────────┼─────────────────┘
                               ▼
                      ┌─────────────────┐
                      │   Synthesizer    │ ──▶ final answer
                      └─────────────────┘

        one shared SLM (Qwen3) plays every role, trained as a single
        LoRA adapter via GRPO over the whole pipeline's trajectories
```

---

## Results

*LoRA + GRPO* (Qwen3-14B):

| Benchmark | Base model | Best GRPO checkpoint | Δ |
|---|---|---|---|
| ResearchQA | 24.3% | **68.6%** | **+44.3pp** |
| HealthBench | 20.4% | **47.2%** | **+26.8pp** |
| FRAMES | 14.0% | 18.0% (stateless) | ~flat, see below |

GRPO training clearly helps ResearchQA and HealthBench. FRAMES needs cross-sub-question context (folding memory strategy).

| Memory strategy | FRAMES accuracy | Errors |
|---|---|---|
| `stateless` (default) | 14% | 1 |
| `folding` | **24%** | **0** |

Training longer doesn't fix a broken pipeline design. Here, picking the right fix (sharing context between sub-questions) mattered more than just running more training steps.

---

## Quickstart

```bash
uv sync --dev                 # core stack (agents, DSPy, DPO/cold-start SFT)
uv sync --dev --extra rl-art  # + OpenPipe ART, for GRPO
uv run ruff check . && uv run mypy . && uv run pytest tests/
```


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

The frontier models run the *same* Manager→Browser→Synthesizer
pipeline (swapped `ModelSpec`), so the comparison isolates the model
choice from the framework.
### RL

```bash
# SFT cold start (runs on CPU/GPU, no serving infra):
uv run python scripts/train_cold_start.py --model-name Qwen/Qwen3-4B \
    --examples examples.jsonl --output-dir runs/cold_start/run1

# DPO from offline preference pairs:
uv run python scripts/train_dpo.py --model-name Qwen/Qwen3-4B \
    --pairs pairs.jsonl --output-dir runs/dpo/run1

# GRPO via ART -- verified, real training run completed on this
uv run python scripts/train_grpo.py --base-model OpenPipe/Qwen3-14B-Instruct \
    --num-steps 500 --batch-size 30 --group-size 2

# Evaluate a checkpoint (any memory strategy), then chart progress over steps:
uv run python scripts/eval_grpo_checkpoint.py --model-name mas-deepr-grpo \
    --step 300 --memory folding
uv run python scripts/watch_grpo_progress.py --model-name mas-deepr-grpo
```

### Interactive Demo (WIP)

```bash
# NOT YET IMPLEMENTED
uv run python scripts/launch_chat.py --model-name mas-deepr-grpo --step latest
```

---

## Roadmap

**Planned, not yet implemented**: full-scale SFT → DPO/GRPO training on
a university SLURM HPC cluster (up to 16×A100), reusing production RL
infrastructure (LLaMA-Factory, Ray, vLLM, DeepSpeed, open-instruct)
instead of W&B-hosted serverless training, and scaling toward 70B+
models via LoRA/QLoRA once the current pipeline's recipe is validated
at 14B.

---

## Observability

- **Telemetry** (`telemetry/`): every LLM call — manager/browser/
  synthesizer/judge/compressor — logs tokens, latency, and cost to a
  JSONL ledger. Never logs prompt/response content (privacy).
- **W&B** (optional mirror): set `WANDB_API_KEY` and eval + training
  runs land in one `mas-deepr` project.
- **Logging** (`logging_config.py`): human-readable progress/debug
  output, separate from telemetry's permanent cost ledger.

---

## Repo Layout

```
src/mas_deepr/
  agents/      Manager -> Browser -> Synthesizer MAF pipeline
  memory/      pluggable context/memory strategies (stateless/folding/retrac)
  optimize/    DSPy/MIPROv2 prompt compilation
  rl/          T3S cold-start SFT, DPO (TRL), GRPO (ART)
  tools/       cached+retried search/fetch, sandboxed code exec
  data/        FRAMES/BrowseComp/ResearchRubrics/ResearchQA/HealthBench/
               MuSiQue/HotpotQA loaders
  evals/       EM/F1 + LLM-judge grading, bootstrap CIs, benchmark runner
  telemetry/   per-call cost/latency/token ledger
  config/      settings, model registry
  prompts/     hand-written + DSPy-compiled prompt templates
scripts/       CLI entry points: baseline/milestone eval, compile, train_*,
               eval_grpo_checkpoint, watch_grpo_progress, auto_eval_loop
docs/          plan + phase reports + runbooks
tests/         one test module per source module; `-m art` isolates the
               OpenPipe ART test (see pyproject.toml for why)
```

---

## Acknowledgments

Developed as a registered engineering thesis at the Warsaw University of Technology, under the
supervision of **Dr. Inez Okulska**.

---

## Team & Contact

**Maksim Razantsau**, **Oleksii Vinichenko** 
