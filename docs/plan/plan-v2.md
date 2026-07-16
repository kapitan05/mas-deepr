# SLM Multi-Agent Deep Research: Plan v2 — Memory/Context Module + Revised GRPO

## Context
Thesis project (`mas-deepr`): multi-agent SLMs (4B–20B) optimized via DSPy prompt compilation + GRPO weight training, approaching frontier-LLM performance on deep-research/browsing benchmarks at a fraction of cost. **Phases 0-2 are done and merged** (env/config/telemetry scaffold; 3-agent MAF pipeline with tools/prompts/eval harness; DSPy/MIPROv2 prompt compilation) — 73+ tests passing, CI green, PRs #4/#5 open on `main`.

This revision was prompted by the user adding 9 ICML-2026-era papers to `papers/` and asking how to extend the system to test cutting-edge methodologies (agent memory/context management, RL post-training tricks) under a hard **~1× H100** compute budget, and specifically: should development proceed as a general framework, or paper-by-paper? Three research passes (paper extraction ×2 themes, current-architecture re-audit) plus one design pass ground everything below in the actual current code, not assumptions.

**Verdict up front:** neither extreme. The repo already has three live, proven registry/extension seams (models, tools, prompts — each "add one entry, touch nothing else," backed by 73+ passing tests). That's real evidence a fourth axis — pluggable context/memory *strategies* — belongs in the same family, sized to 2-3 concretely scoped papers, not an abstract "memory framework" with no forcing function, and not a one-off script that throws away Phases 0-2's infrastructure. One caveat: a memory strategy isn't a leaf value like a tool or prompt — it wraps `run_pipeline`'s entire multi-call control flow, so it's a registry of strategy *objects with hooks*, not of terminal values (see Module Design below).

## Papers reviewed (papers/, 9 PDFs)
| Paper | One-line takeaway | Where it lands |
|---|---|---|
| RE-TRAC | Recursive trajectory→structured-state compression across N full rollout passes; training-free mode alone gives +15-20% over ReAct on BrowseComp at ~50% the tokens/tool-calls of majority-voting; reported at 4B/~3B-active scale | **Phase 3, primary strategy** |
| Context-Folding | `branch()`/`return()` tools fold sub-context into a summary; current mas-deepr Manager→Browser split is already *accidentally* isomorphic to this, just blind (siblings never see each other) | **Phase 3, primary strategy** |
| AMA-Bench / AMA-Agent | Causality-graph + tool-augmented-retrieval memory; beats prior memory systems +11-27pts; memory *architecture* matters far more than model size (8B→32B only +0.038). User's own original plan already named this as a stretch goal | **Phase 3, stretch strategy (M4, time-boxed)** |
| Reuse your FLOPs (PrefixRL) | RL-efficiency trick: condition on off-policy trace prefixes, mask gradient on them, train on-policy. Right model scale (4B/8B) but domain-adjacent (verifiable math reasoning) and needs a live RL loop | **Phase 4, deferred stretch** |
| T3S | SFT trick: mask "anchor tokens" (early-converged, gradient-suppressing) from distillation loss. Cheapest paper by far (hundreds of examples, 50 steps) | **Phase 4, cold-start upgrade** |
| DR Tulu (RLER) | Same task class as mas-deepr (multi-role deep-research agent); reward = judge score against an evolving rubric pool. Self-hosted 8B judge only loses ~1pt vs GPT-4.1. Paper's own RL budget (~27,000 GPU-hrs) is categorically out of reach — used only as a *design pattern*, heavily scoped down | **Phase 4, reward function** |
| S2L-PO | Sample some GRPO group members from a frozen smaller sibling model (parameter-compression diversity > temperature noise); cheapest published GRPO config | **Phase 4, rollout add-on** |
| TULU 3 | Frontier-lab-scale post-training recipe; not reproducible at 1 GPU, used only as a reference for 3 sub-patterns (RLVR-as-verifier ≈ our EM/F1 reward; on-policy DPO warm-start; per-role data ablation) | **Design reference only** |
| EAM (diffusion super-res) | Image super-resolution via diffusion transformers — no connection to LLM agents/RL | **Out of scope, excluded** |

## Grounding notes (confirmed against current source)
- `agents/topology.py`: `PipelineResult = {question_id, question, sub_questions, findings, final_answer}` — nothing else survives a run. `run_pipeline` is single-pass, stateless-by-construction; `_run_agent` only passes `prompt` + `max_function_calls` to `agent.run()`. MAF's `session`/`context_providers`/`compaction_strategy` params exist on `Agent` but are unused.
- **Confirmed (spiked live, not assumed): `AgentResponse.messages: Sequence[Message]` and each `Message.contents` carries the full tool-call trajectory** (search queries, fetch results) — `_run_agent` currently discards all of this and keeps only `.text`. Capturing `resp.messages` is a small, low-risk change that makes real tool-call trajectories available, not just final-answer strings.
- `telemetry/tracker.py::LLMCallRecord.phase` is free-text (already includes `post-grpo`) — new fields/values are additive, non-breaking.
- `evals/judge.py::JudgeClient` already has the exact shape needed for reward reuse (`grade_browsecomp`, `grade_research_rubrics`, both funneling through the same `_run`/`TelemetryTracker` pattern as `_run_agent`); `_parse_rubric_verdicts`'s "extract JSON array, default-false on failure" parsing is directly reusable for RE-TRAC's `StructuredState` and AMA's graph triples.
- `optimize/modules.py::ResearchProgram` is deliberately single-pass/non-tool-calling by design (its own docstring: DSPy's ReAct protocol isn't portable to MAF) — memory strategies operate on exactly the multi-pass/multi-branch control flow DSPy opted out of. **No DSPy-mirror change needed** to ship Phase 3.
- **Polar (original Phase 3) was never built** — every "polar" hit in the repo is a false-positive match on the `polars` dataframe library. No in-flight work to preserve; the phase is being replaced, not interrupted.
- `tools/cache.py::WebCache` (SQLite, thread-locked, `make_key`) is the template for any new graph/embedding store.

## Phase 3 (NEW): Memory/Context Strategy Module

### Module design — `src/mas_deepr/memory/`
```
memory/
  __init__.py       # MEMORY_REGISTRY: dict[str, strategy factory] — mirrors MODEL_REGISTRY's shape
  base.py           # MemoryStrategy protocol + BranchTrajectory/MainThreadState/PassRecord dataclasses
  baseline.py       # StatelessStrategy — explicit no-op, byte-identical to today's behavior
  folding.py        # ContextFoldingStrategy
  retrac.py         # TrajectoryCompressionStrategy
  state.py          # StructuredState (shared compression schema)
  ama_graph.py       # CausalGraphStrategy (stretch, M4)
  graph_store.py     # GraphStore, mirrors WebCache (stretch, M4)
prompts/templates/compressor.yaml   # new "compressor" role prompt
```

**Key interface decision: one `MemoryStrategy` protocol, two hook scopes, not two separate interfaces.** RE-TRAC operates across N full `run_pipeline` passes (outer loop); Context-Folding operates across Browser's per-sub-question branches within one pass (inner loop). A single strategy exposing both hook types (each strategy overrides only what it needs, defaulting the rest to identity) is right because: (1) `run_pipeline` gets exactly one call-site contract, matching the existing registries' single-interface pattern; (2) AMA-Agent genuinely needs both scopes at once (update the graph on every branch return, consult/prune it across passes) — a two-interface split would force it into two unrelated classes; (3) `StatelessStrategy` (every hook = identity/no-op) becomes the regression test that the abstraction is lossless.

```python
class MemoryStrategy(Protocol):
    name: str
    max_passes: int  # 1 for Folding/AMA-inner-only; 4-8 for RE-TRAC

    async def prepare_pass(self, *, question, prior_passes: list[PassRecord], pass_index: int) -> str: ...
    async def prepare_branch(self, *, sub_question: str, main_thread: MainThreadState) -> str: ...
    async def fold_branch(self, *, sub_question, raw_finding, main_thread) -> str: ...
    async def finalize_pass(self, *, result: PipelineResult, main_thread) -> PassRecord: ...
```

**Changes to `agents/topology.py`:**
- `run_pipeline(..., memory: MemoryStrategy | None = None)` — `None` preserves today's exact behavior (existing call sites in `evals/runner.py`, `tests/test_pipeline_smoke.py` need no changes). When set, wrap the existing manager→browser*→synthesizer body in `for pass_index in range(memory.max_passes)`, injecting `prepare_pass`/`prepare_branch` text and calling `fold_branch`/`finalize_pass` at the right points.
- `PipelineResult` grows three additive fields: `raw_findings: list[str]`, `pass_records: list[PassRecord]`, `memory_strategy: str = "stateless"`.
- `ResearchPipeline` gets one new field: `compressor: Agent | None`, built alongside manager/browser/synthesizer (reusing the same client `build_pipeline` already constructs) whenever a strategy needs compression calls.
- `_run_agent`/`LLMCallRecord`/`TelemetryTracker.record()` get additive optional `memory_strategy`/`pass_index` fields.
- `_run_agent` starts capturing `resp.messages` (not just `.text`) into `BranchTrajectory` — this is the concrete change that makes trajectories real and also the prerequisite Phase 4's rollout capture needs, so sequencing this first avoids rework there.

**Manager fast-path (new, scoped small — addresses a concrete DR Tulu critique):** DR Tulu's authors explicitly critique fixed-pipeline deep-research systems for "always producing long reports even for simple factoid questions" and note such systems "are unable to follow instructions to only output the answer" on short-form QA. Today's `run_pipeline` has exactly that failure mode — it always decomposes, always browses, always synthesizes, even for a question with no research need. Fix: extend `prompts/templates/manager.yaml` so the Manager may respond `DIRECT: <answer>` instead of a sub-question list when a question needs no research; `run_pipeline` checks for this prefix immediately after the manager call and, if present, sets `final_answer` directly and returns — skipping the browser/synthesizer loop (and any `memory` strategy wrapping) entirely for that question. This check happens *before* any memory-strategy hook runs, so it composes cleanly with all of Phase 3's strategies rather than needing per-strategy handling. Bundled into M1 (touches the same `run_pipeline`/prompt surface, no new milestone needed); its benefit is directly measurable via the same tokens/calls efficiency metric already planned for the memory-strategy A/B (fewer LLM calls on simple questions, visible in `telemetry.summarize()`). DSPy-mirror update (`optimize/signatures.py::ManagerSignature`) is a natural follow-up, not required to ship this.

**Per-strategy summary:**
- **`retrac.py`** (`max_passes=4-8`): `finalize_pass` calls the new `compressor` Agent on the full pass trajectory, parses into `StructuredState{answer, evidence, analysis, uncertainties, failed_attempts}` (same JSON-array-with-fallback parsing as `_parse_rubric_verdicts`); `prepare_pass` renders the prior pass's state as a prefix with the paper's "treat as hypothesis, verify or override" instruction. Most expensive to *run* (N× LLM calls/question), zero training needed.
- **`folding.py`** (`max_passes=1`): `prepare_branch` renders prior sub-question→finding pairs from `main_thread.folded` as context for the next Browser call — the concrete delta over today (siblings currently never see each other). v1 `fold_branch` is identity (no new LLM call); cheapest strategy to build and run.
- **`ama_graph.py`** (stretch, M4): `GraphStore` mirrors `WebCache`; `fold_branch` parses `(prev_state, action, obs)` triples into graph nodes/edges via the compressor Agent; `prepare_branch` does top-K embedding retrieval + self-eval fallback to graph traversal. Needs a genuinely new capability (an embedding client — no equivalent exists today). Correctly scoped as optional/time-boxed: heaviest lift, and only the retrieval *mechanism* is adopted, not AMA-Bench's own benchmark suite.

### Evaluation — reuses existing infrastructure, no new benchmark work
- `evals/runner.py::run_benchmark` gets a `memory: MemoryStrategy | None` param threaded to `run_pipeline`; `EvalRecord`/`records_to_df`/`write_results` untouched.
- Primary signal: FRAMES + MuSiQue/HotpotQA exact-match/F1 (cheap, no judge cost), small slices (50-150 Qs) per strategy for fast iteration.
- Secondary: BrowseComp/ResearchRubrics via `JudgeClient`, sparingly (judge cost).
- Efficiency signal (RE-TRAC's own headline metric): `telemetry.summarize()` grouped by the new `memory_strategy`/`pass_index` fields — tokens/calls/latency per strategy, directly reproducing "fewer tokens at equal-or-better accuracy."
- Comparison: `evals/stats.py::bootstrap_ci` per strategy (already exists, no new code).
- **Explicitly out of scope**: reproducing AMA-Bench's synthetic QA generator/task suite; any new benchmark; RE-TRAC's SFT/distillation variant (deferred — feeds Phase 4's T3S cold start instead).

## Phase 4 (REVISED): GRPO

**RL backend evaluated and confirmed: OpenPipe ART, not veRL/verl-agent.** A proposal to restructure Phase 3/4 around veRL+verl-agent (Ray orchestration, FSDP2 sharding, gym-style env interfaces) was considered and rejected on evidence, not preference: veRL's own hardware-tuning docs cap tested single-GPU (1×H100) GRPO at ≤3B params — below the locked `qwen3-4b` target already in `MODEL_REGISTRY`; verl-agent's own README states its reference config needs 2×H100 for a 7B model. ART has direct precedent at this exact task and budget instead (its own "Open Deep Research" tutorial trains a 14B deep-research agent via GRPO in 30hrs/~$350 on hardware sized for one rented GPU). Separately, adopting verl-agent's gym-style (`reset`/`step`) environment interface would require reimplementing Manager→Browser→Synthesizer a second time as a step-based loop — a real parallel-orchestration cost (precedented in this repo by why `optimize/modules.py::ResearchProgram` couldn't just reuse MAF's tool-calling loop for DSPy either), the opposite of the "wrap `run_pipeline` unchanged, point `Settings.slm_base_url` at a different server" zero-rework path ART already gives for free. Revisit only if compute budget genuinely grows past 1-2 GPUs.

**Polar (NVIDIA, "Agentic RL on Any Harness at Scale") also evaluated and rejected, same category as veRL, plus one deeper finding.** Polar is a real, separate system from the `polars` dataframe library — a proxy that sits at the LLM API boundary (same indirection point as `Settings.slm_base_url`), transparently capturing token-faithful trajectories from an *unmodified* harness via two reconstruction strategies (`per_request`: one trace per LLM call; `prefix_merging`: merges calls into coherent chains only where a strict token-prefix relationship holds — correctly keeping genuinely distinct roles like Manager/Browser/Synthesizer as separate chains rather than forcing one global trace). Rejected for the same compute-scale reason as veRL: it's cluster infrastructure (rollout server + gateway nodes), validated with an 8×H100 tensor-parallel deployment serving a 122B model, and its own paper says plainly *"Polar is not a replacement trainer... a rollout substrate that can feed asynchronous trainers"* — it would sit **in addition to**, not instead of, ART, solving a data-capture problem ART's own SDK already solves at 1-GPU scale. **The deeper finding, load-bearing for `rubric_reward.py` below**: Polar's own authors ablated naive outcome-reward broadcast across multiple traces in one session and got real reward hacking, calling it *"noisy credit assignment... outside the scope of this work, on our roadmap"* — i.e. even NVIDIA's own system does not solve the exact problem this plan needs solved (reward flowing correctly across Manager/Browser*/Synthesizer calls sharing one policy). Confirms `rubric_reward.py`'s evolving-rubric reward is doing real, unavoidable work, not a nice-to-have.

`src/mas_deepr/rl/` (currently empty) populated in this order:
```
rl/
  cold_start.py     # T3S: anchor-token-masked SFT from role-tagged trajectories
  rubric_reward.py  # scoped DR-Tulu-style evolving rubric pool, built on evals/judge.py
  rollout.py        # ART rollout fn: wraps agents.topology.run_pipeline at Settings.slm_base_url
  s2l_po.py         # LoRA-adapter-swap sibling-model rollout diversity
scripts/train_grpo.py
```
- **`cold_start.py` (T3S)**: consumes `PipelineResult.raw_findings`/`pass_records` (now captured thanks to Phase 3) to build a few hundred role-tagged distillation examples/role, masks anchor tokens from SFT loss. Paper's own config (batch 64, lr 1e-5, 50 steps, hundreds of examples, single 8B run) directly reproducible on 1×H100.
- **`rubric_reward.py` (DR Tulu pattern)**: extract a shared `_grade_criteria(prompt, criteria, response) -> list[bool]` helper inside `evals/judge.py` used by both the existing fixed-rubric grader and this dynamic pool (positive/negative rubrics, pruned by zero-variance-across-group, capped at `K_max`). Self-hosted policy model as judge (paper's own ablation: ~1pt loss vs GPT-4.1) to keep cost near zero. **Scoped drastically down** from DR Tulu's own ~27,000 GPU-hr run — that number is context, not a target.
- **`rollout.py`**: wraps `run_pipeline` pointed at ART's endpoint via `Settings.slm_base_url` — the "designed-but-unused indirection point" flagged in Phase 0; zero change needed to `llm/factory.py`. **Credit-assignment mechanism (concrete, since there's one shared LoRA across 3 roles, not 3 separate models)**: each rollout for one question segments into 1 (manager) + N (browser) + 1 (synthesizer) calls, captured via the `resp.messages`/`BranchTrajectory` fields Phase 3 already adds; several full rollouts of the same question form the GRPO group; the single graded outcome reward (`rubric_reward.py`) broadcasts to every call in a rollout, GRPO advantage normalizes against the group's mean/std, and policy-gradient loss is masked to only the tokens the policy actually generated in each specific call, summed across all calls in the rollout. Watch for the exact reward-hacking failure mode Polar's own ablation hit (naive broadcast + token-count imbalance across roles) at M8's spot-check step — this is real, open risk, not solved by adopting any existing framework.
- **`s2l_po.py`**: cheap add-on once the base loop runs; frozen "smaller sibling" realized via LoRA-adapter-swap (same base weights, different/no adapter) to avoid two resident models on one H100; mixing fraction annealed to 0 over first half of training.
- **Deferred stretch**: PrefixRL, full DR-Tulu-scale RL — noted, not scheduled.
- **Telemetry**: new `telemetry/rollout.py::RolloutRecord` (group id, reward, rubric-pool snapshot id, step) — same JSONL pattern as `TelemetryTracker`, kept separate since per-LLM-call telemetry is explicitly scoped to that.
- **Eval**: no new benchmark work — GRPO's own val-reward curve on a `train_pool` slice is the training signal; official benchmarks stay milestone-only (already-locked decision).

## Milestones
| # | Milestone | Compute | Notes |
|---|---|---|---|
| M0 | ~~Spike: AgentResponse trajectory access~~ | — | **Done** — confirmed `resp.messages` carries full trajectory |
| M1 | `memory/base.py`, `baseline.py`; extend `PipelineResult`/`ResearchPipeline`/`LLMCallRecord`/`run_pipeline`; Manager `DIRECT:` fast-path | CPU | Regression test: `StatelessStrategy` run == current `run_pipeline` exactly (extend `test_pipeline_smoke.py`); fast-path test: simple factoid question returns via manager call alone, no browser/synthesizer calls |
| M2 | `memory/folding.py` | CPU + API $ | A/B vs baseline, FRAMES/MuSiQue slice (~50-100 Qs) |
| M3 | `memory/retrac.py` + `compressor.yaml` + `state.py` | CPU + higher API $ (N passes/Q) | A/B + efficiency metric across passes |
| M4 (optional) | `memory/ama_graph.py` + `graph_store.py` + embedding client | CPU + embedding $ | Skip/defer if M2/M3 already show a clear winner |
| M5 | Compare via `bootstrap_ci`; pick default strategy; sanity-check DSPy-compiled prompts still work under it | CPU | Decision checkpoint before touching `rl/` |
| M6 | `rl/cold_start.py` (T3S SFT) | **1×H100, ~1-3 hrs** | First GPU milestone |
| M7 | `rl/rubric_reward.py` + `evals/judge.py` refactor | CPU/inference $ | Needs served policy model up for judge calls |
| M8 | `rl/rollout.py` + `scripts/train_grpo.py` — base GRPO, qwen3-4b LoRA, small train_pool slice | **1×H100, ~12-24 hrs, time-boxed** | Goal: reward curve rises on val slice, not SOTA numbers |
| M9 | `rl/s2l_po.py` | **1×H100, marginal** | Cheap add-on once M8 stable |
| M10 (deferred) | PrefixRL, full DR-Tulu-scale RL | Out of budget | Future work only |

## Verification
- Every milestone: `uv run ruff check . && uv run mypy . && uv run pytest tests/`.
- M1: `StatelessStrategy` produces byte-identical `PipelineResult` fields and identical telemetry call-count to today's `run_pipeline` with `memory=None` — this is the regression gate for the whole refactor.
- M2/M3/M4: A/B report = accuracy (bootstrap CI) + tokens/calls/latency per strategy, via existing `evals/runner.py` + `telemetry.summarize()`.
- M6: SFT loss curve + held-out accuracy before/after T3S masking (reproduce the paper's own "standard SFT hurts, T3S helps" comparison on your own distilled trajectories).
- M8: val-pool reward curve trends upward over training steps; spot-check rollouts for reward hacking (verbatim copying, format gaming) per DR Tulu's own negative-rubric mechanism.

## Risks
- RE-TRAC's N-pass mode multiplies eval cost linearly — cap N and slice size for the A/B (M2/M3), not full benchmarks.
- M4 (AMA-Agent) is the most likely milestone to slip/be cut — explicitly fine to skip if M2/M3 already answer the "does context strategy matter" question.
- M8's GRPO budget (~12-24hrs on 1×H100) is tight for a real reward-curve signal at 4B+LoRA with live web tool calls (slower rollouts than pure-text math RL) — treat the time-box as a hard stop with "did reward move at all" as success criterion, not a target accuracy number.
