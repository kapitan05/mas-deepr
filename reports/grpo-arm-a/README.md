# GRPO arm A — full results report

Permanent, version-controlled record of arm A (`mas-deepr-grpo`,
Qwen3-14B, OpenPipe ART `ServerlessBackend`). Training ran to checkpoint
~347 before W&B credits expired (~2026-10-01); the process died and
subsequent eval calls started failing with connection errors. Nothing
further can be run on this account — see `docs/plan/team-task-split-2026-10.md`
(local-only, not in git) for what moves to Oleksii next.

Unlike `docs/` and `runs/` (both gitignored, regenerate-per-machine by
design), this `reports/` directory is committed — it's the durable,
shareable copy of the key artifacts, including data that otherwise
existed *only* in W&B.

## Headline numbers (clean checkpoints only)

Two checkpoint evals (`step304`, `step344`) are contaminated by
infra-connection errors (100% `n_errors` on research_qa/health_bench) —
excluded below.

| Benchmark | Base model | GRPO step65 | GRPO step324 (best) | Δ base→best |
|---|---|---|---|---|
| ResearchQA | 24.3% | 47.2% | **68.6%** | **+44.3pp** |
| HealthBench | 20.4% | 34.0% | **47.2%** | **+26.8pp** |
| FRAMES | 14.0% | 14.0% | 12.0% | ~flat (see below) |

## Contents

- `charts/accuracy_comparison.png`, `charts/cost_latency.png` — curated
  base-vs-GRPO-vs-folding comparison (`scripts/plot_grpo_results.py`).
- `charts/{frames,research_qa,health_bench}_progress.png` — full
  29-checkpoint accuracy-vs-training-step curve
  (`scripts/watch_grpo_progress.py`).
- `charts/loss_train.png`, `charts/loss_grad_norm.png`,
  `charts/train_reward.png` — **training-time curves pulled from W&B**
  (`scripts/export_grpo_training_metrics.py`) — this data existed
  *nowhere* on disk before, only inside the W&B run (ART's
  `backend.train()` return value is logged straight to W&B via
  `model.log()`, never written to a local file otherwise). The reward
  curve is real, noisy, and trends clearly upward (~0.30 → ~0.55-0.65).
- `data/eval_checkpoints_summary.csv` — all 85 (checkpoint × benchmark)
  rows behind the progress charts.
- `data/training_metrics.csv` — all 500 logged training-step rows
  (`loss/*`, `train/*`) behind the loss/reward charts.
- `data/comparison_table.md` — the curated chart's underlying table.

## FRAMES: root cause found, and fixed (but not via more training)

GRPO training alone never moved FRAMES off its noisy ~12-20% band.
Root-caused by reading real failed trajectories (not guessed): FRAMES'
sub-questions are sequential/referential ("**this** mayor", "**that**
artist", "**the** vehicle") but each Browser call in the pipeline is
fully isolated — zero visibility into sibling sub-questions' findings.
The Browser literally can't resolve the reference and either guesses
wrong or says "not specified in the query."

Concrete examples (step314/step334 trajectories):
- frames-5: finding[0] correctly IDs the mayor; finding[1] says *"the
  mayor's name is not specified in the query."*
- frames-0: two (later three, at step334) independent sub-questions
  asking "who is the 15th First Lady" return **mutually contradictory**
  answers (Lady Bird Johnson / Eleanor Roosevelt / Jill Biden), since no
  Browser call sees another's answer.

**Fix, validated with a controlled A/B on the identical checkpoint
(step324)**: switching to the `folding` memory strategy (siblings see
each other's findings, `memory/folding.py`, already built/tested, just
never applied to GRPO training or eval before this):

| Memory strategy | FRAMES accuracy | Errors |
|---|---|---|
| `stateless` (default, what arm A trained under) | 12% | 1 |
| `folding` | **20%** | **0** |

More GRPO steps alone cannot fix a structural context-sharing gap in
the pipeline — the right lever (sharing context between sub-questions)
mattered more than more training. Caveat on the accuracy_comparison
chart: the "+ folding" bars are empty (not zero) for ResearchQA/
HealthBench — that checkpoint was only ever evaluated on FRAMES with
folding.

## Separate bug found: tool-call text leak

At step334, frames-1's finding[1] contains a raw, unexecuted
`<tool_call>{"name": "wikipedia_search", "arguments": {...}}</tool_call>`
string directly in the Browser's finding text — the search never ran.
Tool-call emission/parsing failure (`agents/topology.py`'s
`_run_agent`/MAF's tool-calling handling), unrelated to folding.

## A note on the 14.0% vs. 15.6% FRAMES-baseline discrepancy

Not a data issue: `accuracy_comparison_chart`/`comparison_table`
(reused from `evals/charts.py`) exclude infra-error rows via `_ok()`
before averaging, giving the base model 15.6% on FRAMES; the 14.0%
used in the table above counts infra errors as wrong answers
(denominator = all 50 questions) — the stricter convention, kept
consistent everywhere else in this report.

## Known gaps, not completed (credits ran out)

- The planned 3-point folding trend (step206/284/324) only has one
  point (324) — step206/284 folding evals were never run.
- No real frontier-model (GPT-4.1/Gemini/DeepSeek) comparison exists at
  full sample size — the only numbers on disk are n=2 smoke tests.
- `rl/s2l_po.py` (sibling-model rollout diversity) was built and unit
  tested but never wired into `rl/rollout.py` — zero live verification.
- `kl_div`/`kl_policy_ref` never appeared in W&B — confirmed
  structurally unavailable: `ServerlessBackend.train()`'s public
  signature doesn't expose `kl_penalty_coef`, not a bug in our code.

All of the above, plus a full new-experiment backlog, is handed off to
Oleksii — see `docs/plan/team-task-split-2026-10.md` (local-only).
