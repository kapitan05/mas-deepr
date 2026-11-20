"""Phase 4 / M8: GRPO training via OpenPipe ART, on Weights & Biases infra.

Default backend is ``art.serverless.ServerlessBackend`` -- **training and
rollout inference run on W&B's servers**, no local GPU. This is the
"no hardware yet" path: it needs only a W&B account with W&B Training
access + ``WANDB_API_KEY``, and the base ``openpipe-art`` package (the
heavy ``[backend]`` extra is only for ``LocalBackend``). Pass
``--backend local`` to run on your own GPU instead once you have one.

**Still unverified end to end** -- no W&B Training account in this dev
environment, and W&B Training may be beta/waitlisted; confirm access +
cost/quota before relying on this for a deadline. See
``docs/art-wandb-decision.md``.

Everything possible is reported to one W&B run (``project`` =
``Settings.wandb_project``, ``name`` = ``--model-name``):
  - reward / policy loss / entropy / grad norm / learning rate -- ART
    computes these server-side every ``backend.train()`` call, but (unlike
    the trajectory logging below) does NOT log them anywhere on its own --
    confirmed live 2026-09-29 (a completed run showed no "loss" group in
    W&B at all). This script now explicitly forwards
    ``backend.train()``'s own return value via
    ``model.log(metrics=result.metrics, step=result.step)`` right after
    training, which is what actually gets ``loss/entropy``, ``loss/train``,
    ``loss/grad_norm``, ``loss/learning_rate`` onto the run (NOT
    ``loss/kl_div``/``loss/kl_policy_ref`` -- those need
    ``kl_penalty_coef`` > 0, which ``ServerlessBackend.train()``'s public
    signature doesn't expose).
  - GPU / system utilization -- W&B captures these automatically
    (server-side for serverless, local-side for ``--backend local``)
  - per-rollout trajectory metrics -- via ``model.log(groups, split=...)``,
    including ``reward/base``/``reward/turn_penalty``/
    ``reward/hallucination_penalty``/``reward/shaped`` when
    ``--penalize-turns``/``--penalize-hallucination`` are on (see
    ``rl/rubric_reward.py::apply_reward_shaping``)
  - a browsable per-rollout summary table -- via ``WandbSink.log_table``

Requires ``uv sync --extra rl-art``. Importing ``art`` monkeypatches
``transformers`` process-globally (see the ``art`` pytest marker in
pyproject.toml) -- run this in its own process, never alongside
``train_cold_start.py`` / ``train_dpo.py``.

Usage:
    uv run python scripts/train_grpo.py --base-model Qwen/Qwen3-4B \\
        --musique-limit 50 --hotpot-limit 50 --group-size 4
    uv run python scripts/train_grpo.py --base-model Qwen/Qwen3-4B \\
        --backend local            # own GPU instead of W&B infra
    uv run python scripts/train_grpo.py --base-model OpenPipe/Qwen3-14B-Instruct \\
        --num-steps 20 --batch-size 60   # multi-step run, fresh batch/step
"""

import argparse
import asyncio
import itertools
import os
import random
import warnings
from typing import Any

import art
from art.serverless.backend import ServerlessBackend

from mas_deepr.config import PROJECT_ROOT, get_model, get_settings
from mas_deepr.data import build_train_pool, load_dr_tulu_paper_rubrics
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.logging_config import configure_logging
from mas_deepr.memory import MEMORY_REGISTRY
from mas_deepr.rl.local_backend import (
    DEFAULT_KEEP_CHECKPOINT_EVERY,
    LONG_CALL_WARN_FRACTION,
    add_local_gpu_args,
    add_local_train_args,
    dropped_long_calls,
    local_model_kwargs,
    local_train_kwargs,
    prune_checkpoints,
)
from mas_deepr.rl.rollout import run_rollout_group
from mas_deepr.storage import RunSync
from mas_deepr.telemetry import TelemetryTracker, WandbSink, wandb_enabled


def _make_backend(kind: str) -> Any:
    if kind == "serverless":
        # Optional override of ART's built-in https://api.training.wandb.ai/v1,
        # e.g. CoreWeave Forge's documented https://forge.coreweave.com/api/training/v1.
        # Unset -> ART's default address, exactly as before.
        return ServerlessBackend(
            base_url=os.environ.get("ART_SERVERLESS_BASE_URL") or None
        )
    if kind == "local":
        from art.local import LocalBackend

        return LocalBackend()
    if kind == "tinker":
        # Thinking Machines' Tinker: hosted inference + LoRA training.
        # Needs TINKER_API_KEY in .env and the separate .venv-tinker env
        # (openpipe-art[tinker] pins torch/transformers/numpy versions that
        # conflict with the rl-art extra).
        from art.tinker import TinkerBackend

        return TinkerBackend()
    raise ValueError(f"unknown backend {kind!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--model-name",
        default="mas-deepr-grpo",
        help="ART TrainableModel name -- also the W&B run name. Distinct "
        "names register independent LoRA adapters on the same base "
        "weights, so different reward-shaping arms (see --penalize-turns/"
        "--penalize-hallucination) can share --base-model without "
        "colliding. Run arms sequentially, not concurrently -- they share "
        "the same server-side MCP tool rate limits (mcp_backend/"
        "rate_limit.py), which are already at real capacity for a single "
        "arm (confirmed live: Semantic Scholar 429s at today's rate).",
    )
    parser.add_argument("--musique-limit", type=int, default=None)
    parser.add_argument("--hotpot-limit", type=int, default=None)
    parser.add_argument("--dr-tulu-limit", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--judge-model", default="gpt-oss-20b-wandb")
    parser.add_argument("--tool-scope", default="wiki_paper")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Gradient steps to run, each on a fresh shuffled batch drawn "
        "(with wraparound) from the pooled train questions -- not the same "
        "batch retrained repeatedly.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Questions per step (default: whole pool, i.e. --num-steps 1 "
        "behavior unchanged). Set below the pool size for multi-step runs "
        "so each step sees different questions.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Batch shuffle seed.")
    parser.add_argument(
        "--skip-batches",
        type=int,
        default=0,
        help="Skip this many batches of the shuffled question order before "
        "training -- for resuming an interrupted run (scripts/run_queue.py "
        "passes the number of steps already trained), so a resumed run "
        "continues with new questions instead of repeating the first ones.",
    )
    parser.add_argument(
        "--max-exceptions",
        type=float,
        default=20,
        help="Absolute count of failed question-groups tolerated per step "
        "before art.gather_trajectory_groups aborts (art itself checks "
        "exceptions/pbar.total against this only when 0<value<1 -- an "
        "absolute count avoids that ratio's confusing denominator, which "
        "counts trajectories, not groups). art's own default is 0 (zero "
        "tolerance) -- confirmed live to crash the whole run on the first "
        "32768-token context-overflow rollout (a real, expected ~10-25% "
        "rate on multi-hop MuSiQue/HotpotQA questions), losing every step "
        "still in flight.",
    )
    parser.add_argument(
        "--prefer-compiled", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--backend", choices=("serverless", "local", "tinker"), default="serverless"
    )
    parser.add_argument(
        "--penalize-turns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Small additive penalty for rollouts using more than the "
        "3-role-turn floor (manager+browser+synthesizer). Off by default "
        "-- see rl/rubric_reward.py::apply_reward_shaping.",
    )
    parser.add_argument("--turn-penalty-weight", type=float, default=0.03)
    parser.add_argument(
        "--penalize-hallucination",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="One extra judge call per rollout (same --judge-model, no new "
        "provider) checking whether the final answer asserts anything not "
        "supported by the Browser's findings. Off by default.",
    )
    parser.add_argument("--hallucination-penalty-weight", type=float, default=0.15)
    parser.add_argument(
        "--never-give-up",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Never Give Up (arXiv:2609.13443), synchronous and capped: when "
        "no rollout of a question succeeds, run up to --ngu-max-extra-rounds "
        "more rounds of --group-size rollouts and train on all of them as "
        "one larger group. Off by default -- see rl/rollout.py::"
        "run_rollout_group.",
    )
    parser.add_argument(
        "--ngu-max-extra-rounds",
        type=int,
        default=2,
        help="Retry cap per question for --never-give-up (default 2: at most "
        "3x --group-size rollouts per question).",
    )
    parser.add_argument(
        "--memory",
        choices=sorted(MEMORY_REGISTRY),
        default="stateless",
        help="Memory/context strategy every rollout runs under (memory/), "
        "e.g. 'folding' -- the same strategies as run_milestone_eval.py "
        "--memory. Evaluate the trained checkpoint with the same value.",
    )
    parser.add_argument(
        "--keep-checkpoint-every",
        type=int,
        default=DEFAULT_KEEP_CHECKPOINT_EVERY,
        help="local/tinker backends: keep every N-th checkpoint plus the "
        "latest, delete the rest after each step.",
    )
    add_local_gpu_args(parser)
    add_local_train_args(parser)
    parser.add_argument(
        "--evolving-rubrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="DR Tulu-style evolving rubrics (arXiv:2511.19399), scoped down: "
        "for rubric-graded questions the judge writes new criteria from each "
        "group's rollouts and the reward uses persistent + kept new rubrics. "
        "Generated rubrics are logged to runs/grpo/<model-name>-evolving-"
        "rubrics.jsonl. Off by default -- see rl/rollout.py::run_rollout_group.",
    )
    parser.add_argument(
        "--evolving-rubrics-max",
        type=int,
        default=5,
        help="Max new rubrics generated per group (DR Tulu's K_max = 5).",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="When the run finishes, upload its output folder plus the W&B "
        "run folder(s) it created to Cloudflare R2 (needs the r2 extra and "
        "the R2 vars in .env). Failures only warn.",
    )
    args = parser.parse_args()
    if args.ngu_max_extra_rounds < 0:
        raise SystemExit("--ngu-max-extra-rounds must be >= 0")
    if args.evolving_rubrics_max < 1:
        raise SystemExit("--evolving-rubrics-max must be >= 1")
    if args.keep_checkpoint_every < 1:
        raise SystemExit("--keep-checkpoint-every must be >= 1")
    if args.skip_batches < 0:
        raise SystemExit("--skip-batches must be >= 0")
    try:
        train_kwargs = local_train_kwargs(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()
    r2 = RunSync(enabled=args.r2_sync, settings=settings, root=PROJECT_ROOT)

    if not wandb_enabled():
        raise SystemExit(
            "WANDB_API_KEY is required -- the serverless backend runs training "
            "and inference on W&B, and reporting goes to W&B."
        )

    model: art.TrainableModel[Any, Any] = art.TrainableModel(
        name=args.model_name,
        # Paired with WandbSink(run_name=...) below -- same value, same
        # convention already established in this script (see
        # test_train_grpo_cli.py's regression guard). openpipe-art 0.5.20
        # made this a required kwarg (confirmed live 2026-09-29, was
        # optional on the 0.5.18 this was originally written against).
        run_name=args.model_name,
        project=settings.wandb_project,
        base_model=args.base_model,
        **local_model_kwargs(args),
    )
    verifiable_pool = build_train_pool(
        settings, musique_limit=args.musique_limit, hotpot_limit=args.hotpot_limit
    )
    rubric_pool = load_dr_tulu_paper_rubrics(settings, limit=args.dr_tulu_limit)
    # build_train_pool splits MuSiQue/HotpotQA into train/val by hashed
    # question_id -- train only here; DR-Tulu's own rl-data ships with no
    # val split of its own, every row is split="train" (see
    # data/dr_tulu_rl_data.py).
    questions = [q for q in (verifiable_pool + rubric_pool) if q.split == "train"]
    if not questions:
        raise SystemExit("no train-split questions in the pool -- check limits")
    n_musique = sum(1 for q in questions if q.source == "musique")
    n_hotpot = sum(1 for q in questions if q.source == "hotpotqa")
    n_dr_tulu = sum(1 for q in questions if q.source == "dr_tulu_rl_data")

    # Deterministic shuffle so batches differ step-to-step (not the same
    # ~55 questions retrained --num-steps times, which would just overfit
    # that one batch rather than generalize) -- wraps around via
    # itertools.cycle if --num-steps * --batch-size exceeds the pool.
    rng = random.Random(args.seed)
    rng.shuffle(questions)
    batch_size = args.batch_size if args.batch_size is not None else len(questions)

    model.update_wandb_config(
        {
            "base_model": args.base_model,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "backend": args.backend,
            "tool_scope": args.tool_scope,
            "prefer_compiled": args.prefer_compiled,
            "num_steps": args.num_steps,
            "batch_size": batch_size,
            "pool_size": len(questions),
            "max_exceptions": args.max_exceptions,
            "n_musique": n_musique,
            "n_hotpot": n_hotpot,
            "n_dr_tulu": n_dr_tulu,
            "penalize_turns": args.penalize_turns,
            "turn_penalty_weight": args.turn_penalty_weight,
            "penalize_hallucination": args.penalize_hallucination,
            "hallucination_penalty_weight": args.hallucination_penalty_weight,
            "never_give_up": args.never_give_up,
            "ngu_max_extra_rounds": args.ngu_max_extra_rounds,
            "evolving_rubrics": args.evolving_rubrics,
            "evolving_rubrics_max": args.evolving_rubrics_max,
            "keep_checkpoint_every": args.keep_checkpoint_every,
            "memory": args.memory,
            **(
                {
                    "max_seq_length": args.max_seq_length,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "load_in_4bit": args.load_in_4bit,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    "packed_sequence_length": args.packed_sequence_length,
                    "logprob_chunk_size": args.logprob_chunk_size,
                }
                if args.backend == "local"
                else {}
            ),
        }
    )
    backend = _make_backend(args.backend)
    await model.register(backend)

    tracker = TelemetryTracker(
        settings.runs_dir / "grpo" / "telemetry.jsonl", run_id="grpo", phase="post-grpo"
    )
    judge = JudgeClient(
        spec=get_model(args.judge_model), settings=settings, tracker=tracker
    )

    # One semaphore for the whole run, shared across every question's
    # group -- not one per group/question, which wouldn't bound anything
    # across questions (art.gather_trajectory_groups runs every group
    # concurrently). See Settings.grpo_llm_concurrency's docstring.
    llm_semaphore = asyncio.Semaphore(settings.grpo_llm_concurrency)
    evolving_rubrics_log = (
        settings.runs_dir / "grpo" / f"{args.model_name}-evolving-rubrics.jsonl"
        if args.evolving_rubrics
        else None
    )
    pool_cycle = itertools.cycle(questions)
    # Resume support: consume the batches an interrupted run already trained
    # on (same seed -> same shuffled order).
    for _ in itertools.islice(pool_cycle, args.skip_batches * batch_size):
        pass
    all_rows: list[list[Any]] = []

    for step_idx in range(args.num_steps):
        batch = list(itertools.islice(pool_cycle, batch_size))
        groups: list[Any] = [
            run_rollout_group(
                model,
                q,
                group_size=args.group_size,
                judge=judge,
                tracker=tracker,
                tool_scope=args.tool_scope,
                prefer_compiled=args.prefer_compiled,
                llm_semaphore=llm_semaphore,
                penalize_turns=args.penalize_turns,
                turn_penalty_weight=args.turn_penalty_weight,
                penalize_hallucination=args.penalize_hallucination,
                hallucination_penalty_weight=args.hallucination_penalty_weight,
                never_give_up=args.never_give_up,
                ngu_max_extra_rounds=args.ngu_max_extra_rounds,
                evolving_rubrics=args.evolving_rubrics,
                evolving_rubrics_max=args.evolving_rubrics_max,
                evolving_rubrics_log=evolving_rubrics_log,
                memory=args.memory,
            )
            for q in batch
        ]
        trajectory_groups = await art.gather_trajectory_groups(
            groups, max_exceptions=args.max_exceptions
        )

        # 1. Trajectory metrics -> W&B (ART writes a parquet + logs the rollups).
        await model.log(trajectory_groups, split="train")

        # 2. Accumulate the browsable per-rollout summary rows (public-benchmark
        # content), logged as one table after the whole run.
        all_rows.extend(
            [
                str(t.metadata.get("question_id", "")),
                round(float(t.reward), 4),
                str(t.metadata.get("num_sub_questions", "")),
                str(t.metadata.get("final_answer", "")),
            ]
            for tg in trajectory_groups
            for t in tg.trajectories
        )

        # 3. The gradient step. backend.train()'s docstring is explicit:
        # "This method does NOT automatically log trajectories or metrics.
        # Call model.log() explicitly ... if you want to log data." --
        # confirmed live 2026-09-29: the W&B run showed no "loss" group at
        # all (only data/time/train/tables, from the model.log() call
        # above) because this return value used to be discarded outright.
        # ServerlessTrainResult.metrics carries reward/policy_loss/entropy/
        # grad_norm/learning_rate under ART's own key names --
        # _UPSTREAM_TRAIN_METRIC_KEYS in art/serverless/backend.py maps
        # these onto "loss/*" W&B keys once actually logged.
        # ART reports model calls it leaves out of the update for being too
        # long (--packed-sequence-length) only as warnings -- capture them
        # to log the count, then print them as usual.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            train_result = await backend.train(
                model,
                trajectory_groups,
                learning_rate=args.learning_rate,
                **train_kwargs,
            )
        for w in caught:
            print(f"{w.category.__name__}: {w.message}", flush=True)
        step_metrics = dict(train_result.metrics)
        if args.backend == "local":
            n_rollouts = sum(len(tg.trajectories) for tg in trajectory_groups)
            calls, rollouts = dropped_long_calls(str(w.message) for w in caught)
            share = rollouts / max(n_rollouts, 1)
            step_metrics["train/long_calls_dropped"] = calls
            step_metrics["train/long_call_rollouts_frac"] = share
            if share > LONG_CALL_WARN_FRACTION:
                print(
                    f"WARNING: step {step_idx + 1} left {calls} long model "
                    f"call(s) from {rollouts}/{n_rollouts} rollouts ({share:.0%}) "
                    f"out of the update (> {LONG_CALL_WARN_FRACTION:.0%}); "
                    f"--packed-sequence-length={args.packed_sequence_length}",
                    flush=True,
                )
        await model.log(metrics=step_metrics, step=train_result.step)
        step_num = await model.get_step()
        print(
            f"[step {step_idx + 1}/{args.num_steps}] trained on "
            f"{len(trajectory_groups)} question groups -> checkpoint step "
            f"{step_num} ({args.backend} backend)."
        )
        if args.backend in ("local", "tinker"):
            # ART saves a checkpoint every step -- on the VM's disk (local)
            # or in Tinker's storage, whose free tier stops training at 100
            # checkpoints. Keep every --keep-checkpoint-every-th step plus
            # the latest, for evaluation and the thesis's learning curves.
            await prune_checkpoints(
                model, backend, keep_every=args.keep_checkpoint_every
            )

    with WandbSink(
        project=settings.wandb_project,
        run_name=args.model_name,
        disabled=False,
    ) as sink:
        sink.log_table(
            "rollouts",
            ["question_id", "reward", "num_sub_questions", "final_answer"],
            all_rows,
        )

    print(
        f"GRPO run done: {args.num_steps} step(s) on the {args.backend} "
        f"backend. Watch W&B project {settings.wandb_project!r}."
    )
    if args.backend == "serverless":
        # ServerlessBackend trains a LoRA adapter (art.dev.model's
        # max_lora_rank/lora_alpha config), not the full base weights, and
        # W&B already hosts the checkpoint the moment training finishes --
        # no separate upload_lora_artifact() step like train_dpo.py/
        # train_cold_start.py need. This URI is what to register as a new
        # ModelSpec.model_id for eval.
        uri = model.get_inference_name()
        print(f"Servable checkpoint (latest step): {uri}")

    r2.finish([settings.runs_dir / "grpo"])


if __name__ == "__main__":
    asyncio.run(main())
