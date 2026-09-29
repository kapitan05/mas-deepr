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
import random
from typing import Any

import art
from art.serverless.backend import ServerlessBackend

from mas_deepr.config import get_model, get_settings
from mas_deepr.data import build_train_pool, load_dr_tulu_paper_rubrics
from mas_deepr.evals.judge import JudgeClient
from mas_deepr.logging_config import configure_logging
from mas_deepr.rl.rollout import run_rollout_group
from mas_deepr.telemetry import TelemetryTracker, WandbSink, wandb_enabled


def _make_backend(kind: str) -> Any:
    if kind == "serverless":
        return ServerlessBackend()
    if kind == "local":
        from art.local import LocalBackend

        return LocalBackend()
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
        "--backend", choices=("serverless", "local"), default="serverless"
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
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()
    settings.ensure_dirs()

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
    pool_cycle = itertools.cycle(questions)
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
        train_result = await backend.train(
            model, trajectory_groups, learning_rate=args.learning_rate
        )
        await model.log(metrics=train_result.metrics, step=train_result.step)
        step_num = await model.get_step()
        print(
            f"[step {step_idx + 1}/{args.num_steps}] trained on "
            f"{len(trajectory_groups)} question groups -> checkpoint step "
            f"{step_num} ({args.backend} backend)."
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


if __name__ == "__main__":
    asyncio.run(main())
