"""Settings for ART's ``LocalBackend`` on one 24 GB GPU (L4 / RTX 4090).

On W&B's serverless backend the provider chose memory settings and LoRA
size; on our own GPU ART needs them from us, and every GRPO stage plus its
evaluation must use the *same* values so the arms stay comparable -- hence
one shared definition here, used by ``scripts/train_grpo.py`` and
``scripts/eval_grpo_checkpoint.py``.

ART's own defaults that matter here (openpipe-art 0.5.20-0.5.24,
``art/dev/get_model_config.py``):

- The trainer loads the base model in **4-bit** unless told otherwise.
  A 4B model fits a 24 GB card in bf16, so we default to bf16
  (``load_in_4bit=False``) to avoid quantization error in the trained
  policy; ``--load-in-4bit`` is the fallback if the smoke run runs out of
  memory.
- vLLM and the trainer share the GPU: the trainer is moved to CPU memory
  while vLLM generates, and vLLM sleeps while the trainer trains. vLLM's
  memory share still needs headroom for the trainer process's own CUDA
  context, so it is capped below vLLM's 0.9 default.
- Without an explicit length, both the trainer and vLLM take the model's
  native context -- 262,144 tokens for Qwen3-4B-Instruct-2507, far beyond
  24 GB. 32,768 matches the pipeline's token-budget guard
  (``Settings.context_window_tokens``).
- Tool calling is already on (vLLM ``enable_auto_tool_choice`` with the
  ``hermes`` parser, which is Qwen3's format).

Training-step memory (``backend.train()``; found by the first real run,
2026-11-20, which ran out of GPU memory in ART's log-probability pass):

- ART packs the step's model calls into sequences as long as the longest
  call (rounded up to 2,048). At 16K tokens the bf16 model plus the update
  no longer fits 22 GB, so ``packed_sequence_length`` caps the sequence at
  12,288 -- calls longer than that are left out of the update (ART warns;
  ``train_grpo.py`` logs the count). In s1's first 15 steps the longest
  call was ~15K tokens and the 99th percentile ~12.6K.
- ``logprob_calculation_chunk_size`` 1024 -> 256: ART computes log-probs and
  entropy over the 151,936-token vocabulary one chunk at a time, briefly
  holding several ``chunk x vocab`` fp32 tensors (~0.6 GB each at 1024).
  Smaller chunks only change memory and speed, not the computed values.
"""

import argparse
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import art

DEFAULT_MAX_SEQ_LENGTH = 32_768
DEFAULT_GPU_MEMORY_UTILIZATION = 0.8
DEFAULT_LORA_RANK = 16  # same as rl/lora.py (cold-start SFT / DPO)
DEFAULT_LORA_ALPHA = 32
DEFAULT_KEEP_CHECKPOINT_EVERY = 25
DEFAULT_LOGPROB_CHUNK_SIZE = 256
DEFAULT_PACKED_SEQUENCE_LENGTH = 12_288
# train_grpo.py prints a warning when a step leaves more than this share of
# its rollouts partly out of the update because of the sequence cap.
LONG_CALL_WARN_FRACTION = 0.05
_ART_AUTO_SEQUENCE_MULTIPLE = 2_048  # ART's packed length when not capped
_DROPPED_RE = re.compile(r"Dropping (\d+) tokenized results from (\d+) trajectories")


def local_model_configs(
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    load_in_4bit: bool = False,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
) -> "tuple[art.dev.InternalModelConfig, art.dev.LoRAConfig]":
    """``(_internal_config, lora_config)`` for ``art.TrainableModel``.

    ``art`` is imported here, not at module level, so the argument helpers
    below stay importable without loading ART (importing it patches
    ``transformers`` process-globally)."""
    import art

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be >= 1")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    if lora_rank < 1 or lora_alpha < 1:
        raise ValueError("lora_rank and lora_alpha must be >= 1")
    internal = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=max_seq_length, load_in_4bit=load_in_4bit
        ),
        engine_args=art.dev.EngineArgs(
            max_model_len=max_seq_length,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
    )
    lora = art.dev.LoRAConfig(rank=lora_rank, alpha=lora_alpha)
    return internal, lora


def checkpoint_steps_to_keep(latest_step: int, keep_every: int) -> list[int]:
    """Steps to keep: every ``keep_every``-th step (0 included -- the
    untrained base) plus the latest one."""
    if keep_every < 1:
        raise ValueError("keep_every must be >= 1")
    keep = set(range(0, latest_step + 1, keep_every))
    keep.add(latest_step)
    return sorted(keep)


async def prune_checkpoints(
    model: "art.TrainableModel", backend: object, *, keep_every: int
) -> list[int]:
    """Delete all checkpoints of ``model`` except every ``keep_every``-th
    step and the latest, via the backend's own deletion (works for
    ``LocalBackend`` and ``TinkerBackend``, which subclasses it).
    ``model.delete_checkpoints()`` would keep only latest + best, losing
    the intermediate checkpoints. Returns the kept steps."""
    keep = checkpoint_steps_to_keep(await model.get_step(), keep_every)
    await backend._delete_checkpoint_files(model, keep)  # type: ignore[attr-defined]
    return keep


def add_local_gpu_args(parser: argparse.ArgumentParser) -> None:
    """``--backend local`` model flags, shared by scripts/train_grpo.py and
    scripts/eval_grpo_checkpoint.py -- a checkpoint must be served with the
    same values it was trained with."""
    group = parser.add_argument_group("local GPU (--backend local)")
    group.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="vLLM's share of GPU memory while generating rollouts.",
    )
    group.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train on a 4-bit base model instead of bf16 -- fallback if the "
        "GPU runs out of memory.",
    )
    group.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    group.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)


def local_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``art.TrainableModel`` kwargs for ``args.backend == "local"``; empty
    for any other backend, so serverless and Tinker keep their own
    defaults."""
    if args.backend != "local":
        return {}
    internal, lora = local_model_configs(
        max_seq_length=args.max_seq_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    return {"_internal_config": internal, "lora_config": lora}


def add_local_train_args(parser: argparse.ArgumentParser) -> None:
    """Training-step memory flags for ``--backend local`` -- train_grpo.py
    only (evaluation and DSPy don't run training steps)."""
    group = parser.add_argument_group("local GPU training step (--backend local)")
    group.add_argument(
        "--packed-sequence-length",
        type=int,
        default=DEFAULT_PACKED_SEQUENCE_LENGTH,
        help="Longest token sequence one training step processes; model calls "
        "longer than this are left out of the update (counted in the "
        "train/long_calls_dropped metric). 0 = ART's own choice (the step's "
        "longest call, which can run out of memory on a 24 GB GPU).",
    )
    group.add_argument(
        "--logprob-chunk-size",
        type=int,
        default=DEFAULT_LOGPROB_CHUNK_SIZE,
        help="Tokens per chunk in ART's log-probability pass -- smaller uses "
        "less GPU memory, same result.",
    )


def local_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """``backend.train()`` kwargs for ``args.backend == "local"``; empty for
    any other backend. Raises ``ValueError`` on lengths ART would reject
    mid-run (the packed length must be a multiple of the chunk size)."""
    if args.backend != "local":
        return {}
    chunk = args.logprob_chunk_size
    packed = args.packed_sequence_length
    if chunk < 1:
        raise ValueError("--logprob-chunk-size must be >= 1")
    if packed < 0:
        raise ValueError("--packed-sequence-length must be >= 0")
    required_multiple = packed or _ART_AUTO_SEQUENCE_MULTIPLE
    if required_multiple % chunk:
        raise ValueError(
            f"sequence length {required_multiple} must be a multiple of "
            f"--logprob-chunk-size {chunk}"
        )
    return {
        "logprob_calculation_chunk_size": chunk,
        "packed_sequence_length": packed or None,
    }


def dropped_long_calls(messages: Iterable[str]) -> tuple[int, int]:
    """``(model calls, rollouts)`` that ART left out of a training step for
    being too long, summed over its "Dropping N tokenized results from M
    trajectories ..." warnings (one per reason: model max length, packed
    sequence length)."""
    calls = rollouts = 0
    for message in messages:
        for match in _DROPPED_RE.finditer(message):
            calls += int(match.group(1))
            rollouts += int(match.group(2))
    return calls, rollouts
