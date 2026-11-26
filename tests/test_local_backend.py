"""rl/local_backend.py -- the shared --backend local settings.

Marked ``art``: the module imports ``art`` (see test_rollout.py's module
docstring for why those tests run in their own process).
"""

import argparse
from typing import Any

import pytest

pytestmark = pytest.mark.art


def test_local_model_configs_defaults_are_bf16_32k_rank16() -> None:
    from mas_deepr.rl.local_backend import local_model_configs

    internal, lora = local_model_configs()
    assert internal["init_args"]["load_in_4bit"] is False
    assert internal["init_args"]["max_seq_length"] == 32_768
    assert internal["engine_args"]["max_model_len"] == 32_768
    assert internal["engine_args"]["gpu_memory_utilization"] == 0.8
    assert lora == {"rank": 16, "alpha": 32}


def test_local_model_configs_turn_off_async_scheduling() -> None:
    """ART's policy-span bookkeeping crashes vLLM under async scheduling
    when a request is preempted (see rl/local_backend.py's docstring)."""
    from mas_deepr.rl.local_backend import local_model_configs

    internal, _ = local_model_configs()
    assert internal["engine_args"]["async_scheduling"] is False


def test_local_model_configs_overrides_and_validation() -> None:
    from mas_deepr.rl.local_backend import local_model_configs

    internal, lora = local_model_configs(
        max_seq_length=16_384, load_in_4bit=True, lora_rank=8, lora_alpha=16
    )
    assert internal["init_args"] == {"max_seq_length": 16_384, "load_in_4bit": True}
    assert internal["engine_args"]["max_model_len"] == 16_384
    assert lora == {"rank": 8, "alpha": 16}
    with pytest.raises(ValueError):
        local_model_configs(gpu_memory_utilization=1.5)
    with pytest.raises(ValueError):
        local_model_configs(lora_rank=0)


def test_local_model_configs_feed_arts_own_config_builder() -> None:
    """The values must survive ART's get_model_config merge (which would
    otherwise default to 4-bit and the model's 262k native context)."""
    from art.dev.get_model_config import get_model_config

    from mas_deepr.rl.local_backend import local_model_configs

    internal, lora = local_model_configs()
    try:
        cfg = get_model_config(
            base_model="Qwen/Qwen3-4B-Instruct-2507",
            output_dir="/nonexistent-art-output-dir",
            config=internal,
            lora_config=lora,
        )
    except Exception as exc:  # needs the model's HF config for target modules
        pytest.skip(f"ART config builder unavailable offline: {exc}")
    assert cfg["init_args"]["load_in_4bit"] is False
    assert cfg["init_args"]["max_seq_length"] == 32_768
    assert cfg["engine_args"]["max_model_len"] == 32_768
    assert cfg["engine_args"]["async_scheduling"] is False
    assert cfg["lora_config"]["rank"] == 16


@pytest.mark.parametrize(
    ("latest", "every", "expected"),
    [
        (100, 25, [0, 25, 50, 75, 100]),
        (37, 25, [0, 25, 37]),
        (0, 25, [0]),
        (3, 1, [0, 1, 2, 3]),
    ],
)
def test_checkpoint_steps_to_keep(latest: int, every: int, expected: list[int]) -> None:
    from mas_deepr.rl.local_backend import checkpoint_steps_to_keep

    assert checkpoint_steps_to_keep(latest, every) == expected


def test_checkpoint_steps_to_keep_rejects_zero() -> None:
    from mas_deepr.rl.local_backend import checkpoint_steps_to_keep

    with pytest.raises(ValueError):
        checkpoint_steps_to_keep(10, 0)


@pytest.mark.asyncio
async def test_prune_checkpoints_asks_backend_to_keep_the_right_steps() -> None:
    from mas_deepr.rl.local_backend import prune_checkpoints

    class _Model:
        async def get_step(self) -> int:
            return 60

    class _Backend:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, list[int]]] = []

        async def _delete_checkpoint_files(self, model: Any, keep: list[int]) -> None:
            self.calls.append((model, keep))

    model, backend = _Model(), _Backend()
    kept = await prune_checkpoints(model, backend, keep_every=25)  # type: ignore[arg-type]
    assert kept == [0, 25, 50, 60]
    assert backend.calls == [(model, [0, 25, 50, 60])]


def test_local_model_kwargs_only_for_local_backend() -> None:
    from mas_deepr.rl.local_backend import add_local_gpu_args, local_model_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="serverless")
    add_local_gpu_args(parser)
    assert local_model_kwargs(parser.parse_args([])) == {}
    assert local_model_kwargs(parser.parse_args(["--backend", "tinker"])) == {}
    kwargs = local_model_kwargs(
        parser.parse_args(["--backend", "local", "--load-in-4bit", "--lora-rank", "8"])
    )
    assert set(kwargs) == {"_internal_config", "lora_config"}
    assert kwargs["_internal_config"]["init_args"]["load_in_4bit"] is True
    assert kwargs["lora_config"]["rank"] == 8


def test_local_train_kwargs_defaults_and_validation() -> None:
    from mas_deepr.rl.local_backend import add_local_train_args, local_train_kwargs

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="serverless")
    add_local_train_args(parser)
    assert local_train_kwargs(parser.parse_args([])) == {}
    assert local_train_kwargs(parser.parse_args(["--backend", "local"])) == {
        "logprob_calculation_chunk_size": 256,
        "packed_sequence_length": 6_144,
    }
    auto = parser.parse_args(["--backend", "local", "--packed-sequence-length", "0"])
    assert local_train_kwargs(auto)["packed_sequence_length"] is None
    for bad in (
        ["--packed-sequence-length", "1000"],  # not a multiple of 256
        ["--logprob-chunk-size", "0"],
        ["--packed-sequence-length", "-1"],
        ["--packed-sequence-length", "0", "--logprob-chunk-size", "3000"],
    ):
        with pytest.raises(ValueError):
            local_train_kwargs(parser.parse_args(["--backend", "local", *bad]))


def test_dropped_long_calls_parses_art_warnings() -> None:
    from mas_deepr.rl.local_backend import dropped_long_calls

    messages = [
        "Dropping 3 tokenized results from 2 trajectories that do not fit "
        "packed_sequence_length=12288 (max seen 15011). This affects training, "
        "but your model may still learn.",
        "Dropping 1 tokenized results from 1 trajectories longer than model "
        "max_seq_length=32768 (max seen 33000).",
        "some unrelated warning",
    ]
    assert dropped_long_calls(messages) == (4, 3)
    assert dropped_long_calls([]) == (0, 0)
