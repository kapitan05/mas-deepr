"""WandbSink: no-op without a key, logs the right shape with a mocked wandb.

Never touches the real wandb service.
"""

import sys
import types
from typing import Any

import pytest

from mas_deepr.telemetry import WandbSink, upload_lora_artifact, wandb_enabled


class _FakeRun:
    def __init__(self, entity: str = "me") -> None:
        self.logged: list[dict[str, Any]] = []
        self.finished = False
        self.entity = entity
        self.logged_artifacts: list[Any] = []

    def log(self, d: dict[str, Any]) -> None:
        self.logged.append(d)

    def log_artifact(self, artifact: Any) -> None:
        self.logged_artifacts.append(artifact)

    def finish(self) -> None:
        self.finished = True


class _FakeArtifact:
    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        self.added_dir: str | None = None

    def add_dir(self, path: str) -> None:
        self.added_dir = path


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeRun:
    run = _FakeRun()
    mod = types.ModuleType("wandb")
    mod.init = lambda **kwargs: run  # type: ignore[attr-defined]
    mod.Image = lambda fig: ("image", fig)  # type: ignore[attr-defined]
    mod.Table = lambda columns, data: ("table", columns, data)  # type: ignore[attr-defined]
    mod.Artifact = _FakeArtifact  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", mod)
    monkeypatch.setenv("WANDB_API_KEY", "wb-test")
    return run


def test_disabled_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert wandb_enabled() is False
    sink = WandbSink(project="p", run_name="r")  # must not import wandb
    sink.log_summary({"accuracy/frames": 42.0})  # no-op
    sink.log_figure("chart", object())
    sink.finish()


def test_explicit_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "wb-test")
    assert wandb_enabled(disabled=True) is False


def test_logs_summary_metrics(fake_wandb: _FakeRun) -> None:
    with WandbSink(project="mas-deepr", run_name="eval-baseline-qwen3-8b") as sink:
        sink.log_summary({"accuracy/frames": 55.0, "cost_usd_mean/frames": 0.01})
    assert fake_wandb.logged == [
        {"accuracy/frames": 55.0, "cost_usd_mean/frames": 0.01}
    ]
    assert fake_wandb.finished is True


def test_logs_figure(fake_wandb: _FakeRun) -> None:
    sink = WandbSink(project="p", run_name="r")
    marker = object()
    sink.log_figure("accuracy_chart", marker)
    assert fake_wandb.logged == [{"accuracy_chart": ("image", marker)}]


def test_logs_table(fake_wandb: _FakeRun) -> None:
    sink = WandbSink(project="p", run_name="r")
    sink.log_table("results", ["qid", "score"], [["q1", 1.0], ["q2", 0.0]])
    assert fake_wandb.logged == [
        {"results": ("table", ["qid", "score"], [["q1", 1.0], ["q2", 0.0]])}
    ]


def test_table_is_noop_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    WandbSink(project="p", run_name="r").log_table("t", ["a"], [["x"]])  # no-op


def test_upload_lora_artifact_is_noop_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    uri = upload_lora_artifact(
        adapter_dir="/tmp/adapter", name="my_lora", base_model="x", project="p"
    )
    assert uri == ""


def test_upload_lora_artifact_returns_wandb_artifact_uri(
    fake_wandb: _FakeRun,
) -> None:
    uri = upload_lora_artifact(
        adapter_dir="/tmp/adapter",
        name="my_lora",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        project="mas-deepr",
        entity="myteam",
    )
    assert uri == "wandb-artifact:///myteam/mas-deepr/my_lora:latest"
    assert len(fake_wandb.logged_artifacts) == 1
    artifact = fake_wandb.logged_artifacts[0]
    assert artifact.name == "my_lora"
    assert artifact.kwargs["type"] == "lora"
    assert artifact.kwargs["metadata"] == {
        "wandb.base_model": "meta-llama/Llama-3.1-8B-Instruct"
    }
    assert artifact.added_dir == "/tmp/adapter"
    assert fake_wandb.finished is True


def test_upload_lora_artifact_falls_back_to_run_entity(fake_wandb: _FakeRun) -> None:
    uri = upload_lora_artifact(
        adapter_dir="/tmp/adapter", name="my_lora", base_model="x", project="mas-deepr"
    )
    assert uri == "wandb-artifact:///me/mas-deepr/my_lora:latest"
