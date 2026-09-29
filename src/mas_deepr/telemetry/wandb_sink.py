"""Optional W&B mirror for eval/training summary metrics.

The JSONL telemetry ledger (``tracker.py``) stays the canonical record --
this is a thin, opt-in dashboard layer so the whole project (baseline eval,
post-DSPy eval, TRL cold-start/DPO, ART GRPO, post-GRPO eval) lands in one
W&B project with one run-naming scheme. Same privacy rule as ``tracker.py``:
aggregate metrics only, never question text or model outputs.

Gated purely on ``WANDB_API_KEY`` presence (matching ART's own convention)
plus an explicit opt-out; when disabled, every method is a no-op and
``wandb`` is never imported.
"""

import os
from typing import Any, cast


def wandb_enabled(*, disabled: bool = False) -> bool:
    return not disabled and bool(os.environ.get("WANDB_API_KEY"))


class WandbSink:
    """Wraps a single W&B run. No-op when W&B is disabled."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str,
        config: dict[str, Any] | None = None,
        disabled: bool = False,
    ) -> None:
        self._run: Any | None = None
        if not wandb_enabled(disabled=disabled):
            return
        import wandb

        self._run = wandb.init(
            project=project,
            name=run_name,
            id=run_name,
            resume="allow",
            config=config or {},
        )

    def log_summary(self, metrics: dict[str, float]) -> None:
        """Log a flat dict of aggregate numbers (accuracy, cost, latency...)."""
        if self._run is not None:
            self._run.log(metrics)

    def log_figure(self, name: str, fig: Any) -> None:
        if self._run is None:
            return
        import wandb

        self._run.log({name: wandb.Image(fig)})

    def log_table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        """Log a browsable table (per-question eval rows, per-rollout
        trajectory summaries). Content is benchmark data + model outputs --
        fine for public benchmarks, not for anything private."""
        if self._run is None:
            return
        import wandb

        self._run.log(
            {name: wandb.Table(columns=cast("list[str | int]", columns), data=rows)}
        )

    def update_config(self, config: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(config, allow_val_change=True)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def __enter__(self) -> "WandbSink":
        return self

    def __exit__(self, *exc: object) -> None:
        self.finish()


def upload_lora_artifact(
    *,
    adapter_dir: str,
    name: str,
    base_model: str,
    project: str,
    entity: str | None = None,
    storage_region: str = "coreweave-us",
) -> str:
    """Upload a PEFT LoRA adapter directory as a W&B ``"lora"``-type
    artifact, servable via W&B's Serverless LoRA Inference (see
    ``docs/wandb-inference-and-lora.md``). Returns the
    ``"wandb-artifact:///{entity}/{project}/{name}:latest"`` URI to use as
    a new ``ModelSpec.model_id`` -- or ``""`` if W&B is disabled.

    ``adapter_dir`` must be a PEFT-format save (``model.save_pretrained``
    on a ``peft.PeftModel``, which is exactly what
    ``rl/cold_start.py``/``rl/dpo.py`` write when ``config.lora`` is set).
    """
    if not wandb_enabled():
        return ""
    import wandb

    run = wandb.init(project=project, entity=entity, job_type="upload-lora")
    artifact = wandb.Artifact(
        name,
        type="lora",
        metadata={"wandb.base_model": base_model},
        storage_region=storage_region,
    )
    artifact.add_dir(adapter_dir)
    run.log_artifact(artifact)
    resolved_entity = entity or run.entity
    run.finish()
    return f"wandb-artifact:///{resolved_entity}/{project}/{name}:latest"
