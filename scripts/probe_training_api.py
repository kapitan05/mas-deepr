"""TEMPORARY diagnostic -- safe to delete, nothing imports it.

Every ``train_grpo.py`` run currently dies at ``backend.train()`` with a
bare ``500 Internal Server Error`` from the Serverless Training "create RL
training job" endpoint. This script replays an already-saved rollout batch
from ``.art/`` straight at that endpoint (no rollouts, no judge calls) in a
few variants, to see whether ANY form of the request is accepted:

  as_sent        exactly what ART's ``backend.train()`` posts (control)
  non_null_only  same, minus the experimental_config fields that are null
  lr_only        experimental_config = {"learning_rate": ...}
  no_config      experimental_config = null
  alt_as_sent    the control request against another base model
                 (only tried when every Llama variant was rejected)

The default batch is step 0 of ``llama8b-grpo-turnpen`` -- a batch the
service accepted and trained on a brand-new Llama model on 2026-11-07, so
if it is rejected now, the content of the request is not the reason.

What it does on the account:
  - creates one throwaway model ``probe-llama-<time>`` (plus
    ``probe-alt-<time>`` if the alt-model variant runs)
  - if a variant is accepted, ONE real training step runs on that
    throwaway model
  - deletes the throwaway model(s) it created before exiting
    (``--keep-models`` to skip)

Usage:
    uv run python scripts/probe_training_api.py
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pyarrow.parquet as pq
from art.serverless.backend import ServerlessBackend
from art.utils import trajectory_logging

from mas_deepr.config import PROJECT_ROOT, get_settings

_KEEP_HEADERS = {"date", "server", "content-type", "via", "cf-ray"}


def _load_groups(path: Path) -> list[Any]:
    """Rebuild the TrajectoryGroups ART saved to ``.art/`` for one step.

    ART's public ``read_trajectory_groups_parquet()`` needs duckdb, which
    this project does not install. pyarrow (already a dependency) plus ART's
    own v2 row decoder gives the same objects.
    """
    rows = pq.read_table(
        path,
        columns=[
            "format_version",
            "group_index",
            "trajectory_index",
            "group_json",
            "trajectory_json",
        ],
    ).to_pylist()
    return trajectory_logging._read_v2(rows)


class _Captured(Exception):
    """Raised by the recorder so backend.train() stops before any HTTP call."""


def _describe_error(exc: BaseException) -> str:
    if isinstance(exc, openai.APIStatusError):
        resp = exc.response
        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() in _KEEP_HEADERS or k.lower().startswith("x-")
        }
        return f"HTTP {resp.status_code} body={resp.text[:300]!r} headers={headers}"
    return f"{type(exc).__name__}: {exc}"


async def _capture_art_request(
    backend: ServerlessBackend, groups: list[Any], learning_rate: float
) -> dict[str, Any]:
    """Return the exact kwargs ART's own backend.train() would post.

    Swaps ``training_jobs.create`` for a recorder, lets ``backend.train()``
    build its request the normal way, and stops it before anything is sent.
    """
    jobs = backend._client.training_jobs
    real_create = jobs.create
    captured: dict[str, Any] = {}

    async def _recorder(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise _Captured

    jobs.create = _recorder  # type: ignore[method-assign]
    try:
        await backend.train(
            SimpleNamespace(id="capture-only"),  # type: ignore[arg-type]
            groups,
            learning_rate=learning_rate,
        )
    except _Captured:
        pass
    finally:
        jobs.create = real_create  # type: ignore[method-assign]
    if "experimental_config" not in captured:
        raise RuntimeError("could not capture ART's training request")
    return captured


async def _preflight(base_url: str, api_key: str) -> None:
    async with httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    ) as http:
        try:
            r = await http.get("preview/base-models")
            names: Any = r.text[:300]
            try:
                data = r.json()
                items = data.get("data", data) if isinstance(data, dict) else data
                if isinstance(items, list):
                    names = [
                        (i.get("id") or i.get("name") or i.get("base_model"))
                        if isinstance(i, dict)
                        else i
                        for i in items
                    ]
            except ValueError:
                pass
            print(f"  [base-models   ] HTTP {r.status_code} {names}")
        except httpx.HTTPError as exc:
            print(f"  [base-models   ] {type(exc).__name__}: {exc}")
        try:
            r = await http.post("preview/training-jobs", json={})
            print(
                f"  [empty-body    ] HTTP {r.status_code} "
                f"(422 = the endpoint is up and validating) {r.text[:160]!r}"
            )
        except httpx.HTTPError as exc:
            print(f"  [empty-body    ] {type(exc).__name__}: {exc}")


async def _create_model(
    backend: ServerlessBackend, project: str, name: str, base_model: str
) -> Any | None:
    try:
        model = await backend._client.models.create(
            project=project, name=name, base_model=base_model, return_existing=False
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostic: report and move on
        print(f"  [model         ] could not create {name}: {_describe_error(exc)}")
        return None
    print(
        f"  [model         ] created {name} ({base_model}) "
        f"id={model.id} entity={model.entity}"
    )
    return model


async def _try_create(
    backend: ServerlessBackend,
    model_id: str,
    groups: list[Any],
    config: dict[str, Any] | None,
    label: str,
) -> str | None:
    t0 = time.monotonic()
    try:
        job = await backend._client.training_jobs.create(
            model_id=model_id,
            trajectory_groups=groups,
            experimental_config=config,  # type: ignore[arg-type]
        )
    except Exception as exc:  # noqa: BLE001 -- diagnostic: report and move on
        print(
            f"  [{label:<14}] REJECTED {time.monotonic() - t0:5.1f}s "
            f"{_describe_error(exc)}"
        )
        return None
    print(f"  [{label:<14}] ACCEPTED {time.monotonic() - t0:5.1f}s job id={job.id}")
    return str(job.id)


async def _watch_job(backend: ServerlessBackend, job_id: str, timeout_s: float) -> str:
    """Print the job's events until it ends, fails, or the timeout passes."""
    after: str | None = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        await asyncio.sleep(2)
        try:
            async for event in backend._client.training_jobs.events.list(
                training_job_id=job_id, after=after or openai.NOT_GIVEN
            ):
                after = event.id
                print(f"      event {event.type}: {json.dumps(event.data)[:400]}")
                if event.type in ("training_ended", "training_failed"):
                    return event.type
        except Exception as exc:  # noqa: BLE001
            print(f"      could not read events: {_describe_error(exc)}")
            return "events_unreadable"
    return "still_running_at_timeout"


async def main() -> None:
    settings = get_settings()
    default_batch = (
        PROJECT_ROOT
        / ".art"
        / settings.wandb_project
        / "models"
        / "llama8b-grpo-turnpen"
        / "trajectories"
        / "train"
        / "0000.parquet"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=default_batch)
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--alt-base-model",
        default="Qwen/Qwen3.8-27B",
        help="Other catalog model tried only if every Llama variant is "
        "rejected. Pass an empty string to skip.",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument(
        "--watch-seconds",
        type=float,
        default=300,
        help="How long to follow an accepted job's events.",
    )
    parser.add_argument("--keep-models", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise SystemExit("WANDB_API_KEY is not set (expected in .env).")
    if not args.batch.exists():
        raise SystemExit(f"batch not found: {args.batch}")

    # Same construction as train_grpo.py::_make_backend.
    backend = ServerlessBackend(
        base_url=os.environ.get("ART_SERVERLESS_BASE_URL") or None
    )
    groups = _load_groups(args.batch)
    request = await _capture_art_request(backend, groups, args.learning_rate)
    config: dict[str, Any] = dict(request["experimental_config"])
    groups = list(request["trajectory_groups"])
    non_null = {k: v for k, v in config.items() if v is not None}
    body_mb = len(json.dumps([g.model_dump(mode="json") for g in groups])) / 1e6

    try:
        batch_label = str(args.batch.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        batch_label = str(args.batch)
    print(f"endpoint : {backend._base_url}")
    print(
        f"batch    : {batch_label} -- {len(groups)} groups, "
        f"{sum(len(g.trajectories) for g in groups)} trajectories, ~{body_mb:.1f} MB"
    )
    print(
        f"config   : {len(config)} fields, {len(config) - len(non_null)} null: "
        f"{sorted(k for k, v in config.items() if v is None)}"
    )
    print()

    stamp = datetime.now().strftime("%H%M%S")
    created: list[Any] = []
    accepted: tuple[str, str, str] | None = None  # (variant, base model, outcome)
    try:
        await _preflight(backend._base_url, api_key)

        variants: list[tuple[str, dict[str, Any] | None]] = [
            ("as_sent", config),
            ("non_null_only", non_null),
            ("lr_only", {"learning_rate": config.get("learning_rate")}),
            ("no_config", None),
        ]
        model = await _create_model(
            backend, settings.wandb_project, f"probe-llama-{stamp}", args.base_model
        )
        if model is not None:
            created.append(model)
            for label, variant in variants:
                job_id = await _try_create(backend, model.id, groups, variant, label)
                if job_id:
                    outcome = await _watch_job(backend, job_id, args.watch_seconds)
                    accepted = (label, args.base_model, outcome)
                    break

        if accepted is None and args.alt_base_model:
            alt = await _create_model(
                backend,
                settings.wandb_project,
                f"probe-alt-{stamp}",
                args.alt_base_model,
            )
            if alt is not None:
                created.append(alt)
                # One group is enough: only whether the job is accepted matters.
                job_id = await _try_create(
                    backend, alt.id, groups[:1], config, "alt_as_sent"
                )
                if job_id:
                    outcome = await _watch_job(
                        backend, job_id, min(args.watch_seconds, 120)
                    )
                    accepted = ("alt_as_sent", args.alt_base_model, outcome)
    finally:
        for m in created:
            if args.keep_models:
                print(f"  [cleanup       ] kept {m.name} (--keep-models)")
                continue
            try:
                await backend._client.models.delete(model_id=m.id)
                print(f"  [cleanup       ] deleted {m.name}")
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  [cleanup       ] could NOT delete {m.name} -- remove it in "
                    f"the W&B UI: {_describe_error(exc)}"
                )
        await backend.close()

    print()
    if accepted is None:
        print(
            "VERDICT: every variant was rejected, including a batch that trained "
            "fine on 2026-11-07. Nothing on the client side changes the outcome."
        )
    elif accepted[0] == "as_sent":
        print(
            f"VERDICT: the normal request is accepted again (job: {accepted[2]}). "
            "Rerun train_grpo.py as is."
        )
    elif accepted[0] == "alt_as_sent":
        print(
            f"VERDICT: {args.base_model} is rejected but {accepted[1]} is accepted "
            f"(job: {accepted[2]}). The failure is specific to the Llama base model "
            "on the service side."
        )
    else:
        print(
            f"VERDICT: the request passes as '{accepted[0]}' (job: {accepted[2]}). "
            "train_grpo.py can be patched to send its config that way."
        )


if __name__ == "__main__":
    asyncio.run(main())
