"""Cross-run loaders: every ``telemetry.jsonl`` (or ``trajectories.jsonl``)
under a root into one polars DataFrame, with a mtime-keyed parquet cache.

The per-run JSONL write path (``tracker.py``, ``trajectory_log.py``) stays
the canonical record -- this is the read side. Rows already self-identify
(``run_id``, ``phase``, ``model_key``, ``question_id``), so the on-disk
directory layout doesn't matter; we just recurse. Mirrors ART-E's
``evaluate/load_trajectories.py`` (minus the YAML flattening) without
pulling in its ``panza`` dependency.
"""

import hashlib
import json
from pathlib import Path

import polars as pl


def _fingerprint(files: list[Path]) -> str:
    parts = sorted(f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}" for p in files)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _load_jsonl_tree(
    root: Path,
    *,
    filename: str,
    cache_name: str,
    cache_key_name: str,
    phase: str | None,
    model_key: str | None,
    use_cache: bool,
) -> pl.DataFrame:
    files = sorted(p for p in root.rglob(filename) if p.is_file())
    if not files:
        return pl.DataFrame()

    cache_path = root / cache_name
    key_path = root / cache_key_name
    fp = _fingerprint(files)

    if use_cache and cache_path.exists() and key_path.exists():
        try:
            cached_fp = json.loads(key_path.read_text())["fingerprint"]
        except (json.JSONDecodeError, KeyError):
            cached_fp = None
        if cached_fp == fp:
            df = pl.read_parquet(cache_path)
            return _filter(df, phase=phase, model_key=model_key)

    frames = [pl.read_ndjson(f) for f in files if f.stat().st_size > 0]
    df = pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()

    if use_cache and df.height > 0:
        df.write_parquet(cache_path)
        key_path.write_text(json.dumps({"fingerprint": fp}))

    return _filter(df, phase=phase, model_key=model_key)


def _filter(
    df: pl.DataFrame, *, phase: str | None, model_key: str | None
) -> pl.DataFrame:
    if df.height == 0:
        return df
    if phase is not None:
        df = df.filter(pl.col("phase") == phase)
    if model_key is not None:
        df = df.filter(pl.col("model_key") == model_key)
    return df


def load_runs(
    root: Path,
    *,
    phase: str | None = None,
    model_key: str | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Concatenate every ``telemetry.jsonl`` under ``root``.

    ``phase`` / ``model_key`` filter the concatenated frame. ``use_cache``
    reuses ``root/_telemetry_cache.parquet`` when the set of files and their
    mtimes/sizes are unchanged.
    """
    return _load_jsonl_tree(
        root,
        filename="telemetry.jsonl",
        cache_name="_telemetry_cache.parquet",
        cache_key_name="_telemetry_cache.key.json",
        phase=phase,
        model_key=model_key,
        use_cache=use_cache,
    )


def load_trajectories(
    root: Path,
    *,
    phase: str | None = None,
    model_key: str | None = None,
    use_cache: bool = True,
) -> pl.DataFrame:
    """Concatenate every ``trajectories.jsonl`` under ``root`` (see
    ``telemetry/trajectory_log.py``) -- the question/sub-question/Browser-
    finding/final-answer trajectory log, not the per-LLM-call ledger."""
    return _load_jsonl_tree(
        root,
        filename="trajectories.jsonl",
        cache_name="_trajectory_cache.parquet",
        cache_key_name="_trajectory_cache.key.json",
        phase=phase,
        model_key=model_key,
        use_cache=use_cache,
    )
