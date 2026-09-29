"""Shared Hugging Face Hub download helper, cached under settings.data_dir."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from mas_deepr.config import Settings


def hf_file(
    settings: Settings,
    *,
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    revision: str | None = None,
) -> Path:
    """Download (and cache) one file from an HF repo.

    ``revision`` defaults to ``None`` (``hf_hub_download``'s own default,
    i.e. the repo's main branch) -- needed for datasets whose parquet only
    exists on HF's auto-conversion ref (``refs/convert/parquet``), not
    ``main`` (confirmed for ``rl-research/dr-tulu-rl-data`` -- see
    ``data/dr_tulu_rl_data.py``).
    """
    cache_dir = settings.data_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    )
