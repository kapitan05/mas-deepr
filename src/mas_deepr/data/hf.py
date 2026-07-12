"""Shared Hugging Face Hub download helper, cached under settings.data_dir."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from mas_deepr.config import Settings


def hf_file(
    settings: Settings, *, repo_id: str, filename: str, repo_type: str = "dataset"
) -> Path:
    cache_dir = settings.data_dir / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            cache_dir=str(cache_dir),
        )
    )
