"""``hf_file``'s ``revision`` passthrough -- needed for datasets whose
parquet only exists on HF's auto-conversion ref (see
data/dr_tulu_rl_data.py), not on the repo's main branch."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mas_deepr.config import Settings
from mas_deepr.data.hf import hf_file


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


def test_hf_file_passes_revision_through_when_given(settings: Settings) -> None:
    with patch("mas_deepr.data.hf.hf_hub_download", return_value="/x") as mock_dl:
        hf_file(
            settings,
            repo_id="rl-research/dr-tulu-rl-data",
            filename="default/train/0000.parquet",
            revision="refs/convert/parquet",
        )
    assert mock_dl.call_args.kwargs["revision"] == "refs/convert/parquet"


def test_hf_file_defaults_revision_to_none(settings: Settings) -> None:
    with patch("mas_deepr.data.hf.hf_hub_download", return_value="/x") as mock_dl:
        hf_file(settings, repo_id="some/repo", filename="data.jsonl")
    assert mock_dl.call_args.kwargs["revision"] is None
