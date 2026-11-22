"""scripts/vm/patch_art_entropy.py -- on fake copies of ART's train.py, plus
one check against the real installed ART 0.5.20 file (read-only, temp copy)."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "patch_art_entropy_t", _ROOT / "scripts" / "vm" / "patch_art_entropy.py"
)
assert _spec is not None and _spec.loader is not None
pae: Any = importlib.util.module_from_spec(_spec)
sys.modules["patch_art_entropy_t"] = pae
_spec.loader.exec_module(pae)

FAKE = (
    "import torch\n\n\ndef f(chunk_logits, chunk_logsumexp, entropy, i, chunk_size):\n"
    + pae.ORIGINAL_BLOCK
    + "        entropy[:, i : i + chunk_size] = chunk_entropy\n"
)


def _fake(tmp_path: Path, text: str = FAKE) -> Path:
    path = tmp_path / "train.py"
    path.write_text(text, encoding="utf-8")
    return path


def test_apply_patches_once_keeps_backup_and_is_idempotent(tmp_path: Path) -> None:
    path = _fake(tmp_path)
    assert pae.apply(path, "0.5.20").startswith("applied")
    text = path.read_text(encoding="utf-8")
    assert text.count(pae.MARKER) == 1
    assert "with torch.no_grad():" in text and pae.ORIGINAL_BLOCK not in text
    assert (tmp_path / "train.py.mas-deepr-orig").read_text(encoding="utf-8") == FAKE
    assert pae.apply(path, "0.5.20").startswith("already applied")
    assert path.read_text(encoding="utf-8") == text


def test_patched_code_computes_the_same_entropy_without_autograd(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    path = _fake(tmp_path)
    original_ns: dict[str, Any] = {}
    exec(compile(FAKE, "orig", "exec"), original_ns)
    pae.apply(path, "0.5.20")
    patched_ns: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), "patched", "exec"), patched_ns)
    logits = torch.randn(1, 4, 7, requires_grad=True)
    lse = torch.logsumexp(logits, -1)
    e1, e2 = torch.empty(1, 4), torch.empty(1, 4)
    original_ns["f"](logits, lse, e1, 0, 4)
    patched_ns["f"](logits, lse, e2, 0, 4)
    assert torch.allclose(e1.detach(), e2)
    assert e1.requires_grad and not e2.requires_grad


@pytest.mark.parametrize(
    "text", [FAKE.replace("dim=-1", "dim=1"), FAKE + pae.ORIGINAL_BLOCK]
)
def test_refuses_unexpected_code_and_changes_nothing(tmp_path: Path, text: str) -> None:
    path = _fake(tmp_path, text)
    with pytest.raises(pae.PatchError):
        pae.apply(path, "0.5.20")
    assert path.read_text(encoding="utf-8") == text
    assert not (tmp_path / "train.py.mas-deepr-orig").exists()


def test_refuses_other_art_versions(tmp_path: Path) -> None:
    path = _fake(tmp_path)
    with pytest.raises(pae.PatchError):
        pae.apply(path, "0.5.24")
    assert path.read_text(encoding="utf-8") == FAKE


def test_revert_restores_the_original(tmp_path: Path) -> None:
    path = _fake(tmp_path)
    assert pae.revert(path).startswith("not patched")
    pae.apply(path, "0.5.20")
    assert pae.revert(path).startswith("reverted")
    assert path.read_text(encoding="utf-8") == FAKE


def test_check_reports_installed_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _fake(tmp_path)
    monkeypatch.setattr(pae, "train_py_path", lambda: path)
    monkeypatch.setattr(pae, "installed_version", lambda: "0.5.20")
    ok, message = pae.check()
    assert not ok and "NOT patched" in message
    pae.apply(path, "0.5.20")
    ok, message = pae.check()
    assert ok and "applied" in message


def test_matches_the_real_installed_art_file(tmp_path: Path) -> None:
    """The original block is in the real ART 0.5.20 train.py exactly once and
    the patched file compiles (on a temp copy -- the library isn't touched)."""
    if importlib.util.find_spec("art") is None:
        pytest.skip("ART not installed")
    if pae.installed_version() != pae.EXPECTED_VERSION:
        pytest.skip("different ART version installed")
    real = pae.train_py_path().read_text(encoding="utf-8")
    if pae.is_patched(real):
        pytest.skip("installed ART already patched")
    copy = _fake(tmp_path, real)
    assert pae.apply(copy, pae.EXPECTED_VERSION).startswith("applied")
