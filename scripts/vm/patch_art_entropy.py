"""Patch the installed ART 0.5.20 so its training step fits the 22 GB L4.

ART's ``art/unsloth/train.py::_calculate_logprobs`` computes, for every
token, the log-probability *and* the entropy over the full 151,936-token
vocabulary -- both with autograd on, so every chunk's vocabulary-sized fp32
intermediates stay in memory until the backward pass. The entropy is only
logged (``loss/entropy``); ART's loss and gradients use the log-probs alone
(``art/loss.py``). Measured on the L4 (2026-11-22, fp32 hidden states, as
in the real trainer): 15.9 GB at 6,144 tokens as shipped, 5.4 GB with the
entropy computed under ``torch.no_grad()``. This script makes exactly that
change: the entropy lines move under ``torch.no_grad()``; the log-probs,
the loss, the gradients and the logged entropy value stay the same.

Safety:
- only ART ``0.5.20`` (the version read and measured) is patched;
- the original lines must appear exactly once, character for character,
  else nothing is changed;
- the result must compile before it is written; the original is kept as
  ``train.py.mas-deepr-orig``; the file is replaced in one step;
- running it again changes nothing (a marker comment shows it's applied).

Usage (on the GPU VM, with .venv-gpu):
    .venv-gpu/bin/python scripts/vm/patch_art_entropy.py           # apply
    .venv-gpu/bin/python scripts/vm/patch_art_entropy.py --check   # report only
    .venv-gpu/bin/python scripts/vm/patch_art_entropy.py --revert  # restore
"""

import argparse
import importlib.metadata
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

EXPECTED_VERSION = "0.5.20"
MARKER = "# mas-deepr patch: entropy without autograd"
BACKUP_SUFFIX = ".mas-deepr-orig"

ORIGINAL_BLOCK = """\
        # Compute entropy for the chunk
        log_probs_full = chunk_logits - chunk_logsumexp.unsqueeze(-1)
        chunk_entropy = (-torch.exp(log_probs_full) * log_probs_full).sum(
            dim=-1
        )  # [B, chunk_size]
"""

PATCHED_BLOCK = f"""\
        # Compute entropy for the chunk
        {MARKER} -- ART only logs it
        # (loss/entropy); the loss and gradients use log_probs alone.
        with torch.no_grad():
            log_probs_full = chunk_logits - chunk_logsumexp.unsqueeze(-1)
            chunk_entropy = (-torch.exp(log_probs_full) * log_probs_full).sum(
                dim=-1
            )  # [B, chunk_size]
"""


class PatchError(RuntimeError):
    """The installed ART isn't the code this patch was written for."""


def is_patched(text: str) -> bool:
    return MARKER in text


def patch_text(text: str) -> str:
    """Return ``text`` with the entropy block patched; raise ``PatchError``
    unless the original block appears exactly once."""
    count = text.count(ORIGINAL_BLOCK)
    if count != 1:
        raise PatchError(
            f"expected the original entropy block exactly once, found {count} "
            "-- not the ART 0.5.20 code this patch was written for; nothing changed"
        )
    patched = text.replace(ORIGINAL_BLOCK, PATCHED_BLOCK)
    compile(patched, "train.py (patched)", "exec")  # SyntaxError -> nothing written
    return patched


def train_py_path() -> Path:
    """``art/unsloth/train.py`` of the installed ART, found without importing
    ART (that would load Unsloth and the GPU)."""
    spec = importlib.util.find_spec("art")
    if spec is None or not spec.submodule_search_locations:
        raise PatchError("ART (openpipe-art) is not installed in this environment")
    return Path(next(iter(spec.submodule_search_locations))) / "unsloth" / "train.py"


def installed_version() -> str:
    return importlib.metadata.version("openpipe-art")


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def apply(path: Path, version: str) -> str:
    """Patch ``path``; returns a one-line status. Raises ``PatchError``."""
    if version != EXPECTED_VERSION:
        raise PatchError(
            f"installed ART is {version}, this patch is for {EXPECTED_VERSION} "
            "-- nothing changed"
        )
    text = path.read_text(encoding="utf-8")
    if is_patched(text):
        return f"already applied: {path}"
    patched = patch_text(text)
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        _atomic_write(backup, text)
    _atomic_write(path, patched)
    if path.read_text(encoding="utf-8").count(MARKER) != 1:
        raise PatchError(f"verification failed after writing {path}")
    return f"applied: {path} (original kept as {backup.name})"


def revert(path: Path) -> str:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not is_patched(path.read_text(encoding="utf-8")):
        return f"not patched, nothing to revert: {path}"
    if not backup.exists():
        raise PatchError(f"no backup at {backup} -- reinstall ART to restore")
    _atomic_write(path, backup.read_text(encoding="utf-8"))
    return f"reverted: {path}"


def check() -> tuple[bool, str]:
    """``(patched?, message)`` for the installed ART -- run_queue.py's guard."""
    try:
        path = train_py_path()
        version = installed_version()
    except (PatchError, importlib.metadata.PackageNotFoundError) as exc:
        return False, str(exc)
    if is_patched(path.read_text(encoding="utf-8")):
        return True, f"ART {version} entropy patch applied ({path})"
    return False, (
        f"ART {version} is NOT patched -- the training step runs out of GPU "
        "memory without it; run: .venv-gpu/bin/python scripts/vm/patch_art_entropy.py"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Report only")
    mode.add_argument("--revert", action="store_true", help="Restore the original")
    args = parser.parse_args(argv)
    if args.check:
        ok, message = check()
        print(message)
        return 0 if ok else 1
    try:
        path = train_py_path()
        print(revert(path) if args.revert else apply(path, installed_version()))
    except PatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
