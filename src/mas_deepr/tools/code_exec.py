"""Sandboxed Python execution tool.

Placeholder CodeAct backend: runs code in an isolated subprocess (``-I``, no
site packages, hard wall-clock timeout, output cap, rlimits, no cwd write
access) rather than a real micro-VM. Rlimits bound CPU/memory/fork abuse but
do **not** block filesystem reads or network access -- this is still not a
real sandbox. Swap this module for a MAF ``SupportsCodeInterpreterTool`` /
firecracker backend before running unattended (e.g. Phase 5 GRPO rollouts).
Callers only depend on ``run_python`` / ``make_code_exec_tool``.
"""

import asyncio
import subprocess
import sys
import tempfile
from typing import Annotated

from agent_framework import FunctionTool, tool

_MAX_OUTPUT_CHARS = 4000
_MAX_MEMORY_BYTES = 1024 * 1024 * 1024  # 1 GiB
_MAX_OPEN_FILES = 64

# Sets rlimits from *inside* the child interpreter, before the caller's code
# runs -- deliberately not a ``preexec_fn`` on the parent side, since forking
# a multi-threaded process (this runs under ``asyncio.to_thread``) is a
# documented deadlock hazard when ``preexec_fn`` is used. ``resource`` is
# POSIX-only, and individual limits are best-effort: e.g. macOS reports
# RLIMIT_AS as adjustable but rejects any attempt to actually set it, so each
# limit is applied independently rather than all-or-nothing.
_SANDBOX_PRELUDE = f"""
try:
    import resource as _r
except ImportError:
    _r = None
if _r is not None:
    for _limit, _value in (
        (getattr(_r, "RLIMIT_CPU", None), ({{cpu_time_s}}, {{cpu_time_s}})),
        (getattr(_r, "RLIMIT_AS", None), ({_MAX_MEMORY_BYTES}, {_MAX_MEMORY_BYTES})),
        (getattr(_r, "RLIMIT_NOFILE", None), ({_MAX_OPEN_FILES}, {_MAX_OPEN_FILES})),
        (getattr(_r, "RLIMIT_CORE", None), (0, 0)),
        (getattr(_r, "RLIMIT_NPROC", None), (0, 0)),
    ):
        if _limit is None:
            continue
        try:
            _r.setrlimit(_limit, _value)
        except (ValueError, OSError):
            pass
"""


def _run_subprocess(code: str, *, timeout_s: float) -> str:
    prelude = _SANDBOX_PRELUDE.format(cpu_time_s=int(timeout_s) + 1)
    with tempfile.TemporaryDirectory() as scratch_dir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", prelude + code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=scratch_dir,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout_s}s]"

    out = proc.stdout + (f"\nSTDERR:\n{proc.stderr}" if proc.stderr else "")
    return out[:_MAX_OUTPUT_CHARS]


async def run_python(code: str, *, timeout_s: float = 10.0) -> str:
    """Execute ``code`` in an isolated Python subprocess and return stdout+stderr."""
    return await asyncio.to_thread(_run_subprocess, code, timeout_s=timeout_s)


def make_code_exec_tool(*, timeout_s: float = 10.0) -> FunctionTool:
    """Build the MAF-callable Python execution tool."""

    @tool(
        name="run_python",
        description=(
            "Execute Python code in an isolated subprocess (no filesystem/network "
            "side effects assumed safe) and return combined stdout/stderr. Use "
            "for calculations, data transforms, or verifying arithmetic in a "
            "research answer."
        ),
    )
    async def run_python_tool(
        code: Annotated[str, "Python source code to execute."],
    ) -> str:
        return await run_python(code, timeout_s=timeout_s)

    return run_python_tool
