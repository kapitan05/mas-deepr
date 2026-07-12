"""Sandboxed Python execution tool.

Placeholder CodeAct backend: runs code in an isolated subprocess (``-I``,
no site packages, hard timeout, output cap) rather than a real micro-VM.
Swap this module for a MAF ``SupportsCodeInterpreterTool`` / firecracker
backend later -- callers only depend on ``run_python`` / ``make_code_exec_tool``.
"""

import subprocess
import sys
from typing import Annotated

from agent_framework import FunctionTool, tool

_MAX_OUTPUT_CHARS = 4000


async def run_python(code: str, *, timeout_s: float = 10.0) -> str:
    """Execute ``code`` in an isolated Python subprocess and return stdout+stderr."""
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout_s}s]"

    out = proc.stdout + (f"\nSTDERR:\n{proc.stderr}" if proc.stderr else "")
    return out[:_MAX_OUTPUT_CHARS]


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
