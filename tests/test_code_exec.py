import pytest

from mas_deepr.tools.code_exec import run_python


@pytest.mark.asyncio
async def test_run_python_captures_stdout() -> None:
    out = await run_python("print('hello world')")
    assert "hello world" in out


@pytest.mark.asyncio
async def test_run_python_captures_error_in_stderr() -> None:
    out = await run_python("raise ValueError('bad')")
    assert "STDERR" in out
    assert "ValueError" in out


@pytest.mark.asyncio
async def test_run_python_times_out() -> None:
    out = await run_python("import time; time.sleep(5)", timeout_s=0.2)
    assert "timeout" in out
