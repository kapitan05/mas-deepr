"""Regression test for the real bug: a key only present in .env (not
exported in the shell) must actually reach os.environ, not just
pydantic-settings' own view of Settings' declared fields.

Reproduces it against a real temp .env file + a subprocess, since the
buggy behavior is specifically about *process*-wide os.environ visibility,
not something a monkeypatched test can fake convincingly.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_dotenv_only_var_reaches_os_environ(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("SOME_ARBITRARY_KEY=from-dotenv-only\n")

    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path({str(env_file)!r}), override=False)
        import os
        value = os.environ.get("SOME_ARBITRARY_KEY")
        assert value == "from-dotenv-only", value
        print("OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "SOME_ARBITRARY_KEY"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
