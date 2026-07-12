from pathlib import Path

import pytest

from mas_deepr.prompts import load_prompt, save_compiled_prompt
from mas_deepr.prompts import registry as registry_module


@pytest.mark.parametrize("role", ["manager", "browser", "synthesizer"])
def test_load_prompt_hand_written(role: str) -> None:
    template = load_prompt(role)
    assert template.role == role
    assert template.source == "hand_written"
    assert len(template.instructions.strip()) > 0


def test_load_prompt_falls_back_when_no_compiled_variant() -> None:
    hand_written = load_prompt("manager")
    fallback = load_prompt("manager", prefer_compiled=True)
    assert fallback == hand_written


def test_load_prompt_unknown_role_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent-role")


def test_save_compiled_prompt_is_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_module, "_TEMPLATES_DIR", tmp_path)
    (tmp_path / "manager.yaml").write_text(
        "role: manager\nversion: '0.1.0'\nsource: hand_written\n"
        "instructions: |\n  hand-written text\n",
        encoding="utf-8",
    )

    save_compiled_prompt("manager", "dspy-optimized text", version="0.2.0")

    default = load_prompt("manager")
    compiled = load_prompt("manager", prefer_compiled=True)
    assert default.source == "hand_written"
    assert compiled.source == "dspy_compiled"
    assert compiled.instructions == "dspy-optimized text"
