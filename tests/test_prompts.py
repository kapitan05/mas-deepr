import hashlib
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


def test_load_prompt_falls_back_when_no_compiled_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolated from the real templates dir -- a real manager.compiled.yaml
    # exists on disk (DSPy has actually been run), which would make this
    # test of the *no-compiled-variant* fallback path a false pass/fail
    # depending on what's currently compiled.
    monkeypatch.setattr(registry_module, "_TEMPLATES_DIR", tmp_path)
    (tmp_path / "manager.yaml").write_text(
        "role: manager\nversion: '0.1.0'\nsource: hand_written\n"
        "instructions: |\n  hand-written text\n",
        encoding="utf-8",
    )

    hand_written = load_prompt("manager")
    fallback = load_prompt("manager", prefer_compiled=True)
    assert fallback == hand_written


def test_load_prompt_unknown_role_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompt("nonexistent-role")


# sha256 of the exact Synthesizer instructions the model received before the
# FINAL ANSWER block moved from synthesizer.yaml into
# registry.OUTPUT_CONTRACTS (840 chars) -- pins that the move changed nothing
# for hand-written runs, so earlier baseline results stay comparable.
_SYNTHESIZER_HAND_WRITTEN_SHA256 = (
    "6154258dfb2547bfa52d4002cea772ea93fdbc6d29ce1b7d0051700b370ae93f"
)


def test_synthesizer_hand_written_prompt_unchanged() -> None:
    instructions = load_prompt("synthesizer").instructions
    digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
    assert len(instructions) == 840
    assert digest == _SYNTHESIZER_HAND_WRITTEN_SHA256


@pytest.mark.parametrize("prefer_compiled", [False, True])
def test_synthesizer_prompt_ends_with_output_contract(prefer_compiled: bool) -> None:
    instructions = load_prompt(
        "synthesizer", prefer_compiled=prefer_compiled
    ).instructions
    assert instructions.endswith(registry_module.OUTPUT_CONTRACTS["synthesizer"])
    assert instructions.count("FINAL ANSWER:") == 1


def test_output_contract_appended_to_compiled_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registry_module, "_TEMPLATES_DIR", tmp_path)
    (tmp_path / "synthesizer.yaml").write_text(
        "role: synthesizer\nversion: '0.1.0'\nsource: hand_written\n"
        "instructions: |\n  hand-written text\n",
        encoding="utf-8",
    )

    save_compiled_prompt("synthesizer", "dspy-optimized text", version="0.2.0")

    compiled = load_prompt("synthesizer", prefer_compiled=True)
    assert compiled.source == "dspy_compiled"
    assert compiled.instructions == (
        "dspy-optimized text\n\n" + registry_module.OUTPUT_CONTRACTS["synthesizer"]
    )


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
