"""YAML-backed prompt registry, separated from execution logic.

Hand-written baselines live in ``templates/<role>.yaml``. Phase 2's DSPy
compile job writes optimized variants to ``templates/<role>.compiled.yaml``;
``load_prompt`` prefers the compiled file when ``prefer_compiled=True`` so
switching a role between hand-written and DSPy-optimized prompts is a flag,
not a code change.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Fixed output contracts, appended by ``load_prompt`` to a role's prompt
# whatever its source (hand-written or DSPy-compiled), so no prompt
# optimization can drop them. The Synthesizer's FINAL ANSWER line is what
# evals/graders.py::extract_final_answer parses for exact-match/F1 scoring.
# DSPy never sees it: MIPROv2 only optimizes optimize/signatures.py's
# SynthesizerSignature (whose structured ``final_answer`` field the compile
# metric reads directly), and optimize/render.py builds the compiled prompt
# from that alone -- so the compiled Synthesizer prompt used to ship without
# this line, and exact match silently fell back to grading the whole response.
OUTPUT_CONTRACTS: dict[str, str] = {
    "synthesizer": (
        "After your explanation, end the response with exactly one final "
        "line, with\n"
        "nothing else on that line, giving the terse answer alone (a name, "
        "number,\n"
        "date, or short phrase -- not a restated sentence):\n"
        "FINAL ANSWER: <terse answer>\n"
    ),
}


class PromptTemplate(BaseModel):
    role: str
    version: str
    source: str  # hand_written | dspy_compiled
    instructions: str


def _path_for(role: str, *, compiled: bool) -> Path:
    suffix = ".compiled.yaml" if compiled else ".yaml"
    return _TEMPLATES_DIR / f"{role}{suffix}"


def has_compiled_prompt(role: str) -> bool:
    """Whether a DSPy-compiled variant exists for ``role``."""
    return _path_for(role, compiled=True).exists()


def load_prompt(role: str, *, prefer_compiled: bool = False) -> PromptTemplate:
    """Load a role's prompt, optionally preferring a DSPy-compiled variant.

    The role's ``OUTPUT_CONTRACTS`` entry, if any, is appended after the
    instructions (separated by one blank line) for both sources."""
    if prefer_compiled and _path_for(role, compiled=True).exists():
        path = _path_for(role, compiled=True)
    else:
        path = _path_for(role, compiled=False)
    if not path.exists():
        raise FileNotFoundError(f"No prompt template for role={role!r} at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    template = PromptTemplate.model_validate(data)
    contract = OUTPUT_CONTRACTS.get(role)
    if contract is None:
        return template
    instructions = template.instructions.rstrip("\n") + "\n\n" + contract
    return template.model_copy(update={"instructions": instructions})


def save_compiled_prompt(role: str, instructions: str, *, version: str) -> Path:
    """Write a DSPy-compiled prompt back to the registry (used by Phase 2)."""
    path = _path_for(role, compiled=True)
    template = PromptTemplate(
        role=role, version=version, source="dspy_compiled", instructions=instructions
    )
    path.write_text(
        yaml.safe_dump(template.model_dump(), sort_keys=False), encoding="utf-8"
    )
    return path
