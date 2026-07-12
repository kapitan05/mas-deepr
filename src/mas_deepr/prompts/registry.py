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


class PromptTemplate(BaseModel):
    role: str
    version: str
    source: str  # hand_written | dspy_compiled
    instructions: str


def _path_for(role: str, *, compiled: bool) -> Path:
    suffix = ".compiled.yaml" if compiled else ".yaml"
    return _TEMPLATES_DIR / f"{role}{suffix}"


def load_prompt(role: str, *, prefer_compiled: bool = False) -> PromptTemplate:
    """Load a role's prompt, optionally preferring a DSPy-compiled variant."""
    if prefer_compiled and _path_for(role, compiled=True).exists():
        path = _path_for(role, compiled=True)
    else:
        path = _path_for(role, compiled=False)
    if not path.exists():
        raise FileNotFoundError(f"No prompt template for role={role!r} at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PromptTemplate.model_validate(data)


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
