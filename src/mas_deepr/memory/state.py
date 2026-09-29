"""Shared compressed-state schema for trajectory-compression strategies.

RE-TRAC's own contribution isn't the loop, it's this: compress a pass's full
trajectory into a small structured artifact the next pass treats as a
hypothesis to verify or override, instead of either re-reading the raw
trajectory or discarding it.
"""

import json
import re

from pydantic import BaseModel, Field


class StructuredState(BaseModel):
    answer: str = ""
    evidence: list[str] = Field(default_factory=list)
    analysis: str = ""
    uncertainties: list[str] = Field(default_factory=list)
    failed_attempts: list[str] = Field(default_factory=list)

    def render(self) -> str:
        """Render as the "verify or override" prefix for the next pass's
        Manager prompt -- the paper's own framing, not blind trust."""
        lines = [
            "Prior-pass hypothesis (verify or override, do not blindly trust):",
            f"- Tentative answer: {self.answer or '(none yet)'}",
        ]
        if self.evidence:
            lines.append("- Evidence so far: " + "; ".join(self.evidence))
        if self.analysis:
            lines.append(f"- Analysis: {self.analysis}")
        if self.uncertainties:
            lines.append("- Open uncertainties: " + "; ".join(self.uncertainties))
        if self.failed_attempts:
            lines.append(
                "- Approaches already tried and failed (don't repeat): "
                + "; ".join(self.failed_attempts)
            )
        return "\n".join(lines)


def parse_structured_state(raw: str) -> StructuredState:
    """Parse the compressor agent's JSON output; default-empty on failure.

    Same "extract JSON, default on parse failure" shape as
    ``evals/judge.py::_parse_rubric_verdicts`` -- a malformed compressor
    response degrades to an empty state rather than crashing the pass loop.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return StructuredState()
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return StructuredState()
    try:
        return StructuredState.model_validate(data)
    except Exception:
        return StructuredState()
