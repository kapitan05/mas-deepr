"""Shared parsing for the Manager role's sub-question output.

Both the MAF pipeline (agents/topology.py) and the DSPy-compiled pipeline
(optimize/modules.py) produce the same plain-text numbered-list format for
sub-questions, so they share this parser rather than each reimplementing it.
"""

import re

_NUMBERED_LINE = re.compile(r"^\s*(?:\d+[.)]|-)\s*(.+)$")


def parse_sub_questions(text: str, *, max_sub_queries: int) -> list[str]:
    lines = [_NUMBERED_LINE.match(line) for line in text.strip().splitlines()]
    parsed = [m.group(1).strip() for m in lines if m]
    if not parsed:
        parsed = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return parsed[:max_sub_queries] or [text.strip()]
