"""Render a compiled ``dspy.Predict`` (instructions + bootstrapped/labeled
demos) into plain text for the prompt registry.

DSPy's own instruction/demo mechanics aren't reused verbatim: field names
like ``sub_questions`` or ``context`` are DSPy-internal, not something a MAF
system prompt should reference directly. Each role's demos are rendered
with human-readable labels instead, and a hand-written ``preamble`` carries
the MAF-specific tool-use instructions DSPy never sees (it doesn't call
MAF's search/fetch/code-exec tools itself -- see ``optimize.modules``).
"""

import dspy

_FIELD_LABELS: dict[str, dict[str, str]] = {
    "manager": {"question": "Question", "sub_questions": "Sub-questions"},
    "browser": {
        "sub_question": "Sub-question",
        "context": "Evidence gathered",
        "finding": "Answer",
    },
    "synthesizer": {
        "question": "Question",
        "findings": "Findings",
        "final_answer": "Final answer",
    },
}

# MAF-specific tool-use instructions DSPy's optimization never sees, carried
# over from the hand-written prompts (prompts/templates/*.yaml).
PREAMBLES: dict[str, str] = {
    "manager": "",
    "browser": (
        "Use the web_search tool to find candidate sources, then use "
        "fetch_page to read the most promising results before answering. "
        "Verify claims against a fetched source rather than answering from "
        "prior knowledge alone."
    ),
    "synthesizer": (
        "Use the run_python tool if a calculation, unit conversion, date "
        "arithmetic, or count needs to be verified rather than eyeballed."
    ),
}


def render_compiled_prompt(role: str, predictor: dspy.Predict) -> str:
    labels = _FIELD_LABELS[role]
    parts = []

    preamble = PREAMBLES.get(role, "")
    if preamble:
        parts.append(preamble)

    instructions = (predictor.signature.instructions or "").strip()
    if instructions:
        parts.append(instructions)

    if predictor.demos:
        blocks = []
        for i, demo in enumerate(predictor.demos, 1):
            lines = [f"### Example {i}"]
            for key, value in dict(demo).items():
                lines.append(f"{labels.get(key, key)}: {value}")
            blocks.append("\n".join(lines))
        parts.append("Worked examples:\n\n" + "\n\n".join(blocks))

    return "\n\n".join(parts)
