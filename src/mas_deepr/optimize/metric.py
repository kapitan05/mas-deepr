"""DSPy metric for MIPROv2: reuses the same EM/F1 graders the eval harness
uses on verifiable-answer sources, so what DSPy optimizes for matches what
the milestone eval actually scores.
"""

from typing import Any

import dspy

from mas_deepr.evals.graders import best_f1, exact_match


def research_metric(
    example: dspy.Example, pred: dspy.Prediction, trace: Any = None
) -> float:
    """Score in [0, 1]: 1.0 on exact match, else best token-F1 against
    ``example.answer`` (and any ``answer_aliases``)."""
    gold = getattr(example, "answer", "") or ""
    if not gold:
        return 0.0
    aliases = getattr(example, "answer_aliases", None) or []
    predicted = getattr(pred, "final_answer", "") or ""

    if exact_match(predicted, gold, aliases):
        return 1.0
    return best_f1(predicted, gold, aliases)
