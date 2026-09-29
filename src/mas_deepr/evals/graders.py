"""Verifiable-answer graders (FRAMES, MuSiQue, HotpotQA): normalized EM + token F1.

Standard SQuAD-style normalization: lowercase, strip punctuation/articles,
collapse whitespace. Judge-based grading (BrowseComp, ResearchRubrics) lives
in ``evals.judge`` since it requires an LLM call.
"""

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_FINAL_ANSWER_RE = re.compile(r"final answer:\s*(.+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    """Pull the terse ``FINAL ANSWER: ...`` line out of a prose response.

    The Synthesizer is prompted (see prompts/templates/synthesizer.yaml) to
    end every response with exactly one such line -- exact_match/token_f1
    compare against a terse gold string, and grading the full multi-sentence
    explanation against it was a real bug (a response that stated the
    correct answer up front, e.g. "Your future wife's name would be Jane
    Ballou.", scored 0 against gold "Jane Ballou" every time). Last match,
    not first, matching the same reasoning as judge.py's verdict parsing:
    the explanation could itself use the phrase "final answer" in passing.

    Falls back to the raw text unchanged if no such line is present (a
    model that ignores the instruction, or older logged responses recorded
    before this instruction existed) -- never raises, never returns empty
    on a non-empty input.
    """
    matches = _FINAL_ANSWER_RE.findall(text)
    if not matches:
        return text
    return matches[-1].strip()


def exact_match(prediction: str, gold: str, aliases: list[str] | None = None) -> bool:
    candidates = [gold, *(aliases or [])]
    norm_pred = normalize_text(prediction)
    return any(normalize_text(c) == norm_pred for c in candidates)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold: str, aliases: list[str] | None = None) -> float:
    candidates = [gold, *(aliases or [])]
    return max(token_f1(prediction, c) for c in candidates)
