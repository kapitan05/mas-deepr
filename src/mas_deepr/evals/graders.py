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
