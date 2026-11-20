from mas_deepr.data.schema import RubricCriterion
from mas_deepr.evals.judge import (
    _parse_evolving_rubrics,
    _parse_hallucination_verdict,
    _parse_rubric_verdicts,
    _weighted_rubric_score,
)


def test_parse_rubric_verdicts_basic() -> None:
    raw = '[{"index": 0, "satisfied": true}, {"index": 1, "satisfied": false}]'
    assert _parse_rubric_verdicts(raw, num_criteria=2) == [True, False]


def test_parse_rubric_verdicts_with_surrounding_prose() -> None:
    raw = 'Here is my assessment:\n[{"index": 0, "satisfied": true}]\nDone.'
    assert _parse_rubric_verdicts(raw, num_criteria=1) == [True]


def test_parse_rubric_verdicts_missing_indices_default_false() -> None:
    raw = '[{"index": 1, "satisfied": true}]'
    assert _parse_rubric_verdicts(raw, num_criteria=3) == [False, True, False]


def test_parse_rubric_verdicts_malformed_json_defaults_all_false() -> None:
    assert _parse_rubric_verdicts("not json at all", num_criteria=2) == [False, False]


def test_parse_rubric_verdicts_out_of_range_index_ignored() -> None:
    raw = '[{"index": 5, "satisfied": true}]'
    assert _parse_rubric_verdicts(raw, num_criteria=2) == [False, False]


def test_weighted_rubric_score_all_positive_unchanged_by_signed_weight_fix() -> None:
    """Regression guard: the signed-weight generalization must be
    byte-identical to the old plain-sum formula for an all-positive rubric
    set (ResearchRubrics, ResearchQA) -- abs(w) == w there."""
    rubrics = [
        RubricCriterion(criterion="a", weight=2.0, axis="x"),
        RubricCriterion(criterion="b", weight=1.0, axis="y"),
    ]
    assert _weighted_rubric_score(rubrics, [True, False]) == 2.0 / 3.0
    assert _weighted_rubric_score(rubrics, [True, True]) == 1.0
    assert _weighted_rubric_score(rubrics, [False, False]) == 0.0


def test_weighted_rubric_score_negative_weight_satisfied_lowers_score() -> None:
    """HealthBench-shaped case: a satisfied negative-weight (undesirable)
    criterion must subtract, not add."""
    rubrics = [
        RubricCriterion(criterion="good advice", weight=10.0, axis="completeness"),
        RubricCriterion(criterion="dangerous advice", weight=-10.0, axis="accuracy"),
    ]
    # Only the good criterion satisfied, dangerous one correctly avoided:
    # 10/10 (denominator is positive-weight criteria only) -> perfect score.
    assert _weighted_rubric_score(rubrics, [True, False]) == 1.0
    # Only the dangerous criterion triggered -> -10/10 clamped at 0, not -1.
    assert _weighted_rubric_score(rubrics, [False, True]) == 0.0
    # Both satisfied -> the negative cancels the positive.
    assert _weighted_rubric_score(rubrics, [True, True]) == 0.0


def test_weighted_rubric_score_empty_rubrics_does_not_divide_by_zero() -> None:
    assert _weighted_rubric_score([], []) == 0.0


def test_parse_hallucination_verdict_true() -> None:
    assert _parse_hallucination_verdict('{"hallucinated": true}') is True


def test_parse_hallucination_verdict_false() -> None:
    assert _parse_hallucination_verdict('{"hallucinated": false}') is False


def test_parse_hallucination_verdict_with_surrounding_prose() -> None:
    raw = 'My assessment:\n{"hallucinated": true}\nThat is my answer.'
    assert _parse_hallucination_verdict(raw) is True


def test_parse_hallucination_verdict_malformed_defaults_to_not_hallucinating() -> None:
    """Regression guard: a judge parsing failure must default to False
    (not hallucinating), never True -- a garbled judge response must not
    silently zero out a rollout's reward via the shaping penalty."""
    assert _parse_hallucination_verdict("not json at all") is False


def test_parse_hallucination_verdict_missing_key_defaults_false() -> None:
    assert _parse_hallucination_verdict("{}") is False


def test_parse_evolving_rubrics_signed_weights_and_axes() -> None:
    raw = (
        'Sure:\n[{"type": "positive", "criterion": "Names the 2019 cohort study"},'
        ' {"type": "negative", "criterion": "Copies evidence verbatim"}]'
    )
    rubrics = _parse_evolving_rubrics(raw, max_new=5)
    assert [(r.criterion, r.weight, r.axis) for r in rubrics] == [
        ("Names the 2019 cohort study", 1.0, "evolving_positive"),
        ("Copies evidence verbatim", -1.0, "evolving_negative"),
    ]


def test_parse_evolving_rubrics_skips_bad_entries_and_duplicates() -> None:
    raw = (
        '[{"type": "positive", "criterion": "A"}, {"type": "neutral", "criterion":'
        ' "B"}, {"type": "positive", "criterion": ""}, "junk",'
        ' {"type": "Positive", "criterion": "a"}, {"type": "negative"}]'
    )
    rubrics = _parse_evolving_rubrics(raw, max_new=5)
    assert [r.criterion for r in rubrics] == ["A"]


def test_parse_evolving_rubrics_caps_at_max_new() -> None:
    items = ", ".join(f'{{"type": "positive", "criterion": "c{i}"}}' for i in range(8))
    assert len(_parse_evolving_rubrics(f"[{items}]", max_new=5)) == 5


def test_parse_evolving_rubrics_unparseable_returns_empty() -> None:
    assert _parse_evolving_rubrics("no json here", max_new=5) == []
    assert _parse_evolving_rubrics("[not, valid json]", max_new=5) == []
