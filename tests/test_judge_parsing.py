from mas_deepr.data.schema import RubricCriterion
from mas_deepr.evals.judge import _parse_rubric_verdicts, _weighted_rubric_score


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
