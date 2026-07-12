from mas_deepr.evals.judge import _parse_rubric_verdicts


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
