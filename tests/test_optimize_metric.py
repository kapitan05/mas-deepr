import dspy

from mas_deepr.optimize.metric import research_metric


def test_metric_exact_match_scores_one() -> None:
    example = dspy.Example(answer="Paris").with_inputs()
    pred = dspy.Prediction(final_answer="Paris")
    assert research_metric(example, pred) == 1.0


def test_metric_normalized_exact_match_scores_one() -> None:
    example = dspy.Example(answer="Paris").with_inputs()
    pred = dspy.Prediction(final_answer="the Paris.")
    assert research_metric(example, pred) == 1.0


def test_metric_alias_match_scores_one() -> None:
    example = dspy.Example(answer="New York City", answer_aliases=["NYC"]).with_inputs()
    pred = dspy.Prediction(final_answer="NYC")
    assert research_metric(example, pred) == 1.0


def test_metric_partial_overlap_gives_partial_credit() -> None:
    example = dspy.Example(answer="quick brown fox").with_inputs()
    pred = dspy.Prediction(final_answer="the quick brown dog")
    score = research_metric(example, pred)
    assert 0.0 < score < 1.0


def test_metric_no_overlap_scores_zero() -> None:
    example = dspy.Example(answer="Paris").with_inputs()
    pred = dspy.Prediction(final_answer="Tokyo")
    assert research_metric(example, pred) == 0.0


def test_metric_missing_gold_scores_zero() -> None:
    example = dspy.Example(answer="").with_inputs()
    pred = dspy.Prediction(final_answer="anything")
    assert research_metric(example, pred) == 0.0
