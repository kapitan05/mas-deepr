from mas_deepr.evals.graders import (
    best_f1,
    exact_match,
    extract_final_answer,
    normalize_text,
    token_f1,
)


def test_normalize_text_strips_articles_punct_case() -> None:
    assert normalize_text("The Eiffel Tower!") == "eiffel tower"


def test_exact_match_basic() -> None:
    assert exact_match("The answer is Paris.", "paris") is False  # not just substr
    assert exact_match("Paris", "Paris") is True
    assert exact_match("the paris", "Paris") is True


def test_exact_match_aliases() -> None:
    assert exact_match("NYC", "New York City", aliases=["NYC", "New York"]) is True
    assert exact_match("Boston", "New York City", aliases=["NYC"]) is False


def test_token_f1_identical() -> None:
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_partial_overlap() -> None:
    score = token_f1("the quick brown fox", "quick brown dog")
    assert 0.0 < score < 1.0


def test_token_f1_no_overlap() -> None:
    assert token_f1("apple", "orange") == 0.0


def test_best_f1_picks_best_alias() -> None:
    score = best_f1("New York", "NYC", aliases=["New York City", "Big Apple"])
    assert score > token_f1("New York", "NYC")


def test_extract_final_answer_pulls_the_marked_line() -> None:
    text = "Some reasoning about it.\nFINAL ANSWER: Jane Ballou"
    assert extract_final_answer(text) == "Jane Ballou"


def test_extract_final_answer_is_case_insensitive_and_takes_last_match() -> None:
    text = "final answer: draft one\nmore reasoning\nFinal Answer: draft two"
    assert extract_final_answer(text) == "draft two"


def test_extract_final_answer_falls_back_to_full_text_when_no_marker() -> None:
    text = "Your future wife's name would be Jane Ballou. This is because..."
    assert extract_final_answer(text) == text


def test_exact_match_via_extracted_final_answer_regression() -> None:
    """The real baseline-run failure: a response that states the correct
    answer up front but never as a bare exact-match string, now graded
    against its extracted FINAL ANSWER line instead of the full prose."""
    response = (
        "Your future wife's name would be Jane Ballou.\n\n"
        "This is because:\n- The first name of the 15th First Lady's mother "
        "was Jane.\n\nFINAL ANSWER: Jane Ballou"
    )
    assert exact_match(response, "Jane Ballou") is False  # full text still fails
    assert exact_match(extract_final_answer(response), "Jane Ballou") is True
