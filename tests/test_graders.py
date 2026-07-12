from mas_deepr.evals.graders import best_f1, exact_match, normalize_text, token_f1


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
