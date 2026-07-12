from mas_deepr.agents.parsing import parse_sub_questions


def test_parse_numbered_list() -> None:
    text = "1. What is X?\n2. What is Y?"
    assert parse_sub_questions(text, max_sub_queries=4) == ["What is X?", "What is Y?"]


def test_parse_dash_list() -> None:
    text = "- First question\n- Second question"
    assert parse_sub_questions(text, max_sub_queries=4) == [
        "First question",
        "Second question",
    ]


def test_parse_truncates_to_max_sub_queries() -> None:
    text = "1. A\n2. B\n3. C\n4. D\n5. E"
    assert parse_sub_questions(text, max_sub_queries=3) == ["A", "B", "C"]


def test_parse_falls_back_to_raw_lines_when_no_markers() -> None:
    text = "First line\nSecond line"
    assert parse_sub_questions(text, max_sub_queries=4) == [
        "First line",
        "Second line",
    ]


def test_parse_falls_back_to_whole_text_when_empty() -> None:
    assert parse_sub_questions("", max_sub_queries=4) == [""]


def test_parse_single_unmarked_line() -> None:
    assert parse_sub_questions("just one question?", max_sub_queries=4) == [
        "just one question?"
    ]
