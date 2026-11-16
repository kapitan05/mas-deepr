from mas_deepr.agents.parsing import (
    has_leaked_tool_call,
    parse_sub_questions,
    strip_leaked_tool_calls,
)


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


# --- Leaked tool-call text (C1) ---------------------------------------------

_LEAK = '<tool_call>\n{"name": "fetch_page", "arguments": {"url": "https://x.org"}}\n</tool_call>'


def test_has_leaked_tool_call_detects_markup() -> None:
    assert has_leaked_tool_call(_LEAK)
    assert has_leaked_tool_call("Answer first.\n<tool_call>{")
    assert not has_leaked_tool_call("The Eiffel Tower is 330 m tall.")


def test_strip_removes_a_single_block_entirely() -> None:
    assert strip_leaked_tool_calls(_LEAK) == ""


def test_strip_keeps_surrounding_prose() -> None:
    text = f"The tower is 330 m tall.\n{_LEAK}\nSource: https://x.org"
    assert strip_leaked_tool_calls(text) == (
        "The tower is 330 m tall.\n\nSource: https://x.org"
    )


def test_strip_removes_multiple_blocks() -> None:
    text = f"Partial answer. {_LEAK} {_LEAK}"
    assert strip_leaked_tool_calls(text) == "Partial answer."


def test_strip_drops_trailing_unclosed_block() -> None:
    text = 'Partial answer.\n<tool_call>\n{"name": "wikipedia_search", "argu'
    assert strip_leaked_tool_calls(text) == "Partial answer."


def test_strip_removes_stray_closing_tag() -> None:
    assert strip_leaked_tool_calls("Answer.</tool_call>") == "Answer."


def test_strip_leaves_clean_text_unchanged() -> None:
    text = "1. Paris\n2. https://en.wikipedia.org/wiki/Paris"
    assert strip_leaked_tool_calls(text) == text
