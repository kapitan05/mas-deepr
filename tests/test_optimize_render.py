from types import SimpleNamespace

import dspy

from mas_deepr.optimize.render import render_compiled_prompt


def _predictor(instructions: str, demos: list[dspy.Example]) -> SimpleNamespace:
    return SimpleNamespace(
        signature=SimpleNamespace(instructions=instructions), demos=demos
    )


def test_render_includes_preamble_instructions_and_no_demos() -> None:
    predictor = _predictor("Answer the sub-question grounded in context.", [])
    text = render_compiled_prompt("browser", predictor)

    assert "Use the web_search tool" in text  # preamble
    assert "Answer the sub-question grounded in context." in text
    assert "Worked examples" not in text


def test_render_manager_has_no_preamble() -> None:
    predictor = _predictor("Decompose the question.", [])
    text = render_compiled_prompt("manager", predictor)
    assert text == "Decompose the question."


def test_render_demos_use_human_readable_field_labels() -> None:
    demo = dspy.Example(
        sub_question="What is the capital of France?",
        context="[1] Paris is the capital... (http://x.com)",
        finding="Paris, per http://x.com",
    )
    predictor = _predictor("Ground your answer in the evidence.", [demo])
    text = render_compiled_prompt("browser", predictor)

    assert "Sub-question: What is the capital of France?" in text
    assert "Evidence gathered: [1] Paris is the capital" in text
    assert "Answer: Paris, per http://x.com" in text
    # raw DSPy field names should not leak into the rendered prompt
    assert "sub_question:" not in text
    assert "context:" not in text


def test_render_multiple_demos_numbered() -> None:
    demos = [
        dspy.Example(question="q1", findings="f1", final_answer="a1"),
        dspy.Example(question="q2", findings="f2", final_answer="a2"),
    ]
    predictor = _predictor("Synthesize.", demos)
    text = render_compiled_prompt("synthesizer", predictor)
    assert "### Example 1" in text
    assert "### Example 2" in text


def test_render_empty_instructions_omitted() -> None:
    predictor = _predictor("", [])
    text = render_compiled_prompt("manager", predictor)
    assert text == ""
