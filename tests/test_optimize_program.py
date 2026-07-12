"""Offline smoke test for ResearchProgram's DSPy wiring: manager output is
parsed into sub-questions, each is retrieved+answered by the browser
predictor, and the synthesizer combines findings -- all with a scripted
``DummyLM`` so no real model or network call happens.
"""

import dspy
from dspy.utils.dummies import DummyLM

from mas_deepr.optimize.modules import ResearchProgram


def test_research_program_orchestrates_manager_browser_synthesizer() -> None:
    lm = DummyLM(
        [
            {"sub_questions": "1. What is X?\n2. What is Y?"},
            {"finding": "Finding for X"},
            {"finding": "Finding for Y"},
            {"final_answer": "Combined answer about X and Y"},
        ]
    )

    retrieved_queries: list[str] = []

    def fake_retrieve(query: str) -> str:
        retrieved_queries.append(query)
        return f"evidence for: {query}"

    program = ResearchProgram(retrieve=fake_retrieve, max_sub_queries=4)

    with dspy.context(lm=lm):
        pred = program(question="What is the relationship between X and Y?")

    assert pred.sub_questions == ["What is X?", "What is Y?"]
    assert pred.findings == ["Finding for X", "Finding for Y"]
    assert pred.final_answer == "Combined answer about X and Y"
    assert retrieved_queries == ["What is X?", "What is Y?"]


def test_research_program_respects_max_sub_queries() -> None:
    lm = DummyLM(
        [
            {"sub_questions": "1. A\n2. B\n3. C\n4. D\n5. E"},
            {"finding": "finding a"},
            {"finding": "finding b"},
            {"final_answer": "answer"},
        ]
    )
    program = ResearchProgram(retrieve=lambda q: "ctx", max_sub_queries=2)

    with dspy.context(lm=lm):
        pred = program(question="q")

    assert pred.sub_questions == ["A", "B"]
    assert len(pred.findings) == 2
