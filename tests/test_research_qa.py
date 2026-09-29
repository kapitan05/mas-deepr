"""Row->Question mapping for the ResearchQA loader -- pure function, no
network (matches test_browsecomp_decrypt.py's "test what's testable without
hitting the live source" pattern)."""

from mas_deepr.data.research_qa import _row_to_question


def _row(**overrides: object) -> dict:
    base: dict[str, object] = {
        "id": "abc123",
        "query": "How does strain-softening differ between clay types?",
        "rubric": [
            {
                "rubric_item": "Compares pore pressure development.",
                "type": ["Comparison"],
            },
            {"rubric_item": "Mentions shear strength reduction.", "type": ["Example"]},
        ],
        "general_domain": "Life & Earth Sciences",
        "subdomain": "Earth Sciences",
        "field": "Geology",
    }
    base.update(overrides)
    return base


def test_row_to_question_basic_mapping() -> None:
    q = _row_to_question(_row())
    assert q.question_id == "research_qa-abc123"
    assert q.source == "research_qa"
    assert q.split == "test"
    assert q.prompt == "How does strain-softening differ between clay types?"
    assert q.metadata["field"] == "Geology"


def test_row_to_question_rubrics_weighted_uniformly() -> None:
    q = _row_to_question(_row())
    assert q.rubrics is not None
    assert len(q.rubrics) == 2
    assert all(r.weight == 1.0 for r in q.rubrics)
    assert q.rubrics[0].criterion == "Compares pore pressure development."
    assert q.rubrics[0].axis == "Comparison"


def test_row_to_question_multi_value_type_joined_into_axis() -> None:
    row = _row(rubric=[{"rubric_item": "x", "type": ["Comparison", "Example"]}])
    q = _row_to_question(row)
    assert q.rubrics is not None
    assert q.rubrics[0].axis == "Comparison, Example"


def test_row_to_question_missing_type_gives_empty_axis() -> None:
    row = _row(rubric=[{"rubric_item": "x", "type": []}])
    q = _row_to_question(row)
    assert q.rubrics is not None
    assert q.rubrics[0].axis == ""
