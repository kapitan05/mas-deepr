"""Row->Question mapping and paper-scope filtering for the DR-Tulu rl-data
loader -- pure functions, no network (matches test_research_qa.py's
pattern)."""

import json

from mas_deepr.data.dr_tulu_rl_data import _is_paper_scope, _row_to_question


def _row(**overrides: object) -> dict:
    base: dict[str, object] = {
        "source": (
            "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics_adaptive_rubric"
        ),
        "question_type": "long_form",
        "messages": [{"role": "user", "content": "What causes X?"}],
        "ground_truth": json.dumps(
            {
                "query": "What causes X?",
                "rubrics": [
                    {
                        "description": "Cites a primary source.",
                        "title": "Citation",
                        "weight": 3,
                    },
                    {
                        "description": "Explains the mechanism.",
                        "title": "Mechanism",
                        "weight": 2,
                    },
                ],
            }
        ),
        "dataset": "general_rubric",
    }
    base.update(overrides)
    return base


def test_is_paper_scope_keeps_scholarqa_and_openscholar() -> None:
    assert _is_paper_scope(
        "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics_adaptive_rubric"
    )
    assert _is_paper_scope(
        "rl_rag_train_os_0915_2k_search_rubric_longform_rubrics_adaptive_rubric"
    )


def test_is_paper_scope_drops_searcharena() -> None:
    assert not _is_paper_scope("rl_rag_train_sa_3k_longform_rubrics_adaptive_rubric")


def test_row_to_question_basic_mapping() -> None:
    q = _row_to_question(_row())
    assert q.source == "dr_tulu_rl_data"
    assert q.split == "train"
    assert q.prompt == "What causes X?"
    assert q.question_id.startswith("dr_tulu_rl-")
    assert q.metadata["dr_tulu_source"].startswith("rl_rag_train_sqa")


def test_row_to_question_rubric_mapping_preserves_weight_and_axis() -> None:
    q = _row_to_question(_row())
    assert q.rubrics is not None
    assert len(q.rubrics) == 2
    assert q.rubrics[0].criterion == "Cites a primary source."
    assert q.rubrics[0].weight == 3
    assert q.rubrics[0].axis == "Citation"


def test_row_to_question_id_is_deterministic() -> None:
    q1 = _row_to_question(_row())
    q2 = _row_to_question(_row())
    assert q1.question_id == q2.question_id


def test_row_to_question_id_differs_for_different_queries() -> None:
    q1 = _row_to_question(_row())
    q2 = _row_to_question(
        _row(ground_truth=json.dumps({"query": "Something else?", "rubrics": []}))
    )
    assert q1.question_id != q2.question_id


def test_row_to_question_missing_title_gives_empty_axis() -> None:
    row = _row(
        ground_truth=json.dumps(
            {"query": "q", "rubrics": [{"description": "d", "weight": 1}]}
        )
    )
    q = _row_to_question(row)
    assert q.rubrics is not None
    assert q.rubrics[0].axis == ""
