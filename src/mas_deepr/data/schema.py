"""Unified question record shared by every benchmark loader and train-pool builder."""

from typing import Any

from pydantic import BaseModel, Field


class RubricCriterion(BaseModel):
    criterion: str
    weight: float
    axis: str


class Question(BaseModel):
    """One evaluation or training item, normalized across sources.

    ``answer`` is set for sources with a single verifiable ground truth
    (FRAMES, BrowseComp, MuSiQue, HotpotQA). ``rubrics`` is set instead for
    ResearchRubrics' open-ended, rubric-graded prompts.
    """

    question_id: str
    source: str  # frames | browsecomp | research_rubrics | musique | hotpotqa
    split: str  # train | val | test
    prompt: str
    answer: str | None = None
    answer_aliases: list[str] = Field(default_factory=list)
    rubrics: list[RubricCriterion] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
