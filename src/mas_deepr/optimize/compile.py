"""Phase 2 compile job: DSPy/MIPROv2 optimizes the three role prompts against
the train-pool split (MuSiQue + HotpotQA), never the eval-only benchmarks.

Writes one ``*.compiled.yaml`` per role to the prompt registry. Milestone
eval picks these up automatically once ``prefer_compiled=True`` (the
``post-dspy`` milestone, see scripts/run_milestone_eval.py).
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import dspy

from mas_deepr.config import ModelSpec, Settings
from mas_deepr.data import Question, build_train_pool
from mas_deepr.optimize.lm import build_dspy_lm
from mas_deepr.optimize.metric import research_metric
from mas_deepr.optimize.modules import ResearchProgram, make_retriever
from mas_deepr.optimize.render import render_compiled_prompt
from mas_deepr.prompts import save_compiled_prompt
from mas_deepr.tools import WebCache

_ROLES = ("manager", "browser", "synthesizer")


def _to_example(question: Question) -> dspy.Example:
    return dspy.Example(
        question=question.prompt,
        answer=question.answer or "",
        answer_aliases=question.answer_aliases,
    ).with_inputs("question")


def compile_pipeline(
    *,
    spec: ModelSpec,
    settings: Settings,
    auto: Literal["light", "medium", "heavy"] = "light",
    musique_limit: int | None = 200,
    hotpot_limit: int | None = 200,
    val_fraction: float = 0.2,
    version: str | None = None,
) -> dict[str, Path]:
    """Run MIPROv2 against the train pool and save compiled prompts.

    Returns ``{role: path}`` for the three ``*.compiled.yaml`` files written.
    """
    dspy.configure(lm=build_dspy_lm(spec, settings))

    questions = build_train_pool(
        settings,
        val_fraction=val_fraction,
        musique_limit=musique_limit,
        hotpot_limit=hotpot_limit,
    )
    trainset = [_to_example(q) for q in questions if q.split == "train"]
    valset = [_to_example(q) for q in questions if q.split == "val"]
    if not trainset or not valset:
        raise ValueError(
            f"Train pool produced train={len(trainset)}, val={len(valset)} "
            "examples -- need both non-empty to compile."
        )

    cache = WebCache(settings.cache_db)
    retriever = make_retriever(settings=settings, cache=cache)
    program = ResearchProgram(
        retrieve=retriever, max_sub_queries=settings.max_sub_queries
    )

    optimizer = dspy.MIPROv2(metric=research_metric, auto=auto)
    compiled = optimizer.compile(program, trainset=trainset, valset=valset)

    resolved_version = version or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    written: dict[str, Path] = {}
    for role in _ROLES:
        predictor = getattr(compiled, role)
        text = render_compiled_prompt(role, predictor)
        written[role] = save_compiled_prompt(role, text, version=resolved_version)
    return written
