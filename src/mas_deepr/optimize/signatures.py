"""DSPy signatures for the three pipeline roles.

Each output field is plain ``str`` (not a typed list/JSON) so parsing stays
identical to the hand-written MAF prompts and doesn't depend on a small
SLM's ability to emit valid structured output. The Manager's sub-questions
are parsed with the same ``parse_sub_questions`` the MAF pipeline uses.
"""

import dspy


class ManagerSignature(dspy.Signature):
    """Break a research question into 1-4 focused sub-questions whose answers
    give enough evidence to answer the original question."""

    question: str = dspy.InputField(desc="The research question to decompose.")
    sub_questions: str = dspy.OutputField(
        desc="A numbered list (1-4 items), one focused sub-question per line, "
        "no other commentary."
    )


class BrowserSignature(dspy.Signature):
    """Answer a sub-question directly and concisely, grounded only in the
    provided context."""

    sub_question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Search snippets and fetched page text relevant to the sub-question."
    )
    finding: str = dspy.OutputField(
        desc="A direct, evidence-grounded answer to the sub-question, citing "
        "source URLs from the context."
    )


class SynthesizerSignature(dspy.Signature):
    """Synthesize sub-question findings into a final answer to the original
    research question, grounded only in the provided findings."""

    question: str = dspy.InputField()
    findings: str = dspy.InputField(
        desc="Sub-questions and their findings, concatenated."
    )
    final_answer: str = dspy.OutputField(
        desc="The final answer to the original question, stated directly, "
        "grounded only in the findings."
    )
