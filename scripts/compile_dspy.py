"""Phase 2: compile Manager/Browser/Synthesizer prompts with DSPy's MIPROv2.

Optimizes against the train-pool split (MuSiQue + HotpotQA) only -- FRAMES,
BrowseComp, and ResearchRubrics are never touched here. Writes
``*.compiled.yaml`` to the prompt registry; run the ``post-dspy`` milestone
afterward to evaluate the frozen result.

Usage:
    uv run python scripts/compile_dspy.py --model qwen3-8b --auto light

    # Then, to score it:
    uv run python scripts/run_milestone_eval.py --milestone post-dspy \\
        --models qwen3-8b
"""

import argparse

from mas_deepr.config import get_model, get_settings
from mas_deepr.optimize import compile_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Model registry key, e.g. qwen3-8b"
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 search budget",
    )
    parser.add_argument(
        "--musique-limit", type=int, default=200, help="Cap MuSiQue examples pulled"
    )
    parser.add_argument(
        "--hotpot-limit", type=int, default=200, help="Cap HotpotQA examples pulled"
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2, help="Train-pool val split fraction"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Compiled-prompt version tag (default: timestamp)",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    spec = get_model(args.model)

    written = compile_pipeline(
        spec=spec,
        settings=settings,
        auto=args.auto,
        musique_limit=args.musique_limit,
        hotpot_limit=args.hotpot_limit,
        val_fraction=args.val_fraction,
        version=args.version,
    )

    print("Compiled prompts written:")
    for role, path in written.items():
        print(f"  {role}: {path}")


if __name__ == "__main__":
    main()
