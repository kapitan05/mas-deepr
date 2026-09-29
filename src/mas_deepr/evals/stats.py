"""Bootstrap confidence intervals for benchmark accuracy.

Test-set sizes here are modest (hundreds, not thousands of questions), so
point estimates alone overstate precision -- milestone reports use these CIs
instead.
"""

import random
import statistics


def bootstrap_ci(
    scores: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for a ``1 - alpha`` bootstrap CI over ``scores``."""
    if not scores:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(scores)
    means = [statistics.fmean(rng.choices(scores, k=n)) for _ in range(n_boot)]
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return statistics.fmean(scores), means[lo_idx], means[hi_idx]


def paired_bootstrap_diff(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI on the paired per-question score difference (A - B).

    ``scores_a[i]`` and ``scores_b[i]`` must be the same question, in the
    same order. The CI not containing 0 is the "A beats B" (or vice-versa)
    signal for baseline-vs-DSPy, strategy-vs-strategy, etc. Resamples
    question indices (not the two lists independently) so the pairing is
    preserved.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired diff needs equal-length score lists: "
            f"{len(scores_a)} vs {len(scores_b)}"
        )
    diffs = [a - b for a, b in zip(scores_a, scores_b, strict=True)]
    if not diffs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    means = [statistics.fmean(rng.choices(diffs, k=n)) for _ in range(n_boot)]
    means.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot) - 1
    return statistics.fmean(diffs), means[lo_idx], means[hi_idx]
