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
