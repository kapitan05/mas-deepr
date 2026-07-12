from mas_deepr.evals.stats import bootstrap_ci


def test_bootstrap_ci_empty() -> None:
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_constant_scores() -> None:
    mean, lo, hi = bootstrap_ci([1.0] * 20)
    assert mean == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_mean_matches_and_bounds_ordered() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    mean, lo, hi = bootstrap_ci(scores, n_boot=500)
    assert abs(mean - 0.625) < 1e-9
    assert lo <= mean <= hi


def test_bootstrap_ci_deterministic_with_seed() -> None:
    scores = [0.3, 0.7, 1.0, 0.0, 0.5]
    r1 = bootstrap_ci(scores, seed=42)
    r2 = bootstrap_ci(scores, seed=42)
    assert r1 == r2
