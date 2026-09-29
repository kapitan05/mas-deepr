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


def test_paired_bootstrap_diff_length_mismatch_raises() -> None:
    import pytest

    from mas_deepr.evals.stats import paired_bootstrap_diff

    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap_diff([1.0, 0.0], [1.0])


def test_paired_bootstrap_diff_detects_clear_improvement() -> None:
    from mas_deepr.evals.stats import paired_bootstrap_diff

    # A correct on every question B gets wrong.
    a = [1.0] * 30
    b = [0.0] * 30
    mean_diff, lo, _hi = paired_bootstrap_diff(a, b)
    assert mean_diff == 1.0
    assert lo > 0.0  # CI excludes 0 -> A significantly beats B


def test_paired_bootstrap_diff_tie_ci_contains_zero() -> None:
    from mas_deepr.evals.stats import paired_bootstrap_diff

    a = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    b = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    _mean_diff, lo, hi = paired_bootstrap_diff(a, b, seed=1)
    assert lo <= 0.0 <= hi
