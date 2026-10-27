"""PSR and DSR against the Bailey and López de Prado formulas, not against another library."""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from backtest_receipt.stats import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    estimate_sharpe,
    expected_maximum_sharpe,
    null_sharpe_dispersion,
    probabilistic_sharpe_ratio,
)


def test_psr_zero_sharpe_is_one_half() -> None:
    got = probabilistic_sharpe_ratio(0.0, n_obs=50, skewness=0.0, kurtosis=3.0, benchmark=0.0)
    assert got == pytest.approx(0.5)


def test_psr_matches_a_hand_expanded_case() -> None:
    # variance = 1 - 0 * 0.5 + ((3 - 1) / 4) * 0.5^2 = 1.125
    # z = 0.5 * sqrt(100) / sqrt(1.125) = 4.714045207910317
    # Φ(z) = 0.9999987857662636
    got = probabilistic_sharpe_ratio(0.5, n_obs=101, skewness=0.0, kurtosis=3.0, benchmark=0.0)
    assert got == pytest.approx(0.9999987857662636, abs=1e-12)


def test_psr_below_benchmark_is_under_one_half() -> None:
    got = probabilistic_sharpe_ratio(0.0, n_obs=80, skewness=0.0, kurtosis=3.0, benchmark=0.1)
    assert got < 0.5


def test_negative_skew_and_fat_tails_reduce_confidence() -> None:
    base = probabilistic_sharpe_ratio(0.2, n_obs=80, skewness=0.0, kurtosis=3.0)
    skewed = probabilistic_sharpe_ratio(0.2, n_obs=80, skewness=-1.5, kurtosis=3.0)
    heavy = probabilistic_sharpe_ratio(0.2, n_obs=80, skewness=0.0, kurtosis=8.0)
    assert skewed < base
    assert heavy < base


def test_non_positive_sampling_variance_is_undefined() -> None:
    # 1 - 10 * 1 + ((3 - 1) / 4) * 1 = -8.5
    got = probabilistic_sharpe_ratio(1.0, n_obs=40, skewness=10.0, kurtosis=3.0)
    assert math.isnan(got)
    assert math.isnan(probabilistic_sharpe_ratio(1.0, n_obs=1, skewness=0.0, kurtosis=3.0))


def test_expected_maximum_grows_with_trials_and_is_zero_for_one_trial() -> None:
    assert expected_maximum_sharpe(1, 0.2) == 0.0
    assert expected_maximum_sharpe(4, 0.0) == 0.0
    small = expected_maximum_sharpe(2, 1.0)
    large = expected_maximum_sharpe(16, 1.0)
    assert small == pytest.approx(0.5197553442805939, abs=1e-12)
    assert large > small
    with pytest.raises(ValueError):
        expected_maximum_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_maximum_sharpe(2, -0.1)


def test_deflated_ratio_uses_the_trial_benchmark() -> None:
    result = deflated_sharpe_ratio(
        0.3, n_obs=120, n_trials=10, skewness=0.0, kurtosis=3.0, trial_sharpe_std=0.16
    )
    benchmark = expected_maximum_sharpe(10, 0.16)
    direct = probabilistic_sharpe_ratio(
        0.3, n_obs=120, skewness=0.0, kurtosis=3.0, benchmark=benchmark
    )
    versus_zero = probabilistic_sharpe_ratio(
        0.3, n_obs=120, skewness=0.0, kurtosis=3.0, benchmark=0.0
    )
    assert result.sharpe_benchmark == pytest.approx(benchmark)
    assert result.deflated_sharpe == pytest.approx(direct)
    assert result.probabilistic_sharpe == pytest.approx(versus_zero)
    assert result.deflated_sharpe < result.probabilistic_sharpe


def test_single_trial_deflation_matches_psr_and_null_dispersion_is_available() -> None:
    single = deflated_sharpe_ratio(
        0.1, n_obs=30, n_trials=1, skewness=0.0, kurtosis=3.0, trial_sharpe_std=1.0
    )
    psr = probabilistic_sharpe_ratio(0.1, n_obs=30, skewness=0.0, kurtosis=3.0)
    assert single.sharpe_benchmark == 0.0
    assert single.deflated_sharpe == pytest.approx(psr)
    fallback = deflated_sharpe_ratio(0.1, n_obs=30, n_trials=5, skewness=0.0, kurtosis=3.0)
    assert fallback.trial_sharpe_std == pytest.approx(null_sharpe_dispersion(30))


def test_more_trials_lower_the_deflated_sharpe() -> None:
    few = deflated_sharpe_ratio(
        0.15, n_obs=90, n_trials=2, skewness=0.0, kurtosis=3.0, trial_sharpe_std=0.1
    )
    many = deflated_sharpe_ratio(
        0.15, n_obs=90, n_trials=40, skewness=0.0, kurtosis=3.0, trial_sharpe_std=0.1
    )
    assert many.deflated_sharpe < few.deflated_sharpe


def test_estimate_sharpe_moments_and_annualization() -> None:
    flat = estimate_sharpe([0.01, 0.01, 0.01])
    assert math.isnan(flat.sharpe_per_period)
    short = estimate_sharpe([0.01])
    assert short.n_obs == 1 and math.isnan(short.sharpe_per_period)
    series = [0.0, 0.02]
    moments = estimate_sharpe(series)
    assert moments.n_obs == 2
    assert moments.mean == pytest.approx(0.01)
    assert moments.sharpe_per_period == pytest.approx(0.01 / math.sqrt(0.0002))
    assert moments.kurtosis == pytest.approx(1.0)
    assert annualized_sharpe(moments.sharpe_per_period, 252) == pytest.approx(
        moments.sharpe_per_period * math.sqrt(252)
    )
    assert math.isnan(annualized_sharpe(float("nan"), 252))
    with pytest.raises(ValueError):
        estimate_sharpe([[0.01, 0.02]])
    with pytest.raises(ValueError):
        estimate_sharpe([0.01, float("inf")])
    with pytest.raises(ValueError):
        annualized_sharpe(1.0, 0)
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, n_obs=True, skewness=0.0, kurtosis=3.0)  # type: ignore[arg-type]


def test_estimator_does_not_reference_other_sharpe_packages() -> None:
    from pathlib import Path

    source = Path(probabilistic_sharpe_ratio.__code__.co_filename).read_text(encoding="utf-8")
    # The normal cdf below is only the hand check in this test.
    assert norm.cdf(0.0) == pytest.approx(0.5)
    for banned in ("purgedcv", "mlfinlab", "pypbo"):
        assert banned not in source
