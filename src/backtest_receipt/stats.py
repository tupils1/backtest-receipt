"""Probabilistic and deflated Sharpe ratios.

The estimators follow Bailey and López de Prado, *The Sharpe Ratio Efficient
Frontier* (Journal of Risk, 2012) and *The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality* (The Journal of
Portfolio Management, 2014).

Sharpe ratios passed to these functions are per period (mean divided by the
sample standard deviation), not annualized. Annualized inputs make the
probability meaningless because the sample length is counted in periods.

Moment convention used here, stated so it can be checked against the papers:

* Sharpe ratio uses the sample standard deviation (divisor ``n - 1``).
* Skewness and kurtosis are plug-in central moments with divisor ``n``.
* Kurtosis is non-excess. A normal population has kurtosis 3, not 0.
* The deflation benchmark scales the cross-trial standard deviation of those
  per-period Sharpe ratios. When that dispersion is not supplied and there is
  more than one trial, the null sampling standard deviation ``1 / sqrt(n - 1)``
  is used. That is the paper's standard error at a true Sharpe of zero under
  normal i.i.d. returns.

This module does not call any third-party Sharpe implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# Euler–Mascheroni constant used by the expected-maximum approximation.
_EULER_MASCHERONI = 0.5772156649015328606


@dataclass(frozen=True)
class SharpeMoments:
    """Moments of one return series, in per-period units."""

    n_obs: int
    mean: float
    stdev: float
    sharpe_per_period: float
    skewness: float
    kurtosis: float


@dataclass(frozen=True)
class DeflatedSharpe:
    """PSR against zero and PSR against the expected maximum of ``n_trials``."""

    probabilistic_sharpe: float
    deflated_sharpe: float
    sharpe_benchmark: float
    trial_sharpe_std: float
    n_trials: int


def estimate_sharpe(returns: np.ndarray | list[float] | tuple[float, ...]) -> SharpeMoments:
    """Estimate per-period Sharpe, skewness, and non-excess kurtosis.

    Non-finite values are rejected. Fewer than two observations, or a zero
    sample standard deviation, yields a non-finite Sharpe rather than a
    fabricated confidence.
    """
    series = np.asarray(returns, dtype=float)
    if series.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if series.size and not np.isfinite(series).all():
        raise ValueError("returns must be finite")
    n_obs = int(series.size)
    if n_obs < 2:
        nan = float("nan")
        return SharpeMoments(n_obs, nan, nan, nan, nan, nan)
    mean = float(series.mean())
    stdev = float(series.std(ddof=1))
    if stdev <= 0.0:
        nan = float("nan")
        return SharpeMoments(n_obs, mean, 0.0, nan, nan, nan)
    centered = series - mean
    second = float(np.mean(centered**2))
    third = float(np.mean(centered**3))
    fourth = float(np.mean(centered**4))
    skewness = third / second**1.5
    kurtosis = fourth / second**2
    sharpe = mean / stdev
    return SharpeMoments(n_obs, mean, stdev, float(sharpe), float(skewness), float(kurtosis))


def annualized_sharpe(sharpe_per_period: float, periods_per_year: int) -> float:
    """Scale a per-period Sharpe by ``sqrt(periods_per_year)`` for display.

    The probabilistic and deflated ratios do not use this value.
    """
    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year < 1
    ):
        raise ValueError("periods_per_year must be an integer >= 1")
    if not math.isfinite(sharpe_per_period):
        return float("nan")
    return float(sharpe_per_period * math.sqrt(periods_per_year))


def _estimation_variance(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Numerator of the Sharpe-ratio sampling variance, before dividing by ``n - 1``."""
    return 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2


def probabilistic_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    skewness: float,
    kurtosis: float,
    benchmark: float = 0.0,
) -> float:
    """Probability that the true per-period Sharpe exceeds ``benchmark``.

    Returns NaN when the sample is shorter than two observations or when the
    estimated sampling variance is not positive. Callers must treat NaN as
    "not estimated", never as a pass.
    """
    if isinstance(n_obs, bool) or not isinstance(n_obs, int):
        raise ValueError("n_obs must be an integer")
    for name, value in (
        ("sharpe", sharpe),
        ("skewness", skewness),
        ("kurtosis", kurtosis),
        ("benchmark", benchmark),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must be a finite number")
    if n_obs < 2:
        return float("nan")
    variance = _estimation_variance(float(sharpe), float(skewness), float(kurtosis))
    if not math.isfinite(variance) or variance <= 0.0:
        return float("nan")
    standardized = (float(sharpe) - float(benchmark)) * math.sqrt(n_obs - 1) / math.sqrt(variance)
    if not math.isfinite(standardized):
        return float("nan")
    return float(norm.cdf(standardized))


def null_sharpe_dispersion(n_obs: int) -> float:
    """Sampling standard deviation of a zero true Sharpe under normal i.i.d. returns."""
    if isinstance(n_obs, bool) or not isinstance(n_obs, int) or n_obs < 2:
        raise ValueError("n_obs must be an integer >= 2")
    return 1.0 / math.sqrt(n_obs - 1)


def expected_maximum_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected maximum per-period Sharpe across independent null trials.

    For a single trial there is no selection bias and the benchmark is 0.
    ``trial_sharpe_std`` is the cross-trial standard deviation of the Sharpe
    estimates, in the same per-period units as the Sharpe itself.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")
    if isinstance(trial_sharpe_std, bool) or not isinstance(trial_sharpe_std, (int, float)):
        raise ValueError("trial_sharpe_std must be a finite number")
    dispersion = float(trial_sharpe_std)
    if not math.isfinite(dispersion) or dispersion < 0.0:
        raise ValueError("trial_sharpe_std must be a finite number >= 0")
    if n_trials == 1 or dispersion == 0.0:
        return 0.0
    near_one = 1.0 - 1.0 / n_trials
    further = 1.0 - 1.0 / (n_trials * math.e)
    if not (0.0 < near_one < 1.0 and 0.0 < further < 1.0):
        raise ValueError("n_trials is too large to evaluate the expected-maximum approximation")
    near_quantile = float(norm.ppf(near_one))
    further_quantile = float(norm.ppf(further))
    weight = _EULER_MASCHERONI
    return dispersion * ((1.0 - weight) * near_quantile + weight * further_quantile)


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    skewness: float,
    kurtosis: float,
    trial_sharpe_std: float | None = None,
) -> DeflatedSharpe:
    """Deflate ``sharpe`` for selecting it from ``n_trials`` configurations.

    ``probabilistic_sharpe`` on the result is always the probability versus a
    benchmark of zero. ``deflated_sharpe`` uses the expected-maximum benchmark.
    """
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1")
    if n_trials == 1:
        dispersion = 0.0
    elif trial_sharpe_std is None:
        dispersion = null_sharpe_dispersion(n_obs)
    else:
        dispersion = float(trial_sharpe_std)
    benchmark = expected_maximum_sharpe(n_trials, dispersion)
    versus_zero = probabilistic_sharpe_ratio(
        sharpe, n_obs=n_obs, skewness=skewness, kurtosis=kurtosis, benchmark=0.0
    )
    deflated = probabilistic_sharpe_ratio(
        sharpe, n_obs=n_obs, skewness=skewness, kurtosis=kurtosis, benchmark=benchmark
    )
    return DeflatedSharpe(
        probabilistic_sharpe=versus_zero,
        deflated_sharpe=deflated,
        sharpe_benchmark=benchmark,
        trial_sharpe_std=dispersion,
        n_trials=n_trials,
    )
