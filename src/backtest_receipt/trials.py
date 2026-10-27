"""Count every configuration tried under a declaration.

Duplicate parameter sets do not increase the trial count used by the deflated
Sharpe ratio. A duplicate that produced different returns is a conflict: the
first result stays in the count, and the receipt records the disagreement.
Configurations outside the declared grid are still stored, so they cannot be
kept off the books, and they fail the receipt when that gate is on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from backtest_receipt.canonical import content_hash, to_jsonable
from backtest_receipt.errors import TrialError
from backtest_receipt.prereg import PreRegistration, params_within_grid
from backtest_receipt.stats import annualized_sharpe, estimate_sharpe

_DERIVED = {
    "sharpe": "sharpe",
    "sharpe_per_period": "sharpe_per_period",
    "mean_return": "mean_return",
}


@dataclass(frozen=True)
class RecordedTrial:
    """First attempt of one parameter configuration, with moments recomputed."""

    seq: int
    params: dict[str, Any]
    params_hash: str
    metrics: dict[str, Any]
    returns: list[float] | None
    outside_grid: bool
    prereg_hash: str
    sharpe_per_period: float | None
    skewness: float | None
    kurtosis: float | None
    n_obs: int | None
    primary: float | None


@dataclass(frozen=True)
class TrialSummary:
    """Search-wide facts the receipt needs."""

    attempts: int
    unique: tuple[RecordedTrial, ...]
    outside_grid: int
    conflicting_reruns: int
    missing_returns: int
    foreign_prereg: int
    selected: RecordedTrial | None
    trial_sharpe_std: float | None
    dispersion_source: str


def build_trial_body(
    spec: PreRegistration,
    params: Mapping[str, Any],
    returns: Sequence[float] | np.ndarray | None,
    user_metrics: Mapping[str, Any] | None,
    *,
    prereg_hash: str,
) -> dict[str, Any]:
    """Normalize one attempt into a JSON-safe log body."""
    try:
        normalized = to_jsonable(dict(params))
    except (TypeError, ValueError) as exc:
        raise TrialError("trial parameters must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise TrialError("trial parameters must be a mapping")
    series = _as_returns(returns)
    supplied = {} if user_metrics is None else dict(user_metrics)
    moments = estimate_sharpe(series) if series is not None else None
    metrics = _fill_metrics(spec, supplied, moments)
    outside = not params_within_grid(normalized, spec.parameters)
    sharpe = _finite(moments.sharpe_per_period) if moments is not None else None
    return {
        "params": normalized,
        "params_hash": content_hash(normalized),
        "prereg_hash": prereg_hash,
        "metrics": metrics,
        "returns": series,
        "outside_grid": outside,
        "n_obs": moments.n_obs if moments is not None else (0 if series is not None else None),
        "sharpe_per_period": sharpe,
        "skewness": _finite(moments.skewness) if moments is not None else None,
        "kurtosis": _finite(moments.kurtosis) if moments is not None else None,
    }


def build_holdout_body(
    params: Mapping[str, Any], returns: Sequence[float] | np.ndarray | None
) -> dict[str, Any]:
    """Normalize the single out-of-sample result stored after an unlock."""
    series = _as_returns(returns)
    if not series:
        raise TrialError("holdout result requires a non-empty return series")
    try:
        normalized = to_jsonable(dict(params))
    except (TypeError, ValueError) as exc:
        raise TrialError("holdout parameters must be JSON-compatible") from exc
    if not isinstance(normalized, dict):
        raise TrialError("holdout parameters must be a mapping")
    return {
        "params": normalized,
        "params_hash": content_hash(normalized),
        "returns": series,
        "n_obs": len(series),
    }


def summarize_trials(spec: PreRegistration, records: list[dict[str, Any]]) -> TrialSummary:
    """Reduce the trial log to unique configurations and the selected one.

    The selected configuration is the earliest attempt that maximizes the
    declared primary metric. Higher is better. Non-finite values sort last.
    """
    attempts = [record for record in records if record["event"] == "trial"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in attempts:
        grouped.setdefault(record["body"]["params_hash"], []).append(record)
    conflicting = 0
    unique: list[RecordedTrial] = []
    for bucket in grouped.values():
        digests = {_returns_digest(item["body"].get("returns")) for item in bucket}
        if len(digests) > 1:
            conflicting += 1
        unique.append(_recompute(spec, bucket[0]))
    selected = min(unique, key=_selection_key) if unique else None
    finite_sharpes = [
        trial.sharpe_per_period
        for trial in unique
        if trial.sharpe_per_period is not None and math.isfinite(trial.sharpe_per_period)
    ]
    if len(finite_sharpes) >= 2:
        dispersion = float(np.std(np.asarray(finite_sharpes, dtype=float), ddof=1))
        source = "cross_trial"
    elif len(unique) <= 1:
        dispersion = 0.0
        source = "single_trial"
    else:
        dispersion = None
        source = "undefined"
    return TrialSummary(
        attempts=len(attempts),
        unique=tuple(unique),
        outside_grid=sum(1 for trial in unique if trial.outside_grid),
        conflicting_reruns=conflicting,
        missing_returns=sum(1 for trial in unique if trial.returns is None),
        foreign_prereg=sum(1 for trial in unique if trial.prereg_hash != spec.content_hash()),
        selected=selected,
        trial_sharpe_std=dispersion,
        dispersion_source=source,
    )


def _fill_metrics(
    spec: PreRegistration,
    supplied: dict[str, Any],
    moments: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for name, value in supplied.items():
        metrics[name] = _json_metric(value)
    if moments is not None:
        derived = {
            "sharpe": _finite(annualized_sharpe(moments.sharpe_per_period, spec.periods_per_year)),
            "sharpe_per_period": _finite(moments.sharpe_per_period),
            "mean_return": _finite(moments.mean),
        }
        for name in spec.metrics:
            if name in _DERIVED and name not in metrics:
                metrics[name] = derived[name]
    missing = [name for name in spec.metrics if name not in metrics]
    extra = [name for name in metrics if name not in spec.metrics]
    if extra:
        raise TrialError("undeclared metrics: " + ", ".join(sorted(extra)))
    if missing:
        raise TrialError("trial is missing declared metrics: " + ", ".join(missing))
    return {name: metrics[name] for name in spec.metrics}


def _recompute(spec: PreRegistration, record: dict[str, Any]) -> RecordedTrial:
    body = record["body"]
    returns = body.get("returns")
    series = None if returns is None else [float(item) for item in returns]
    moments = estimate_sharpe(series) if series is not None else None
    metrics = dict(body.get("metrics") or {})
    sharpe = (
        _finite(moments.sharpe_per_period)
        if moments is not None
        else _finite(body.get("sharpe_per_period"))
    )
    skewness = _finite(moments.skewness) if moments is not None else _finite(body.get("skewness"))
    kurtosis = _finite(moments.kurtosis) if moments is not None else _finite(body.get("kurtosis"))
    n_obs = moments.n_obs if moments is not None else body.get("n_obs")
    if moments is not None:
        if "sharpe" in spec.metrics:
            metrics["sharpe"] = _finite(
                annualized_sharpe(moments.sharpe_per_period, spec.periods_per_year)
            )
        if "sharpe_per_period" in spec.metrics:
            metrics["sharpe_per_period"] = sharpe
        if "mean_return" in spec.metrics:
            metrics["mean_return"] = _finite(moments.mean)
    primary = metrics.get(spec.primary_metric)
    outside = not params_within_grid(body["params"], spec.parameters)
    return RecordedTrial(
        seq=int(record["seq"]),
        params=dict(body["params"]),
        params_hash=str(body["params_hash"]),
        metrics=metrics,
        returns=series,
        outside_grid=outside,
        prereg_hash=str(body.get("prereg_hash", "")),
        sharpe_per_period=sharpe if isinstance(sharpe, float) or sharpe is None else float(sharpe),
        skewness=skewness if isinstance(skewness, float) or skewness is None else float(skewness),
        kurtosis=kurtosis if isinstance(kurtosis, float) or kurtosis is None else float(kurtosis),
        n_obs=None if n_obs is None else int(n_obs),
        primary=_finite(primary) if not isinstance(primary, bool) else None,
    )


def _selection_key(trial: RecordedTrial) -> tuple[bool, float, int]:
    value = trial.primary
    finite = value is not None and math.isfinite(value)
    return (not finite, -(value if finite and value is not None else 0.0), trial.seq)


def _as_returns(returns: Sequence[float] | np.ndarray | None) -> list[float] | None:
    if returns is None:
        return None
    series = np.asarray(
        list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float
    )
    if series.ndim != 1:
        raise TrialError("returns must be one-dimensional")
    if series.size and not np.isfinite(series).all():
        raise TrialError("returns must be finite")
    return [float(item) for item in series]


def _returns_digest(returns: Any) -> str:
    if returns is None:
        return "missing"
    return content_hash(returns)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _json_metric(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return _finite(value)
    raise TrialError(
        f"metric values must be numbers, strings, or booleans, got {type(value).__name__}"
    )
