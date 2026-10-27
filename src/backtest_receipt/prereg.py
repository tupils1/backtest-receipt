"""Pre-registration of a study.

The prose fields (``hypothesis``, ``notes``, ``universe`` description) are part
of the hashed document so they cannot be rewritten quietly. They are not inputs
to the Sharpe math. Thresholds, the search space, and the holdout boundary are.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from backtest_receipt.canonical import content_hash, to_jsonable
from backtest_receipt.errors import PreRegistrationError

_TOP_LEVEL = {
    "study_id",
    "hypothesis",
    "universe",
    "parameters",
    "metrics",
    "primary_metric",
    "pass_criteria",
    "holdout",
    "periods_per_year",
    "notes",
}
_PASS_KEYS = {
    "min_deflated_sharpe",
    "min_probabilistic_sharpe",
    "min_oos_sharpe_annualized",
    "max_sharpe_decay_annualized",
    "min_oos_observations",
    "require_single_holdout_unlock",
    "reject_post_result_amendments",
    "reject_trials_outside_grid",
    "reject_conflicting_reruns",
    "require_return_series",
}
_HOLDOUT_KEYS = {"time_column", "start", "embargo_periods"}

# Conventional PSR/DSR confidence used when the declaration omits the hurdle.
# Explicit null in the document turns the hurdle off.
_DEFAULT_MIN_DEFLATED_SHARPE = 0.95
_DEFAULT_MIN_OOS_OBSERVATIONS = 30


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise PreRegistrationError(f"unknown {where} field(s): {names}")


def _study_id_ok(value: str) -> bool:
    if not value or len(value) > 64:
        return False
    first, rest = value[0], value[1:]
    if not first.isalnum():
        return False
    return all(character.isalnum() or character in "_.-" for character in rest)


def _as_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreRegistrationError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise PreRegistrationError(f"{field} must be finite")
    return number


def _optional_probability(raw: Mapping[str, Any], key: str, default: float | None) -> float | None:
    if key not in raw:
        return default
    value = raw[key]
    if value is None:
        return None
    number = _as_number(value, key)
    if not 0.0 <= number <= 1.0:
        raise PreRegistrationError(f"{key} must be between 0 and 1")
    return number


def _optional_float(raw: Mapping[str, Any], key: str, default: float | None) -> float | None:
    if key not in raw:
        return default
    value = raw[key]
    if value is None:
        return None
    return _as_number(value, key)


def _optional_int(raw: Mapping[str, Any], key: str, default: int | None) -> int | None:
    if key not in raw:
        return default
    value = raw[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise PreRegistrationError(f"{key} must be an integer")
    if value < 1:
        raise PreRegistrationError(f"{key} must be >= 1")
    return value


def _optional_bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in raw or raw[key] is None:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        raise PreRegistrationError(f"{key} must be true or false")
    return value


@dataclass(frozen=True)
class PassCriteria:
    """Thresholds fixed before results are known.

    A missing key in the source document receives the library default. An
    explicit null disables that numeric hurdle. The resolved values are what
    get hashed.
    """

    min_deflated_sharpe: float | None = _DEFAULT_MIN_DEFLATED_SHARPE
    min_probabilistic_sharpe: float | None = None
    min_oos_sharpe_annualized: float | None = 0.0
    max_sharpe_decay_annualized: float | None = None
    min_oos_observations: int | None = _DEFAULT_MIN_OOS_OBSERVATIONS
    require_single_holdout_unlock: bool = True
    reject_post_result_amendments: bool = True
    reject_trials_outside_grid: bool = True
    reject_conflicting_reruns: bool = True
    require_return_series: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_deflated_sharpe": self.min_deflated_sharpe,
            "min_probabilistic_sharpe": self.min_probabilistic_sharpe,
            "min_oos_sharpe_annualized": self.min_oos_sharpe_annualized,
            "max_sharpe_decay_annualized": self.max_sharpe_decay_annualized,
            "min_oos_observations": self.min_oos_observations,
            "require_single_holdout_unlock": self.require_single_holdout_unlock,
            "reject_post_result_amendments": self.reject_post_result_amendments,
            "reject_trials_outside_grid": self.reject_trials_outside_grid,
            "reject_conflicting_reruns": self.reject_conflicting_reruns,
            "require_return_series": self.require_return_series,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> PassCriteria:
        data = {} if raw is None else dict(raw)
        if not isinstance(raw, Mapping) and raw is not None:
            raise PreRegistrationError("pass_criteria must be a mapping")
        _reject_unknown(data, _PASS_KEYS, "pass_criteria")
        return cls(
            min_deflated_sharpe=_optional_probability(
                data, "min_deflated_sharpe", _DEFAULT_MIN_DEFLATED_SHARPE
            ),
            min_probabilistic_sharpe=_optional_probability(data, "min_probabilistic_sharpe", None),
            min_oos_sharpe_annualized=_optional_float(data, "min_oos_sharpe_annualized", 0.0),
            max_sharpe_decay_annualized=_optional_float(data, "max_sharpe_decay_annualized", None),
            min_oos_observations=_optional_int(
                data, "min_oos_observations", _DEFAULT_MIN_OOS_OBSERVATIONS
            ),
            require_single_holdout_unlock=_optional_bool(
                data, "require_single_holdout_unlock", True
            ),
            reject_post_result_amendments=_optional_bool(
                data, "reject_post_result_amendments", True
            ),
            reject_trials_outside_grid=_optional_bool(data, "reject_trials_outside_grid", True),
            reject_conflicting_reruns=_optional_bool(data, "reject_conflicting_reruns", True),
            require_return_series=_optional_bool(data, "require_return_series", True),
        )


@dataclass(frozen=True)
class HoldoutSpec:
    """Where the locked out-of-sample rows begin.

    ``embargo_periods`` is a count of rows immediately before ``start`` that are
    dropped from the in-sample set, so a label that reaches across the boundary
    is not available while tuning. It is not a calendar duration.
    """

    time_column: str
    start: str | int | float
    embargo_periods: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "time_column": self.time_column,
            "start": self.start,
            "embargo_periods": self.embargo_periods,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> HoldoutSpec:
        if not isinstance(raw, Mapping):
            raise PreRegistrationError("holdout must be a mapping")
        _reject_unknown(raw, _HOLDOUT_KEYS, "holdout")
        column = raw.get("time_column")
        if not isinstance(column, str) or not column:
            raise PreRegistrationError("holdout.time_column must be a non-empty string")
        if "start" not in raw:
            raise PreRegistrationError("holdout.start is required")
        start = _normalize_boundary(raw["start"])
        embargo = raw.get("embargo_periods", 0)
        if isinstance(embargo, bool) or not isinstance(embargo, int) or embargo < 0:
            raise PreRegistrationError("holdout.embargo_periods must be an integer >= 0")
        return cls(time_column=column, start=start, embargo_periods=embargo)


def _normalize_boundary(value: Any) -> str | int | float:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, bool):
        raise PreRegistrationError("holdout.start must be a date, string, or number")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreRegistrationError("holdout.start must be finite")
        return value
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise PreRegistrationError("holdout.start must be a date, string, or number")


def _parse_parameter(name: str, spec: Any) -> dict[str, Any]:
    if isinstance(spec, list):
        if not spec:
            raise PreRegistrationError(f"parameter {name} has an empty values list")
        return {"values": [_parameter_value(name, item) for item in spec]}
    if not isinstance(spec, Mapping):
        raise PreRegistrationError(
            f"parameter {name} must be a list of values or a min/max mapping"
        )
    keys = set(spec)
    if keys == {"values"}:
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise PreRegistrationError(f"parameter {name} values must be a non-empty list")
        return {"values": [_parameter_value(name, item) for item in values]}
    if keys == {"min", "max"}:
        low = _as_number(spec["min"], f"{name}.min")
        high = _as_number(spec["max"], f"{name}.max")
        if low > high:
            raise PreRegistrationError(f"parameter {name} has min greater than max")
        return {"min": low, "max": high}
    raise PreRegistrationError(f"parameter {name}: declare either values or both min and max")


def _parameter_value(name: str, value: Any) -> Any:
    converted = to_jsonable(value)
    if isinstance(converted, float) and not math.isfinite(converted):
        raise PreRegistrationError(f"parameter {name} has a non-finite value")
    if isinstance(converted, (str, int, float, bool)):
        return converted
    raise PreRegistrationError(f"parameter {name} values must be strings, numbers, or booleans")


def parse_parameter_space(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a declared search space."""
    if not isinstance(raw, Mapping) or not raw:
        raise PreRegistrationError("parameters must be a non-empty mapping")
    parsed: dict[str, Any] = {}
    for name, spec in raw.items():
        if not isinstance(name, str) or not name:
            raise PreRegistrationError("parameter names must be non-empty strings")
        parsed[name] = _parse_parameter(name, spec)
    return parsed


def _numbers_close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)
    return False


def value_allowed(value: Any, spec: Mapping[str, Any]) -> bool:
    """Return whether one trial value sits in one declared parameter spec."""
    try:
        candidate = _parameter_value("trial", value)
    except (PreRegistrationError, TypeError, ValueError):
        return False
    if "values" in spec:
        for allowed in spec["values"]:
            if isinstance(candidate, bool) or isinstance(allowed, bool):
                if (
                    isinstance(candidate, bool)
                    and isinstance(allowed, bool)
                    and candidate == allowed
                ):
                    return True
                continue
            if candidate == allowed or _numbers_close(candidate, allowed):
                return True
        return False
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return False
    return float(spec["min"]) <= float(candidate) <= float(spec["max"])


def params_within_grid(params: Mapping[str, Any], space: Mapping[str, Any]) -> bool:
    """A trial matches the grid only when its keys are exactly the declared ones."""
    if set(params) != set(space):
        return False
    return all(value_allowed(params[name], space[name]) for name in space)


@dataclass(frozen=True)
class PreRegistration:
    """Hashed declaration of hypothesis, search, metrics, and pass criteria."""

    study_id: str
    hypothesis: str
    universe: Any
    parameters: dict[str, Any]
    metrics: tuple[str, ...]
    primary_metric: str
    pass_criteria: PassCriteria
    holdout: HoldoutSpec
    periods_per_year: int = 252
    notes: str = ""

    def document(self) -> dict[str, Any]:
        """Normalized document. This mapping is what :func:`content_hash` covers."""
        universe = to_jsonable(self.universe)
        return {
            "study_id": self.study_id,
            "hypothesis": self.hypothesis,
            "universe": universe,
            "parameters": self.parameters,
            "metrics": list(self.metrics),
            "primary_metric": self.primary_metric,
            "pass_criteria": self.pass_criteria.as_dict(),
            "holdout": self.holdout.as_dict(),
            "periods_per_year": self.periods_per_year,
            "notes": self.notes,
        }

    def content_hash(self) -> str:
        return content_hash(self.document())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> PreRegistration:
        if not isinstance(raw, Mapping):
            raise PreRegistrationError("pre-registration must be a mapping")
        _reject_unknown(raw, _TOP_LEVEL, "pre-registration")
        study_id = raw.get("study_id")
        if not isinstance(study_id, str) or not _study_id_ok(study_id):
            raise PreRegistrationError("study_id must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
        hypothesis = raw.get("hypothesis")
        if not isinstance(hypothesis, str) or len(hypothesis.strip()) < 10:
            raise PreRegistrationError("hypothesis must be a string of at least 10 characters")
        hypothesis = hypothesis.strip()
        if "universe" not in raw:
            raise PreRegistrationError("universe is required")
        universe = _normalize_universe(raw["universe"])
        parameters = parse_parameter_space(raw.get("parameters"))  # type: ignore[arg-type]
        metrics_raw = raw.get("metrics")
        if not isinstance(metrics_raw, list) or not metrics_raw:
            raise PreRegistrationError("metrics must be a non-empty list of strings")
        metrics: list[str] = []
        for metric in metrics_raw:
            if not isinstance(metric, str) or not metric:
                raise PreRegistrationError("metrics must be a non-empty list of strings")
            metrics.append(metric)
        if len(set(metrics)) != len(metrics):
            raise PreRegistrationError("metrics contains a duplicate")
        primary = raw.get("primary_metric")
        if not isinstance(primary, str) or primary not in metrics:
            raise PreRegistrationError("primary_metric must be one of metrics")
        periods = raw.get("periods_per_year", 252)
        if isinstance(periods, bool) or not isinstance(periods, int) or periods < 1:
            raise PreRegistrationError("periods_per_year must be an integer >= 1")
        notes = raw.get("notes", "")
        if notes is None:
            notes = ""
        if not isinstance(notes, str):
            raise PreRegistrationError("notes must be a string")
        if "holdout" not in raw:
            raise PreRegistrationError("holdout is required")
        holdout_raw = raw["holdout"]
        if not isinstance(holdout_raw, Mapping):
            raise PreRegistrationError("holdout must be a mapping")
        return cls(
            study_id=study_id,
            hypothesis=hypothesis,
            universe=universe,
            parameters=parameters,
            metrics=tuple(metrics),
            primary_metric=primary,
            pass_criteria=PassCriteria.from_mapping(raw.get("pass_criteria")),  # type: ignore[arg-type]
            holdout=HoldoutSpec.from_mapping(holdout_raw),
            periods_per_year=periods,
            notes=notes.strip(),
        )


def _normalize_universe(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if len(text) < 3:
            raise PreRegistrationError(
                "universe must describe the data the study is allowed to see"
            )
        return text
    if isinstance(value, Mapping):
        if not value:
            raise PreRegistrationError("universe mapping must not be empty")
        return to_jsonable(value)
    raise PreRegistrationError("universe must be a string or a mapping")


def dump_preregistration(spec: PreRegistration) -> str:
    """YAML text of the normalized document."""
    text = yaml.safe_dump(
        spec.document(),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    return text


def load_preregistration(path: str | Path) -> PreRegistration:
    """Load a YAML declaration and normalize it."""
    raw_text = Path(path).read_text(encoding="utf-8")
    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, Mapping):
        raise PreRegistrationError("pre-registration YAML must be a mapping")
    return PreRegistration.from_mapping(loaded)
