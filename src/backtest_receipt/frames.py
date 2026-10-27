"""Split a table on a pre-registered boundary and serialize it for sealing.

The index is dropped. Callers that need a row identifier should keep it as a
column. Datetime columns must be timezone-naive; convert to UTC and drop the
timezone before sealing so the comparison with ``holdout.start`` is unambiguous.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from backtest_receipt.errors import HoldoutError, PreRegistrationError


def split_holdout(
    frame: pd.DataFrame,
    *,
    time_column: str,
    start: str | int | float,
    embargo_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(in_sample, embargo, holdout)``.

    Rows at or after ``start`` are the holdout. The ``embargo_periods`` rows
    immediately before that boundary are in neither set.
    """
    if time_column not in frame.columns:
        raise HoldoutError(f"time column {time_column!r} is not in the frame")
    if frame.columns.has_duplicates:
        raise HoldoutError("duplicate column names")
    if (
        isinstance(embargo_periods, bool)
        or not isinstance(embargo_periods, int)
        or embargo_periods < 0
    ):
        raise HoldoutError("embargo_periods must be an integer >= 0")
    ordered = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    column = ordered[time_column]
    boundary = _coerce_boundary(column, start)
    try:
        at_or_after = column >= boundary
    except TypeError as exc:
        raise HoldoutError("holdout.start is not comparable with the time column") from exc
    positions = at_or_after.to_numpy().nonzero()[0]
    if len(positions) == 0:
        raise HoldoutError("holdout is empty; no row is at or after holdout.start")
    first = int(positions[0])
    if first == 0:
        raise HoldoutError("in-sample is empty; every row is at or after holdout.start")
    if embargo_periods > first:
        raise HoldoutError("embargo_periods consumes the entire in-sample")
    in_sample = ordered.iloc[: first - embargo_periods].reset_index(drop=True)
    embargo = ordered.iloc[first - embargo_periods : first].reset_index(drop=True)
    holdout = ordered.iloc[first:].reset_index(drop=True)
    if in_sample.empty or holdout.empty:
        raise HoldoutError("split produced an empty in-sample or holdout")
    return in_sample, embargo, holdout


def _coerce_boundary(column: pd.Series, start: str | int | float) -> Any:
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        raise PreRegistrationError("time column must be timezone-naive")
    if pd.api.types.is_datetime64_any_dtype(column):
        try:
            return pd.Timestamp(start)
        except (TypeError, ValueError) as exc:
            raise HoldoutError("holdout.start is not a timestamp") from exc
    if pd.api.types.is_numeric_dtype(column):
        if isinstance(start, str):
            raise HoldoutError("numeric time columns require a numeric holdout.start")
        return start
    return str(start)


def frame_to_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Serialize a frame to a JSON-ready payload, preserving a small set of dtypes."""
    if frame.columns.has_duplicates:
        raise HoldoutError("duplicate column names")
    columns: list[dict[str, Any]] = []
    data: dict[str, list[Any]] = {}
    for name in frame.columns:
        if not isinstance(name, str):
            raise HoldoutError("column names must be strings")
        series = frame[name]
        kind, values = _encode_series(series)
        columns.append({"name": name, "kind": kind})
        data[name] = values
    return {"columns": columns, "data": data}


def frame_from_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Inverse of :func:`frame_to_payload`."""
    columns = payload["columns"]
    data = payload["data"]
    restored: dict[str, pd.Series] = {}
    order: list[str] = []
    for column in columns:
        name = column["name"]
        order.append(name)
        restored[name] = _decode_series(name, column["kind"], data[name])
    return pd.DataFrame(restored, columns=order)


def payload_bytes(payload: dict[str, Any]) -> bytes:
    """Stable bytes of a payload, used as the plaintext that gets encrypted."""
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def _encode_series(series: pd.Series) -> tuple[str, list[Any]]:
    if pd.api.types.is_datetime64_any_dtype(series):
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            raise PreRegistrationError("time column must be timezone-naive")
        values = [None if pd.isna(item) else pd.Timestamp(item).isoformat() for item in series]
        return "datetime", values
    if pd.api.types.is_bool_dtype(series):
        values = [None if pd.isna(item) else bool(item) for item in series]
        return "bool", values
    if pd.api.types.is_integer_dtype(series):
        values = []
        for item in series:
            if pd.isna(item):
                raise HoldoutError(
                    "integer columns with missing values cannot be sealed; use floats"
                )
            values.append(int(item))
        return "int", values
    if pd.api.types.is_float_dtype(series):
        values = [None if pd.isna(item) else float(item) for item in series]
        return "float", values
    values = [None if pd.isna(item) else str(item) for item in series]
    return "str", values


def _decode_series(name: str, kind: str, values: list[Any]) -> pd.Series:
    if kind == "datetime":
        return pd.to_datetime(pd.Series(values, name=name))
    if kind == "bool":
        return pd.Series(values, dtype="boolean", name=name)
    if kind == "int":
        return pd.Series(values, dtype="int64", name=name)
    if kind == "float":
        restored = [float("nan") if item is None else float(item) for item in values]
        return pd.Series(restored, dtype="float64", name=name)
    if kind == "str":
        return pd.Series(values, dtype="object", name=name)
    raise HoldoutError(f"unknown column kind {kind!r}")
