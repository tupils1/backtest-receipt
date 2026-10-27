"""Canonical JSON for content hashes.

The hash is over a normalized document, not over raw YAML bytes, so key order
and insignificant whitespace do not change it. The same bytes are used for the
append-only log's line hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import numpy as np

GENESIS = "0" * 64


def to_jsonable(value: Any) -> Any:
    """Return a structure that :func:`json.dumps` can encode without lossy types."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numbers cannot be canonicalized")
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"mapping keys must be strings, got {type(key).__name__}")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"unsupported value type: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """UTF-8 JSON with sorted keys and no insignificant whitespace."""
    payload = to_jsonable(value)
    text = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return text.encode("utf-8")


def content_hash(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_bytes`."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
