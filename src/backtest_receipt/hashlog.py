"""Append-only JSONL log with a hash chain.

Each line commits to the previous line's hash. Rewriting an earlier line, or
splicing the file, breaks verification. Deleting the tail and everything that
pointed at it is not detectable from the file alone; keep the study directory
in version control if you need that guarantee.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest_receipt.canonical import GENESIS, content_hash
from backtest_receipt.errors import ChainBroken

_UNSIGNED = ("seq", "prev", "event", "recorded_at", "body")


class HashChainLog:
    """A single append-only chain stored as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_verified(self) -> list[dict[str, Any]]:
        """Return every record, or raise if the chain does not verify."""
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        if text == "":
            return []
        records: list[dict[str, Any]] = []
        previous = GENESIS
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.strip() == "":
                raise ChainBroken(f"{self.path}:{lineno}: blank line in append-only log")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChainBroken(f"{self.path}:{lineno}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ChainBroken(f"{self.path}:{lineno}: record is not an object")
            try:
                unsigned = {key: record[key] for key in _UNSIGNED}
            except KeyError as exc:
                raise ChainBroken(f"{self.path}:{lineno}: missing {exc}") from exc
            expected = content_hash(unsigned)
            seq = len(records) + 1
            if (
                record.get("line_hash") != expected
                or record.get("seq") != seq
                or record.get("prev") != previous
            ):
                raise ChainBroken(f"{self.path}:{lineno}: hash chain does not verify")
            if not isinstance(record["event"], str) or not isinstance(record["body"], dict):
                raise ChainBroken(f"{self.path}:{lineno}: malformed event")
            previous = expected
            records.append(record)
        return records

    def append(
        self, event: str, body: dict[str, Any], *, recorded_at: str | None = None
    ) -> dict[str, Any]:
        """Verify the current chain, then append one record.

        ``recorded_at`` is injectable so tests can be deterministic. Production
        callers should leave it empty.
        """
        if not event or not isinstance(body, dict):
            raise ChainBroken("event name and object body are required")
        records = self.read_verified()
        previous = records[-1]["line_hash"] if records else GENESIS
        when = recorded_at if recorded_at is not None else datetime.now(UTC).isoformat()
        unsigned = {
            "seq": len(records) + 1,
            "prev": previous,
            "event": event,
            "recorded_at": when,
            "body": body,
        }
        record = {**unsigned, "line_hash": content_hash(unsigned)}
        line = json.dumps(
            record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record
