"""Seals, splits, and the append-only chain."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from backtest_receipt.errors import ChainBroken, HoldoutError, HoldoutKeyError, PreRegistrationError
from backtest_receipt.frames import frame_from_payload, frame_to_payload, split_holdout
from backtest_receipt.hashlog import HashChainLog
from backtest_receipt.holdout import inspect_seal, read_seal, write_seal


def _frame(n: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"date": dates, "close": [100.0 + i for i in range(n)], "name": ["row"] * n}
    )


def test_split_drops_the_embargo_gap() -> None:
    frame = _frame(12)
    start = frame["date"].iloc[8].strftime("%Y-%m-%d")
    ins, gap, holdout = split_holdout(frame, time_column="date", start=start, embargo_periods=2)
    assert len(ins) == 6
    assert len(gap) == 2
    assert len(holdout) == 4
    assert ins["date"].max() < gap["date"].min()
    assert gap["date"].max() < holdout["date"].min()


def test_split_rejects_empty_sides_and_naive_timezone() -> None:
    frame = _frame(6)
    with pytest.raises(HoldoutError):
        split_holdout(frame, time_column="date", start="1999-01-01", embargo_periods=0)
    with pytest.raises(HoldoutError):
        split_holdout(frame, time_column="date", start="2099-01-01", embargo_periods=0)
    with pytest.raises(HoldoutError):
        split_holdout(frame, time_column="missing", start="2024-01-01", embargo_periods=0)
    aware = frame.copy()
    aware["date"] = aware["date"].dt.tz_localize("UTC")
    with pytest.raises(PreRegistrationError):
        split_holdout(aware, time_column="date", start="2024-01-08", embargo_periods=0)


def test_numeric_boundary() -> None:
    frame = pd.DataFrame({"day": [1, 2, 3, 4, 5], "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ins, gap, holdout = split_holdout(frame, time_column="day", start=4, embargo_periods=1)
    assert ins["day"].tolist() == [1, 2]
    assert gap["day"].tolist() == [3]
    assert holdout["day"].tolist() == [4, 5]
    with pytest.raises(HoldoutError):
        split_holdout(frame, time_column="day", start="4", embargo_periods=0)


def test_frame_roundtrip_and_missing_integer_rejected() -> None:
    frame = _frame(4)
    frame.loc[1, "close"] = float("nan")
    restored = frame_from_payload(frame_to_payload(frame))
    assert restored["name"].tolist() == frame["name"].tolist()
    assert restored["close"].iloc[0] == pytest.approx(100.0)
    assert pd.isna(restored["close"].iloc[1])
    assert restored["date"].iloc[0] == frame["date"].iloc[0]
    broken = pd.DataFrame({"count": pd.Series([1, pd.NA], dtype="Int64")})
    with pytest.raises(HoldoutError):
        frame_to_payload(broken)


def test_seal_roundtrip_wrong_key_and_no_overwrite(tmp_path) -> None:
    frame = _frame(5)
    seal = tmp_path / "holdout.seal.json"
    key = tmp_path / "holdout.key"
    info = write_seal(
        frame,
        seal_path=seal,
        key_path=key,
        metadata={"prereg_hash": "abc", "n_rows": 5, "columns": ["date", "close", "name"]},
    )
    assert key.stat().st_mode & 0o777 == 0o600
    peeked = inspect_seal(seal)
    assert peeked.seal_id == info.seal_id
    restored, metadata, again = read_seal(seal, key)
    assert metadata["prereg_hash"] == "abc"
    assert again.plaintext_sha256 == info.plaintext_sha256
    assert restored["close"].tolist() == frame["close"].tolist()
    other = tmp_path / "other.key"
    write_seal(
        frame.iloc[:1],
        seal_path=tmp_path / "other.seal.json",
        key_path=other,
        metadata={"prereg_hash": "abc"},
    )
    with pytest.raises(HoldoutKeyError):
        read_seal(seal, other)
    with pytest.raises(HoldoutError):
        write_seal(frame, seal_path=seal, key_path=tmp_path / "fresh.key", metadata={})


def test_tampered_ciphertext_is_rejected(tmp_path) -> None:
    frame = _frame(3)
    seal = tmp_path / "holdout.seal.json"
    key = tmp_path / "holdout.key"
    write_seal(frame, seal_path=seal, key_path=key, metadata={"label": "v"})
    outer = json.loads(seal.read_text(encoding="utf-8"))
    outer["ciphertext"] = outer["ciphertext"][:-2] + (
        "A" if outer["ciphertext"][-1] != "A" else "B"
    )
    seal.write_text(json.dumps(outer), encoding="utf-8")
    with pytest.raises(HoldoutError):
        inspect_seal(seal)


def test_hash_chain_detects_an_edited_line(tmp_path) -> None:
    log = HashChainLog(tmp_path / "events.jsonl")
    log.append("register", {"value": 1}, recorded_at="2026-01-01T00:00:00+00:00")
    log.append("amend", {"value": 2}, recorded_at="2026-01-02T00:00:00+00:00")
    records = log.read_verified()
    assert [record["seq"] for record in records] == [1, 2]
    assert records[1]["prev"] == records[0]["line_hash"]
    text = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(text.replace('"value":1', '"value":9'), encoding="utf-8")
    with pytest.raises(ChainBroken):
        log.read_verified()
    with pytest.raises(ChainBroken):
        log.append("amend", {"value": 3})


def test_hash_chain_rejects_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    log = HashChainLog(path)
    log.append("register", {"value": 1}, recorded_at="2026-01-01T00:00:00+00:00")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ChainBroken):
        log.read_verified()
