"""Error paths and gates that the happy-path tests do not reach."""

from __future__ import annotations

import json
import runpy
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from backtest_receipt import Study, content_hash
from backtest_receipt.errors import (
    ChainBroken,
    HoldoutError,
    PreRegistrationError,
    TrialError,
)
from backtest_receipt.frames import frame_from_payload, frame_to_payload, split_holdout
from backtest_receipt.hashlog import HashChainLog
from backtest_receipt.holdout import read_seal
from backtest_receipt.stats import (
    expected_maximum_sharpe,
    null_sharpe_dispersion,
    probabilistic_sharpe_ratio,
)
from backtest_receipt.trials import build_holdout_body, build_trial_body
from tests.conftest import declaration


def test_canonical_numpy_dates_and_bad_keys() -> None:
    stamped = datetime(2024, 1, 2, 3, 4, 5)
    assert content_hash(
        {"n": np.int64(3), "x": np.float64(1.5), "flag": np.bool_(True), "when": stamped, "day": date(2024, 1, 2)}
    )
    with pytest.raises(TypeError):
        content_hash({1: "nope"})
    with pytest.raises(ValueError):
        content_hash(np.float64("nan"))


def test_split_and_frame_edges() -> None:
    frame = pd.DataFrame({"day": list("abcdef"), "flag": [True, False, True, False, True, False], "count": [1, 2, 3, 4, 5, 6]})
    ins, gap, holdout = split_holdout(frame, time_column="day", start="d", embargo_periods=1)
    assert ins["day"].tolist() == ["a", "b"]
    assert gap["day"].tolist() == ["c"]
    assert holdout["day"].tolist() == ["d", "e", "f"]
    payload = frame_to_payload(ins)
    assert frame_from_payload(payload)["flag"].tolist() == [True, False]
    with pytest.raises(HoldoutError):
        frame_from_payload({"columns": [{"name": "x", "kind": "nope"}], "data": {"x": [1]}})
    duplicated = frame.copy()
    duplicated.columns = ["day", "day", "count"]
    with pytest.raises(HoldoutError):
        split_holdout(duplicated, time_column="day", start="d", embargo_periods=0)
    with pytest.raises(HoldoutError):
        split_holdout(frame, time_column="day", start="c", embargo_periods=5)
    with pytest.raises(HoldoutError):
        split_holdout(frame.iloc[0:0], time_column="day", start="a", embargo_periods=0)


def test_hash_chain_rejects_garbage(tmp_path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ChainBroken):
        HashChainLog(path).read_verified()
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ChainBroken):
        HashChainLog(path).read_verified()
    with pytest.raises(ChainBroken):
        HashChainLog(tmp_path / "fresh.jsonl").append("", {})


def test_metadata_tamper_and_bad_probability_inputs(tmp_path) -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=3), "close": [1.0, 2.0, 3.0]})
    from backtest_receipt.holdout import write_seal

    seal = tmp_path / "seal.json"
    key = tmp_path / "key"
    write_seal(frame, seal_path=seal, key_path=key, metadata={"prereg_hash": "abc"})
    outer = json.loads(seal.read_text(encoding="utf-8"))
    outer["metadata"] = {"prereg_hash": "changed"}
    seal.write_text(json.dumps(outer), encoding="utf-8")
    with pytest.raises(HoldoutError):
        read_seal(seal, key)
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, n_obs=10, skewness=0.0, kurtosis=3.0, benchmark=float("nan"))
    with pytest.raises(ValueError):
        null_sharpe_dispersion(1)
    with pytest.raises(ValueError):
        expected_maximum_sharpe(10**18, 1.0)


def test_decay_gate_null_fallback_and_protocol_errors(tmp_path) -> None:
    prices = pd.DataFrame(
        {"date": pd.bdate_range("2023-01-02", periods=40), "close": np.linspace(100, 110, 40)}
    )
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    spec = declaration(
        holdout={"time_column": "date", "start": boundary, "embargo_periods": 1},
        parameters={"lookback": [5, 10]},
        pass_criteria={
            "min_deflated_sharpe": 0.0,
            "min_oos_sharpe_annualized": None,
            "min_oos_observations": 2,
            "max_sharpe_decay_annualized": 0.0,
        },
    )
    study = Study.create(spec, tmp_path / "decay")
    study.seal_holdout(prices)
    varying = np.array([0.02, 0.01, -0.005, 0.02, 0.015, 0.01])
    study.record_trial({"lookback": 5}, returns=varying)
    study.record_trial({"lookback": 10}, returns=np.array([0.01, 0.01, 0.01, 0.01]))
    study.evaluate_holdout(lambda frame, params: np.array([-0.01, -0.02, 0.0, -0.005, 0.001]))
    receipt = study.issue_receipt()
    assert receipt["statistics"]["dispersion_source"] == "null_fallback"
    assert any(gate["name"] == "sharpe_decay" and gate["state"] == "fail" for gate in receipt["gates"])

    with pytest.raises(HoldoutError):
        study.seal_holdout(prices)
    with pytest.raises(PreRegistrationError):
        study.amend(declaration(study_id="other-study", holdout=spec.holdout.as_dict()))
    flat = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 0}),
        tmp_path / "flat",
    )
    flat.seal_holdout(prices)
    flat.record_trial({"lookback": 5}, returns=np.array([0.01, 0.01, 0.01, 0.01]))
    with pytest.raises(TrialError):
        flat.selected_trial()
    with pytest.raises(HoldoutError):
        flat.record_holdout_result([0.01, -0.01, 0.02], params={"lookback": 5})

    tracked = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 0}),
        tmp_path / "tracked",
    )
    tracked.seal_holdout(prices)

    @tracked.track
    def run(params: dict) -> dict:
        return {"returns": [0.01, -0.01, 0.02, 0.0], "metrics": {"mean_return": 0.005}}

    run({"lookback": 5})
    callback = tracked.optuna_callback()

    class _Trial:
        params = {"lookback": 10}
        value = None
        user_attrs = {"metrics": {"sharpe": 0.4, "mean_return": 0.01}, "returns": [0.02, 0.0, -0.01, 0.01]}

    callback(None, _Trial())
    assert len(tracked.trial_log.read_verified()) == 2

    yaml_less = Study.create(declaration(), tmp_path / "yaml")
    yaml_less.prereg_path.unlink()
    restored = Study.open(yaml_less.root)
    assert restored.prereg_path.exists()
    empty = Study.create(declaration(study_id="empty-study"), tmp_path / "empty")
    blocked = empty.issue_receipt()
    assert blocked["status"] == "blocked"
    with pytest.raises(TrialError):
        build_trial_body(empty.spec, {"lookback": 5}, [0.01, 0.02], {"not-a-metric": 1}, prereg_hash="x")
    with pytest.raises(TrialError):
        build_holdout_body({"lookback": 5}, None)


def test_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["backtest-receipt", "--version"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("backtest_receipt", run_name="__main__")
    assert caught.value.code == 0
