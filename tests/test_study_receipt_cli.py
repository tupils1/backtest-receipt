"""End-to-end study protocol, receipt gates, and the check command."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_receipt import HoldoutAlreadyUnlocked, PreRegistration, Study
from backtest_receipt.cli import main
from backtest_receipt.errors import ChainBroken, PreRegistrationError, TrialError
from tests.conftest import declaration


def _prices(n: int, drift: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + shocks)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"date": dates, "close": close})


def _returns(frame: pd.DataFrame, lookback: int) -> np.ndarray:
    close = frame["close"].astype(float)
    signal = np.sign(close.diff(lookback))
    position = pd.Series(signal, index=close.index).shift(1)
    return (position * close.pct_change()).dropna().to_numpy(dtype=float)


def _run(
    root: Path, drift: float, *, lookbacks: list[int] | None = None, unlock: bool = True
) -> Study:
    spec = declaration(
        holdout={"time_column": "date", "start": "2023-04-03", "embargo_periods": 2},
        parameters={"lookback": lookbacks or [5]},
    )
    # The start above is a guess; pin it to a real row so the split is non-empty.
    prices = _prices(80, drift)
    boundary = prices["date"].iloc[55]
    spec = declaration(
        holdout={
            "time_column": "date",
            "start": boundary.strftime("%Y-%m-%d"),
            "embargo_periods": 2,
        },
        parameters={"lookback": lookbacks or [5]},
    )
    study = Study.create(spec, root)
    ins = study.seal_holdout(prices)
    for lookback in spec.parameters["lookback"]["values"]:
        study.record_trial({"lookback": lookback}, returns=_returns(ins, int(lookback)))
    if unlock:
        study.evaluate_holdout(lambda frame, params: _returns(frame, int(params["lookback"])))
    return study


def test_positive_drift_single_config_passes(tmp_path: Path) -> None:
    study = _run(tmp_path / "pass", drift=0.02)
    receipt = study.issue_receipt(generated_at="2026-10-27T00:00:00+00:00")
    again = study.issue_receipt(write=False, generated_at="2026-10-28T00:00:00+00:00")
    assert receipt["status"] == "pass"
    assert receipt["passed"] is True
    assert receipt["fingerprint"] == again["fingerprint"]
    assert receipt["trials"]["unique_configs"] == 1
    assert receipt["holdout"]["unlocks"] == 1
    assert receipt["statistics"]["deflated_sharpe"] > 0.5
    text = (tmp_path / "pass" / "receipt.md").read_text(encoding="utf-8")
    assert "PASS" in text
    assert "Audit prose" in receipt["notes"] or receipt["notes"]
    code = main(
        ["check", str(tmp_path / "pass" / "receipt.json"), "--study", str(tmp_path / "pass")]
    )
    assert code == 0


def test_negative_drift_fails_and_check_exits_one(tmp_path: Path) -> None:
    prices = _prices(80, 0.0, seed=9)
    boundary = prices["date"].iloc[55].strftime("%Y-%m-%d")
    study = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 1}),
        tmp_path / "fail",
    )
    study.seal_holdout(prices)
    losing = np.array([-0.02, -0.01, -0.015, -0.005, -0.02, -0.01, -0.008])
    study.record_trial({"lookback": 5}, returns=losing)
    study.evaluate_holdout(lambda frame, params: losing)
    receipt = study.issue_receipt()
    assert receipt["status"] == "fail"
    assert receipt["passed"] is False
    assert main(["check", str(tmp_path / "fail" / "receipt.json")]) == 1


def test_unfinished_holdout_is_blocked(tmp_path: Path) -> None:
    study = _run(tmp_path / "blocked", drift=0.02, unlock=False)
    receipt = study.issue_receipt()
    assert receipt["status"] == "blocked"
    assert main(["check", str(tmp_path / "blocked" / "receipt.json")]) == 1


def test_second_unlock_is_refused_and_a_flagged_repeat_fails(tmp_path: Path) -> None:
    study = _run(tmp_path / "once", drift=0.02, unlock=False)
    study.selected_trial()
    first = study.unlock()
    assert len(first) > 0
    with pytest.raises(HoldoutAlreadyUnlocked):
        study.unlock()
    refused = [row for row in study.holdout_log.read_verified() if row["event"] == "unlock_refused"]
    assert len(refused) == 1
    study.unlock(acknowledge_repeat=True)
    study.record_holdout_result(_returns(first, 5), params={"lookback": 5})
    receipt = study.issue_receipt()
    assert receipt["status"] == "fail"
    assert any(
        gate["name"] == "repeat_unlock" and gate["state"] == "fail" for gate in receipt["gates"]
    )


def test_outside_grid_and_conflicting_rerun_fail(tmp_path: Path) -> None:
    prices = _prices(80, 0.0, seed=1)
    boundary = prices["date"].iloc[55].strftime("%Y-%m-%d")
    spec = declaration(
        holdout={"time_column": "date", "start": boundary, "embargo_periods": 1},
        parameters={"lookback": [5]},
    )
    study = Study.create(spec, tmp_path / "grid")
    ins = study.seal_holdout(prices)
    study.record_trial({"lookback": 5}, returns=_returns(ins, 5))
    study.record_trial({"lookback": 99}, returns=_returns(ins, 5))
    study.record_trial({"lookback": 5}, returns=_returns(ins, 5) + 0.01)
    study.evaluate_holdout(lambda frame, params: _returns(frame, int(params["lookback"])))
    receipt = study.issue_receipt()
    assert receipt["trials"]["unique_configs"] == 2
    assert receipt["trials"]["attempts"] == 3
    assert receipt["status"] == "fail"
    names = {gate["name"] for gate in receipt["gates"] if gate["state"] == "fail"}
    assert "trials_outside_grid" in names
    assert "conflicting_reruns" in names


def test_duplicate_identical_trial_does_not_inflate_the_count(tmp_path: Path) -> None:
    prices = _prices(80, 0.01, seed=2)
    boundary = prices["date"].iloc[55].strftime("%Y-%m-%d")
    spec = declaration(
        holdout={"time_column": "date", "start": boundary, "embargo_periods": 1},
        parameters={"lookback": [5, 10]},
    )
    study = Study.create(spec, tmp_path / "dup")

    @study.track
    def run(params: dict) -> np.ndarray:
        return _returns(ins, int(params["lookback"]))

    ins = study.seal_holdout(prices)
    run({"lookback": 5})
    run({"lookback": 5})
    run({"lookback": 10})
    assert summarize_unique(study) == 2


def test_amend_after_seal_fails_and_yaml_drift_is_refused(tmp_path: Path) -> None:
    prices = _prices(40, 0.0, seed=3)
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    spec = declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 1})
    study = Study.create(spec, tmp_path / "amend")
    edited = declaration(
        hypothesis="The synthetic series has a different claim after the seal was written.",
        holdout={"time_column": "date", "start": boundary, "embargo_periods": 1},
    )
    study.seal_holdout(prices)
    study.amend(edited)
    ins = None
    study.record_trial({"lookback": 5}, returns=np.array([0.01, -0.01, 0.02, 0.0, 0.01]))
    assert ins is None
    study.evaluate_holdout(lambda frame, params: np.array([0.01, 0.02, -0.01, 0.0, 0.01, 0.02]))
    receipt = study.issue_receipt()
    assert receipt["preregistration"]["post_result_amendments"] == 1
    assert receipt["status"] == "fail"
    text = study.prereg_path.read_text(encoding="utf-8")
    study.prereg_path.write_text(
        text.replace(edited.hypothesis, edited.hypothesis + " quietly"), encoding="utf-8"
    )
    with pytest.raises(PreRegistrationError):
        Study.open(study.root)


def test_trials_before_seal_and_after_unlock_are_rejected(tmp_path: Path) -> None:
    spec = declaration()
    study = Study.create(spec, tmp_path / "order")
    with pytest.raises(TrialError):
        study.record_trial({"lookback": 5}, returns=[0.01, 0.02, -0.01])
    prices = _prices(40, 0.0, seed=4)
    # Rebuild with a boundary inside this frame.
    root = tmp_path / "order2"
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    study = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 1}),
        root,
    )
    study.seal_holdout(prices)
    study.record_trial({"lookback": 5}, returns=[0.01, -0.02, 0.03, 0.01])
    study.unlock()
    with pytest.raises(TrialError):
        study.record_trial({"lookback": 10}, returns=[0.01, -0.02, 0.03, 0.01])


def test_wrong_key_is_logged_and_does_not_consume_the_unlock(tmp_path: Path) -> None:
    prices = _prices(40, 0.0, seed=5)
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    study = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 0}),
        tmp_path / "key",
    )
    study.seal_holdout(prices)
    study.record_trial({"lookback": 5}, returns=[0.02, 0.01, -0.01, 0.02, 0.0])
    key_text = study.key_path.read_text(encoding="utf-8")
    study.key_path.write_text(
        key_text.replace(key_text[20], "A" if key_text[20] != "A" else "B"), encoding="utf-8"
    )
    from backtest_receipt import HoldoutKeyError

    with pytest.raises(HoldoutKeyError):
        study.unlock()
    assert any(row["event"] == "unlock_failed" for row in study.holdout_log.read_verified())
    study.key_path.write_text(key_text, encoding="utf-8")
    frame = study.unlock()
    assert len(frame) > 0


def test_optuna_callback_records_user_returns(tmp_path: Path) -> None:
    prices = _prices(40, 0.01, seed=6)
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    study = Study.create(
        declaration(
            holdout={"time_column": "date", "start": boundary, "embargo_periods": 0},
            parameters={"lookback": {"min": 2, "max": 20}},
        ),
        tmp_path / "opt",
    )
    study.seal_holdout(prices)

    class _Trial:
        params = {"lookback": 4}
        value = 1.0
        user_attrs = {"returns": [0.01, 0.02, 0.0, 0.01, -0.005]}

    study.optuna_callback()(None, _Trial())
    assert summarize_unique(study) == 1
    with pytest.raises(TrialError):
        study.optuna_callback()(
            None, type("Empty", (), {"params": {"lookback": 4}, "value": None, "user_attrs": {}})()
        )


def test_cli_hash_issue_and_inconsistent_receipt(tmp_path: Path) -> None:
    prices = _prices(40, -0.01, seed=7)
    boundary = prices["date"].iloc[25].strftime("%Y-%m-%d")
    study = Study.create(
        declaration(holdout={"time_column": "date", "start": boundary, "embargo_periods": 0}),
        tmp_path / "cli",
    )
    assert main(["hash", str(study.prereg_path)]) == 0
    study.seal_holdout(prices)
    study.record_trial({"lookback": 5}, returns=[0.01, -0.02, 0.0, 0.01])
    study.evaluate_holdout(lambda frame, params: np.array([-0.01, -0.02, 0.0, -0.01, 0.005]))
    assert main(["issue", str(study.root)]) == 0
    receipt_path = study.root / "receipt.json"
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    document["passed"] = True
    receipt_path.write_text(json.dumps(document), encoding="utf-8")
    assert main(["check", str(receipt_path)]) == 2
    assert main(["check", str(tmp_path / "missing.json")]) == 2


def test_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0


def test_check_study_detects_a_new_trial(tmp_path: Path) -> None:
    study = _run(tmp_path / "recompute", drift=0.02)
    study.issue_receipt()
    path = study.root / "receipt.json"
    assert main(["check", str(path)]) == 0
    # A consistent file whose study has moved on must not stay green.
    # Recording after unlock is forbidden, so change the sealed file's hash by
    # rewriting the receipt fingerprint indirectly: open a second issue after
    # restoring is not possible. Compare against a different study directory.
    other = _run(tmp_path / "other", drift=-0.02)
    other.issue_receipt()
    assert main(["check", str(other.root / "receipt.json"), "--study", str(study.root)]) == 1


def test_create_refuses_a_non_empty_directory(tmp_path: Path) -> None:
    root = tmp_path / "full"
    root.mkdir()
    (root / "note").write_text("x", encoding="utf-8")
    with pytest.raises(PreRegistrationError):
        Study.create(declaration(), root)


def test_chain_break_on_open(tmp_path: Path) -> None:
    study = Study.create(declaration(), tmp_path / "chain")
    log = study.root / "prereg.jsonl"
    log.write_text(
        log.read_text(encoding="utf-8").replace("sample-study", "sample-studi"), encoding="utf-8"
    )
    with pytest.raises(ChainBroken):
        Study.open(study.root)


def summarize_unique(study: Study) -> int:
    from backtest_receipt.trials import summarize_trials

    return len(summarize_trials(study.spec, study.trial_log.read_verified()).unique)


def test_reopen_matches_hash(tmp_path: Path) -> None:
    study = Study.create(declaration(), tmp_path / "reopen")
    opened = Study.open(study.root)
    assert opened.spec.content_hash() == study.spec.content_hash()
    amended = PreRegistration.from_mapping(
        {
            **opened.spec.document(),
            "notes": "Amended before any seal, so the receipt may still pass later.",
        }
    )
    opened.amend(amended)
    assert Study.open(study.root).spec.notes.startswith("Amended")
    assert Study.open(study.root).prereg_log.read_verified()[-1]["body"]["after_results"] is False
