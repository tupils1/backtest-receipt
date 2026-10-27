"""Delegation to purgedcv, and the synthetic demonstration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_receipt import make_combinatorial_purged_cv, make_purged_kfold


def test_purged_kfold_and_cpcv_delegate() -> None:
    prediction = pd.Series(pd.date_range("2024-01-01", periods=30, freq="D"))
    evaluation = prediction + pd.Timedelta(days=1)
    folder = make_purged_kfold(prediction, evaluation, n_splits=5, embargo=pd.Timedelta(days=1))
    folds = list(folder.split(np.zeros((30, 1))))
    assert len(folds) == 5
    for train, test in folds:
        assert set(train).isdisjoint(set(test))
    combinatorial = make_combinatorial_purged_cv(
        prediction,
        evaluation,
        n_splits=4,
        n_test_groups=2,
        embargo_fraction=0.01,
    )
    assert combinatorial.get_n_splits() == 6


def test_synthetic_overfit_fails_and_edge_passes(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "examples" / "synthetic_study.py"
    for mode in ("overfit", "edge"):
        completed = subprocess.run(
            [sys.executable, str(script), "--mode", mode, "--out", str(tmp_path / mode)],
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
