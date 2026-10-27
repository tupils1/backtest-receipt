"""Purged and combinatorial cross-validation, delegated to purgedcv.

These splitters are for model fitting inside the in-sample window. They do not
see the sealed holdout, and they do not count trials. Each configuration you
keep after looking at a cross-validation score still has to be recorded with
:meth:`backtest_receipt.Study.record_trial`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from purgedcv import CombinatorialPurgedCV, PurgedKFold


def make_purged_kfold(
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    *,
    n_splits: int = 5,
    purge_horizon: Any = None,
    embargo: Any = None,
    embargo_fraction: float | None = None,
) -> PurgedKFold:
    """Return a purged k-fold splitter from ``purgedcv``.

    ``prediction_times`` and ``evaluation_times`` describe each row's label
    horizon so training rows that overlap a test fold can be purged. ``embargo``
    is a duration after each test label, forwarded unchanged.
    """
    return PurgedKFold(
        n_splits,
        prediction_times=prediction_times,
        evaluation_times=evaluation_times,
        purge_horizon=purge_horizon,
        embargo=embargo,
        embargo_fraction=embargo_fraction,
    )


def make_combinatorial_purged_cv(
    prediction_times: pd.Series,
    evaluation_times: pd.Series,
    *,
    n_splits: int,
    n_test_groups: int,
    purge_horizon: Any = None,
    embargo: Any = None,
    embargo_fraction: float | None = None,
) -> CombinatorialPurgedCV:
    """Return a combinatorial purged CV splitter from ``purgedcv``.

    The number of folds is the combination count ``C(n_splits, n_test_groups)``.
    """
    return CombinatorialPurgedCV(
        n_splits,
        n_test_groups,
        prediction_times=prediction_times,
        evaluation_times=evaluation_times,
        purge_horizon=purge_horizon,
        embargo=embargo,
        embargo_fraction=embargo_fraction,
    )
