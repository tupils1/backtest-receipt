"""Shared declarations for the discipline-layer tests."""

from __future__ import annotations

from backtest_receipt import PreRegistration


def declaration(**overrides: object) -> PreRegistration:
    raw: dict = {
        "study_id": "sample-study",
        "hypothesis": "The synthetic series has a positive mean after costs of zero.",
        "universe": "synthetic daily series",
        "parameters": {"lookback": [5, 10]},
        "metrics": ["sharpe", "mean_return"],
        "primary_metric": "sharpe",
        "periods_per_year": 252,
        "notes": "Audit prose stays out of the estimator.",
        "holdout": {"time_column": "date", "start": "2024-06-03", "embargo_periods": 2},
        "pass_criteria": {
            "min_deflated_sharpe": 0.5,
            "min_oos_sharpe_annualized": 0.0,
            "min_oos_observations": 5,
        },
    }
    raw.update(overrides)
    return PreRegistration.from_mapping(raw)
