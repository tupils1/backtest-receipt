"""Two synthetic studies: a noise search that must fail, and a one-config drift that must pass.

The strategy is a moving-average crossover on an artificial price path. The
position used at each bar is the signal from the previous bar. Nothing here is
a claim about a traded market.

Run from the repository root:

    python examples/synthetic_study.py --mode overfit --out /tmp/overfit-study
    backtest-receipt check /tmp/overfit-study/receipt.json

    python examples/synthetic_study.py --mode edge --out /tmp/edge-study
    backtest-receipt check /tmp/edge-study/receipt.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_receipt import PreRegistration, Study


def make_prices(n: int, drift: float, vol: float, seed: int) -> pd.DataFrame:
    """Daily prices from i.i.d. simple returns. The first close is 100."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, vol, size=n)
    close = np.empty(n, dtype=float)
    close[0] = 100.0
    for i in range(1, n):
        step = shocks[i]
        if step <= -0.95:
            step = -0.95
        close[i] = close[i - 1] * (1.0 + step)
    dates = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"date": dates, "close": close})


def strategy_returns(frame: pd.DataFrame, params: dict[str, int]) -> np.ndarray:
    """Lagged moving-average sign, times the bar's simple return.

    ``position`` is shifted by one bar so the signal at t does not multiply the
    return that formed the close at t.
    """
    close = frame["close"].astype(float)
    fast = close.rolling(int(params["fast"]), min_periods=int(params["fast"])).mean()
    slow = close.rolling(int(params["slow"]), min_periods=int(params["slow"])).mean()
    signal = pd.Series(np.sign(fast.to_numpy() - slow.to_numpy()), index=close.index)
    position = signal.shift(1)
    asset = close.pct_change()
    strategy = (position * asset).dropna()
    return strategy.to_numpy(dtype=float)


def _spec(mode: str, holdout_start: str) -> PreRegistration:
    if mode == "overfit":
        parameters = {"fast": [5, 10, 20, 40], "slow": [50, 80, 120, 160]}
        hypothesis = "Some moving-average pair on this synthetic path has a positive Sharpe ratio."
        notes = "Noise search. The prose is stored and hashed, and it does not enter the math."
    elif mode == "edge":
        parameters = {"fast": [10], "slow": [30]}
        hypothesis = "One pre-declared moving-average pair rides a constant positive drift."
        notes = "Single configuration, so the deflator has no selection term."
    else:
        raise ValueError("mode must be overfit or edge")
    return PreRegistration.from_mapping(
        {
            "study_id": f"synthetic-{mode}",
            "hypothesis": hypothesis,
            "universe": "one synthetic daily price path with no listed venue",
            "parameters": parameters,
            "metrics": ["sharpe", "mean_return"],
            "primary_metric": "sharpe",
            "periods_per_year": 252,
            "notes": notes,
            "holdout": {"time_column": "date", "start": holdout_start, "embargo_periods": 5},
            "pass_criteria": {
                "min_deflated_sharpe": 0.95,
                "min_oos_sharpe_annualized": 0.0,
                "min_oos_observations": 30,
                "max_sharpe_decay_annualized": None,
            },
        }
    )


def run(mode: str, out: str | Path, *, seed: int | None = None) -> dict:
    """Create ``out``, run the study, and return the receipt.

    The process exits through ``main``. This function itself does not check
    the gate; ``main`` does, so a demonstration that fails to fail (or fails
    to pass) is reported by the process status.
    """
    if mode == "overfit":
        drift, vol, chosen = 0.0, 0.01, 7 if seed is None else seed
    else:
        drift, vol, chosen = 0.0015, 0.008, 1 if seed is None else seed
    prices = make_prices(1500, drift, vol, chosen)
    boundary = prices["date"].iloc[1500 - 450]
    spec = _spec(mode, boundary.strftime("%Y-%m-%d"))
    study = Study.create(spec, out)
    in_sample = study.seal_holdout(prices)
    fast_values = spec.parameters["fast"]["values"]
    slow_values = spec.parameters["slow"]["values"]
    for fast in fast_values:
        for slow in slow_values:
            params = {"fast": fast, "slow": slow}
            study.record_trial(params, returns=strategy_returns(in_sample, params))
    study.evaluate_holdout(lambda frame, params: strategy_returns(frame, params))
    return study.issue_receipt()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic pre-registration demo")
    parser.add_argument("--mode", choices=("overfit", "edge"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    receipt = run(args.mode, args.out, seed=args.seed)
    dsr = receipt["statistics"]["deflated_sharpe"]
    oos = receipt["holdout"]["oos_sharpe_annualized"]
    print(f"status={receipt['status']} dsr={dsr} oos={oos}")
    print(f"receipt={args.out / 'receipt.json'}")
    expected = "fail" if args.mode == "overfit" else "pass"
    if receipt["status"] != expected:
        print(f"demo expected status {expected}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
