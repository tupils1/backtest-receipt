"""Research discipline layer for backtests.

Declare the study, seal the holdout, count every configuration, and issue a
receipt a CI job can reject. Probabilistic and deflated Sharpe ratios are
implemented in this package. Purged and combinatorial cross-validation are
delegated to ``purgedcv``.
"""

from backtest_receipt.canonical import content_hash
from backtest_receipt.cv import make_combinatorial_purged_cv, make_purged_kfold
from backtest_receipt.errors import (
    BacktestReceiptError,
    ChainBroken,
    HoldoutAlreadyUnlocked,
    HoldoutError,
    HoldoutKeyError,
    PreRegistrationError,
    ReceiptError,
    TrialError,
)
from backtest_receipt.prereg import PreRegistration
from backtest_receipt.stats import (
    SharpeMoments,
    annualized_sharpe,
    deflated_sharpe_ratio,
    estimate_sharpe,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)
from backtest_receipt.study import Study
from backtest_receipt.version import __version__

__all__ = [
    "BacktestReceiptError",
    "ChainBroken",
    "HoldoutAlreadyUnlocked",
    "HoldoutError",
    "HoldoutKeyError",
    "PreRegistration",
    "PreRegistrationError",
    "ReceiptError",
    "SharpeMoments",
    "Study",
    "TrialError",
    "__version__",
    "annualized_sharpe",
    "content_hash",
    "deflated_sharpe_ratio",
    "estimate_sharpe",
    "expected_maximum_sharpe",
    "make_combinatorial_purged_cv",
    "make_purged_kfold",
    "probabilistic_sharpe_ratio",
]
