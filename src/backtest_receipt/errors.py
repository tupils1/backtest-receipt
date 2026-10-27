"""Errors raised by the research-discipline checks."""

from __future__ import annotations


class BacktestReceiptError(Exception):
    """Base error for this package."""


class PreRegistrationError(BacktestReceiptError):
    """The declaration is missing, internally inconsistent, or drifted from its log."""


class ChainBroken(BacktestReceiptError):
    """An append-only log was edited, truncated in the middle, or is unreadable."""


class HoldoutError(BacktestReceiptError):
    """The sealed holdout could not be written, read, or matched to its declaration."""


class HoldoutKeyError(HoldoutError):
    """The key file did not decrypt the seal."""


class HoldoutAlreadyUnlocked(HoldoutError):
    """A successful unlock was already logged for this seal."""


class TrialError(BacktestReceiptError):
    """A trial could not be recorded under the study rules."""


class ReceiptError(BacktestReceiptError):
    """A receipt file is missing, unreadable, or internally inconsistent."""
