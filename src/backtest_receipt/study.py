"""A study directory: declaration, sealed holdout, trial log, and receipt.

The supported order is: create, optionally amend, seal the holdout, record
in-sample trials, decrypt the holdout once, record the out-of-sample returns,
then issue the receipt. Amending, or decrypting again, after results exist is
logged and fails the receipt.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_receipt.errors import (
    HoldoutAlreadyUnlocked,
    HoldoutError,
    HoldoutKeyError,
    PreRegistrationError,
    TrialError,
)
from backtest_receipt.frames import split_holdout
from backtest_receipt.hashlog import HashChainLog
from backtest_receipt.holdout import inspect_seal, read_seal, write_seal
from backtest_receipt.prereg import PreRegistration, dump_preregistration, load_preregistration
from backtest_receipt.receipt import build_receipt, render_markdown
from backtest_receipt.trials import build_holdout_body, build_trial_body, summarize_trials

_RESULT_EVENTS = {"seal", "unlock", "unlock_repeat", "holdout_result"}


class Study:
    """Filesystem-backed research study."""

    def __init__(self, root: str | Path, spec: PreRegistration) -> None:
        self.root = Path(root)
        self.spec = spec

    @property
    def prereg_path(self) -> Path:
        return self.root / "prereg.yaml"

    @property
    def seal_path(self) -> Path:
        return self.root / "holdout.seal.json"

    @property
    def key_path(self) -> Path:
        return self.root / "holdout.key"

    @property
    def prereg_log(self) -> HashChainLog:
        return HashChainLog(self.root / "prereg.jsonl")

    @property
    def trial_log(self) -> HashChainLog:
        return HashChainLog(self.root / "trials.jsonl")

    @property
    def holdout_log(self) -> HashChainLog:
        return HashChainLog(self.root / "holdout.jsonl")

    @classmethod
    def create(cls, spec: PreRegistration, root: str | Path) -> Study:
        """Create an empty study and append the first registration."""
        directory = Path(root)
        if directory.exists() and any(directory.iterdir()):
            raise PreRegistrationError(f"{directory} is not an empty directory")
        directory.mkdir(parents=True, exist_ok=True)
        study = cls(directory, spec)
        study.prereg_path.write_text(dump_preregistration(spec), encoding="utf-8")
        study.prereg_log.append(
            "register",
            {
                "document": spec.document(),
                "content_hash": spec.content_hash(),
                "previous_hash": None,
                "after_results": False,
            },
        )
        return study

    @classmethod
    def open(cls, root: str | Path) -> Study:
        """Open a study whose YAML working copy matches the log."""
        directory = Path(root)
        records = HashChainLog(directory / "prereg.jsonl").read_verified()
        if not records:
            raise PreRegistrationError("preregistration log is missing or empty")
        active = _active_document(records)
        spec = PreRegistration.from_mapping(active)
        yaml_path = directory / "prereg.yaml"
        if yaml_path.exists():
            loaded = load_preregistration(yaml_path)
            if loaded.content_hash() != spec.content_hash():
                raise PreRegistrationError("prereg.yaml does not match the append-only log")
        else:
            yaml_path.write_text(dump_preregistration(spec), encoding="utf-8")
        study = cls(directory, spec)
        study.trial_log.read_verified()
        study.holdout_log.read_verified()
        return study

    def amend(self, spec: PreRegistration) -> None:
        """Append a new declaration. After a seal or a trial this fails the receipt."""
        if spec.study_id != self.spec.study_id:
            raise PreRegistrationError("study_id cannot change")
        self.prereg_log.append(
            "amend",
            {
                "document": spec.document(),
                "content_hash": spec.content_hash(),
                "previous_hash": self.spec.content_hash(),
                "after_results": self.results_seen(),
            },
        )
        self.spec = spec
        self.prereg_path.write_text(dump_preregistration(spec), encoding="utf-8")

    def results_seen(self) -> bool:
        """True once a seal, a trial, or an unlock exists."""
        if self.trial_log.read_verified():
            return True
        return any(record["event"] in _RESULT_EVENTS for record in self.holdout_log.read_verified())

    def seal_holdout(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Split ``frame``, encrypt the holdout, and return only the in-sample rows.

        The embargo gap is discarded. It is not written into the in-sample frame
        and it is not part of the sealed holdout.
        """
        if self.seal_path.exists() or _events(self.holdout_log, "seal"):
            raise HoldoutError("holdout is already sealed")
        spec = self.spec.holdout
        in_sample, embargo, holdout = split_holdout(
            frame,
            time_column=spec.time_column,
            start=spec.start,
            embargo_periods=spec.embargo_periods,
        )
        metadata = {
            "prereg_hash": self.spec.content_hash(),
            "time_column": spec.time_column,
            "holdout_start": spec.start,
            "embargo_periods": spec.embargo_periods,
            "n_rows": int(len(holdout)),
            "columns": [str(column) for column in holdout.columns],
        }
        info = write_seal(
            holdout, seal_path=self.seal_path, key_path=self.key_path, metadata=metadata
        )
        self.holdout_log.append(
            "seal",
            {
                "seal_id": info.seal_id,
                "plaintext_sha256": info.plaintext_sha256,
                "prereg_hash": self.spec.content_hash(),
                "n_insample": int(len(in_sample)),
                "n_embargo": int(len(embargo)),
                "n_holdout": int(len(holdout)),
                "time_column": spec.time_column,
                "holdout_start": spec.start,
                "embargo_periods": spec.embargo_periods,
            },
        )
        return in_sample

    def record_trial(
        self,
        params: Mapping[str, Any],
        returns: Sequence[float] | np.ndarray | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one attempt. The holdout must already be sealed."""
        if not _events(self.holdout_log, "seal"):
            raise TrialError("seal the holdout before recording trials")
        if _events(self.holdout_log, "unlock") or _events(self.holdout_log, "unlock_repeat"):
            raise TrialError("trials cannot be recorded after the holdout has been unlocked")
        body = build_trial_body(
            self.spec, params, returns, metrics, prereg_hash=self.spec.content_hash()
        )
        return self.trial_log.append("trial", body)

    def track(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Record each call. The first argument is the parameter mapping.

        Return a return series, or a mapping with ``returns`` and optional ``metrics``.
        """

        @functools.wraps(fn)
        def wrapper(params: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            out = fn(params, *args, **kwargs)
            if isinstance(out, Mapping) and "returns" in out:
                extra = out.get("metrics")
                self.record_trial(
                    params,
                    returns=out["returns"],
                    metrics=extra if isinstance(extra, Mapping) else None,
                )
            else:
                self.record_trial(params, returns=out)
            return out

        return wrapper

    def optuna_callback(self, metric_name: str | None = None) -> Callable[..., None]:
        """Return an Optuna callback that records each completed trial.

        Optuna is not imported. Put the return series on
        ``trial.user_attrs["returns"]`` when you have it; a callback that only
        sees the objective value is still logged, and the receipt then fails
        the return-series gate.
        """
        name = metric_name or self.spec.primary_metric

        def _callback(optuna_study: Any, trial: Any) -> None:
            del optuna_study
            user = getattr(trial, "user_attrs", {}) or {}
            metrics: dict[str, Any] = {}
            extra = user.get("metrics")
            if isinstance(extra, Mapping):
                metrics.update(extra)
            if name not in metrics:
                if getattr(trial, "value", None) is None:
                    raise TrialError("optuna trial has no value to record")
                metrics[name] = trial.value
            self.record_trial(dict(trial.params), returns=user.get("returns"), metrics=metrics)

        return _callback

    def selected_trial(self) -> dict[str, Any]:
        """Parameters of the best recorded configuration. Higher primary metric wins."""
        summary = summarize_trials(self.spec, self.trial_log.read_verified())
        if summary.selected is None or summary.selected.primary is None:
            raise TrialError("no trial with a finite primary metric")
        return {"params": summary.selected.params, "params_hash": summary.selected.params_hash}

    def unlock(self, *, acknowledge_repeat: bool = False) -> pd.DataFrame:
        """Decrypt the holdout once.

        A second call raises :class:`HoldoutAlreadyUnlocked` after logging the
        refusal. ``acknowledge_repeat=True`` returns the rows anyway, logs
        ``unlock_repeat``, and fails a later receipt.
        """
        successes = _events(self.holdout_log, "unlock")
        if successes and not acknowledge_repeat:
            seal_id = successes[0]["body"].get("seal_id")
            self.holdout_log.append(
                "unlock_refused", {"seal_id": seal_id, "reason": "already_unlocked"}
            )
            raise HoldoutAlreadyUnlocked("holdout was already unlocked; a second look is refused")
        try:
            frame, metadata, info = read_seal(self.seal_path, self.key_path)
        except HoldoutKeyError:
            self.holdout_log.append("unlock_failed", {"reason": "key_rejected"})
            raise
        event = "unlock_repeat" if successes else "unlock"
        self.holdout_log.append(
            event,
            {
                "seal_id": info.seal_id,
                "plaintext_sha256": info.plaintext_sha256,
                "prereg_hash": metadata.get("prereg_hash"),
            },
        )
        return frame

    def record_holdout_result(
        self,
        returns: Sequence[float] | np.ndarray,
        params: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Store the out-of-sample returns for ``params`` after an unlock."""
        if not _events(self.holdout_log, "unlock") and not _events(
            self.holdout_log, "unlock_repeat"
        ):
            raise HoldoutError("unlock the holdout before recording an out-of-sample result")
        body = build_holdout_body(params, returns)
        return self.holdout_log.append("holdout_result", body)

    def evaluate_holdout(
        self, fn: Callable[[pd.DataFrame, Mapping[str, Any]], Sequence[float]]
    ) -> Sequence[float]:
        """Unlock once, call ``fn(holdout_frame, selected_params)``, and store its returns.

        If ``fn`` raises, the unlock has already been logged. A retry has to
        acknowledge the repeat, which fails the receipt.
        """
        selected = self.selected_trial()
        frame = self.unlock()
        returns = fn(frame, selected["params"])
        self.record_holdout_result(returns, params=selected["params"])
        return returns

    def issue_receipt(
        self,
        *,
        write: bool = True,
        json_path: str | Path | None = None,
        markdown_path: str | Path | None = None,
        generated_at: str | None = None,
    ) -> dict[str, Any]:
        """Build the receipt from the logs. Optionally write JSON and Markdown."""
        import json

        peeked = _peek(self.seal_path)
        receipt = build_receipt(
            self.spec,
            prereg_records=self.prereg_log.read_verified(),
            trial_records=self.trial_log.read_verified(),
            holdout_records=self.holdout_log.read_verified(),
            seal_present=peeked is not None,
            seal_id=None if peeked is None else peeked.seal_id,
            seal_plaintext_sha256=None if peeked is None else peeked.plaintext_sha256,
            generated_at=generated_at,
        )
        if write:
            encoded = json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            markdown = render_markdown(receipt)
            json_target = Path(json_path) if json_path is not None else self.root / "receipt.json"
            markdown_target = (
                Path(markdown_path) if markdown_path is not None else self.root / "receipt.md"
            )
            json_target.write_text(encoded, encoding="utf-8")
            markdown_target.write_text(markdown, encoding="utf-8")
        return receipt


def _active_document(records: list[dict[str, Any]]) -> dict[str, Any]:
    chosen: dict[str, Any] | None = None
    for record in records:
        if record["event"] in {"register", "amend"}:
            chosen = record["body"]["document"]
    if chosen is None:
        raise PreRegistrationError("preregistration log has no declaration")
    return chosen


def _events(log: HashChainLog, event: str) -> list[dict[str, Any]]:
    return [record for record in log.read_verified() if record["event"] == event]


def _peek(path: Path) -> Any:
    if not path.exists():
        return None
    return inspect_seal(path)
