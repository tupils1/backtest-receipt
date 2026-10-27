"""Build a pass/fail research receipt and render it as JSON or Markdown.

``check`` trusts a receipt only after its gates, status, and fingerprint agree.
That catches a hand-edit of the ``passed`` flag. It does not recompute the
statistics. Recomputing requires the study directory (``check --study``).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from backtest_receipt.canonical import content_hash
from backtest_receipt.errors import PreRegistrationError, ReceiptError
from backtest_receipt.prereg import PreRegistration
from backtest_receipt.stats import (
    annualized_sharpe,
    deflated_sharpe_ratio,
    estimate_sharpe,
    null_sharpe_dispersion,
)
from backtest_receipt.trials import TrialSummary, summarize_trials
from backtest_receipt.version import __version__

SCHEMA_VERSION = 1

_REFERENCES = [
    (
        "Bailey, D. H. and López de Prado, M. (2012). "
        "The Sharpe Ratio Efficient Frontier. Journal of Risk, 15(2), 3-44."
    ),
    (
        "Bailey, D. H. and López de Prado, M. (2014). "
        "The Deflated Sharpe Ratio: Correcting for Selection Bias, "
        "Backtest Overfitting and Non-Normality. "
        "The Journal of Portfolio Management, 40(5), 94-107."
    ),
]


def build_receipt(
    spec: PreRegistration,
    *,
    prereg_records: list[dict[str, Any]],
    trial_records: list[dict[str, Any]],
    holdout_records: list[dict[str, Any]],
    seal_present: bool,
    seal_id: str | None,
    seal_plaintext_sha256: str | None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a receipt from already-verified log records."""
    registration = _registration_view(prereg_records)
    trials = summarize_trials(spec, trial_records)
    holdout = _holdout_view(holdout_records)
    gates: list[dict[str, Any]] = []
    registration["public"]["working_copy_matches"] = (
        registration["active_hash"] == spec.content_hash()
    )
    gates.extend(
        _integrity_gates(
            spec, registration, trials, holdout, seal_present, seal_id, seal_plaintext_sha256
        )
    )
    statistics, scientific = _scientific_gates(spec, trials, holdout)
    gates.extend(scientific)
    passed, status = status_from_gates(gates)
    when = generated_at if generated_at is not None else datetime.now(UTC).isoformat()
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "library_version": __version__,
        "generated_at": when,
        "passed": passed,
        "status": status,
        "study_id": spec.study_id,
        "hypothesis": spec.hypothesis,
        "notes": spec.notes,
        "notes_are_not_inputs": True,
        "preregistration": registration["public"],
        "pass_criteria": spec.pass_criteria.as_dict(),
        "trials": _trials_public(trials),
        "statistics": statistics,
        "holdout": holdout["public"],
        "gates": gates,
        "references": list(_REFERENCES),
    }
    receipt["fingerprint"] = decision_fingerprint(receipt)
    return receipt


def decision_fingerprint(receipt: dict[str, Any]) -> str:
    """Hash the decision fields. Timestamps and the fingerprint itself are excluded."""
    payload = {
        "schema_version": receipt["schema_version"],
        "passed": receipt["passed"],
        "status": receipt["status"],
        "study_id": receipt["study_id"],
        "preregistration": {
            "content_hash": receipt["preregistration"]["content_hash"],
            "original_hash": receipt["preregistration"]["original_hash"],
            "post_result_amendments": receipt["preregistration"]["post_result_amendments"],
            "working_copy_matches": receipt["preregistration"]["working_copy_matches"],
        },
        "trials": {
            "unique_configs": receipt["trials"]["unique_configs"],
            "attempts": receipt["trials"]["attempts"],
            "outside_grid": receipt["trials"]["outside_grid"],
            "conflicting_reruns": receipt["trials"]["conflicting_reruns"],
            "selected_params_hash": receipt["trials"]["selected_params_hash"],
        },
        "statistics": {
            key: receipt["statistics"][key]
            for key in (
                "n_obs",
                "n_trials",
                "sharpe_per_period",
                "sharpe_annualized",
                "probabilistic_sharpe",
                "deflated_sharpe",
                "sharpe_benchmark",
                "trial_sharpe_std",
            )
        },
        "holdout": {
            key: receipt["holdout"][key]
            for key in (
                "unlocks",
                "repeat_unlocks",
                "oos_n",
                "oos_sharpe_per_period",
                "oos_sharpe_annualized",
            )
        },
        "gates": [
            {
                "name": gate["name"],
                "state": gate["state"],
                "actual": gate["actual"],
                "threshold": gate["threshold"],
            }
            for gate in receipt["gates"]
        ],
    }
    return content_hash(payload)


def status_from_gates(gates: list[dict[str, Any]]) -> tuple[bool, str]:
    """Collapse gate states. An integrity failure outranks an unfinished study."""
    if any(gate["integrity"] and gate["state"] == "fail" for gate in gates):
        return False, "fail"
    if any(gate["state"] == "blocked" for gate in gates):
        return False, "blocked"
    if any(gate["state"] == "fail" for gate in gates):
        return False, "fail"
    return True, "pass"


def validate_receipt(document: dict[str, Any]) -> None:
    """Raise if a receipt disagrees with its own gates or fingerprint."""
    if not isinstance(document, dict):
        raise ReceiptError("receipt must be a JSON object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ReceiptError("unsupported receipt schema_version")
    for key in (
        "passed",
        "status",
        "gates",
        "fingerprint",
        "preregistration",
        "trials",
        "statistics",
        "holdout",
    ):
        if key not in document:
            raise ReceiptError(f"receipt is missing {key}")
    try:
        passed, status = status_from_gates(document["gates"])
        fingerprint = decision_fingerprint(document)
    except (KeyError, TypeError) as exc:
        raise ReceiptError("receipt gates are malformed") from exc
    if document["passed"] is not passed or document["status"] != status:
        raise ReceiptError("receipt status does not match its gates")
    if document["fingerprint"] != fingerprint:
        raise ReceiptError("receipt fingerprint does not match its decision block")


def render_markdown(receipt: dict[str, Any]) -> str:
    """Human-readable receipt. The JSON file is the artifact ``check`` reads."""
    status = str(receipt["status"]).upper()
    lines = [
        f"# Research receipt: {receipt['study_id']}",
        "",
        f"**{status}**",
        "",
        receipt["hypothesis"],
        "",
        "## Pre-registration",
        "",
        f"- Content hash: `{receipt['preregistration']['content_hash']}`",
        f"- Original hash: `{receipt['preregistration']['original_hash']}`",
        f"- Amendments: {receipt['preregistration']['amendments']}",
        f"- Amendments after results: {receipt['preregistration']['post_result_amendments']}",
        f"- Working copy matches log: {receipt['preregistration']['working_copy_matches']}",
        "",
        "Notes are stored for the audit trail and are not inputs to the decision.",
        "",
        "## Trials",
        "",
        f"- Unique configurations: {receipt['trials']['unique_configs']}",
        f"- Attempts: {receipt['trials']['attempts']}",
        f"- Outside the declared grid: {receipt['trials']['outside_grid']}",
        f"- Conflicting reruns: {receipt['trials']['conflicting_reruns']}",
        f"- Selected configuration hash: `{receipt['trials']['selected_params_hash']}`",
        "",
        "## Statistics",
        "",
        _fmt("In-sample observations", receipt["statistics"]["n_obs"]),
        _fmt("Per-period Sharpe", receipt["statistics"]["sharpe_per_period"]),
        _fmt("Annualized Sharpe", receipt["statistics"]["sharpe_annualized"]),
        _fmt("Probabilistic Sharpe (benchmark 0)", receipt["statistics"]["probabilistic_sharpe"]),
        _fmt("Deflated Sharpe", receipt["statistics"]["deflated_sharpe"]),
        _fmt("Sharpe benchmark", receipt["statistics"]["sharpe_benchmark"]),
        _fmt("Trial-Sharpe dispersion", receipt["statistics"]["trial_sharpe_std"]),
        f"- Dispersion source: {receipt['statistics']['dispersion_source']}",
        f"- Trials entering the deflator: {receipt['statistics']['n_trials']}",
        "",
        "PSR and DSR use per-period Sharpe ratios. Annualized numbers are display-only.",
        "",
        "## Holdout",
        "",
        f"- Unlocks: {receipt['holdout']['unlocks']}",
        f"- Flagged repeat unlocks: {receipt['holdout']['repeat_unlocks']}",
        f"- Refused unlocks: {receipt['holdout']['refused_unlocks']}",
        _fmt("Out-of-sample observations", receipt["holdout"]["oos_n"]),
        _fmt("Out-of-sample per-period Sharpe", receipt["holdout"]["oos_sharpe_per_period"]),
        _fmt("Out-of-sample annualized Sharpe", receipt["holdout"]["oos_sharpe_annualized"]),
        "",
        "## Gates",
        "",
    ]
    for gate in receipt["gates"]:
        lines.append(
            f"- `{gate['name']}` **{gate['state']}**: "
            f"actual `{gate['actual']}`, threshold `{gate['threshold']}`. {gate['detail']}"
        )
    lines.extend(["", "## References", ""])
    for reference in receipt["references"]:
        lines.append(f"- {reference}")
    lines.append("")
    return "\n".join(lines)


def _fmt(label: str, value: Any) -> str:
    if isinstance(value, float):
        return f"- {label}: {value:.6g}"
    return f"- {label}: {value}"


def _registration_view(records: list[dict[str, Any]]) -> dict[str, Any]:
    events = [record for record in records if record["event"] in {"register", "amend"}]
    if not events or events[0]["event"] != "register":
        raise PreRegistrationError("preregistration log must start with a register event")
    for record in events:
        body = record["body"]
        if content_hash(body["document"]) != body["content_hash"]:
            raise PreRegistrationError("stored content hash does not match the logged document")
    active = events[-1]["body"]
    amendments = [record for record in events[1:]]
    post_result = sum(1 for record in amendments if record["body"].get("after_results") is True)
    public = {
        "content_hash": active["content_hash"],
        "original_hash": events[0]["body"]["content_hash"],
        "amendments": len(amendments),
        "post_result_amendments": post_result,
        "working_copy_matches": True,
        "history": [
            {
                "event": record["event"],
                "recorded_at": record["recorded_at"],
                "content_hash": record["body"]["content_hash"],
                "previous_hash": record["body"].get("previous_hash"),
                "after_results": bool(record["body"].get("after_results")),
            }
            for record in events
        ],
    }
    return {"public": public, "post_result": post_result, "active_hash": active["content_hash"]}


def _holdout_view(records: list[dict[str, Any]]) -> dict[str, Any]:
    seals = [record for record in records if record["event"] == "seal"]
    unlocks = [record for record in records if record["event"] == "unlock"]
    repeats = [record for record in records if record["event"] == "unlock_repeat"]
    refused = [record for record in records if record["event"] == "unlock_refused"]
    failed = [record for record in records if record["event"] == "unlock_failed"]
    results = [record for record in records if record["event"] == "holdout_result"]
    oos = _oos_moments(results[0]["body"].get("returns")) if len(results) == 1 else None
    public = {
        "sealed": bool(seals),
        "seal_id": None if not seals else seals[-1]["body"].get("seal_id"),
        "plaintext_sha256": None if not seals else seals[-1]["body"].get("plaintext_sha256"),
        "prereg_hash": None if not seals else seals[-1]["body"].get("prereg_hash"),
        "n_insample": None if not seals else seals[-1]["body"].get("n_insample"),
        "n_embargo": None if not seals else seals[-1]["body"].get("n_embargo"),
        "n_holdout_rows": None if not seals else seals[-1]["body"].get("n_holdout"),
        "unlocks": len(unlocks),
        "repeat_unlocks": len(repeats),
        "refused_unlocks": len(refused),
        "failed_unlocks": len(failed),
        "results": len(results),
        "oos_n": None if oos is None else oos["n_obs"],
        "oos_sharpe_per_period": None if oos is None else oos["sharpe_per_period"],
        "oos_sharpe_annualized": None,
        "oos_params_hash": None if len(results) != 1 else results[0]["body"].get("params_hash"),
        "events": [
            {
                "event": record["event"],
                "recorded_at": record["recorded_at"],
                "seal_id": record["body"].get("seal_id"),
            }
            for record in records
            if record["event"]
            in {"seal", "unlock", "unlock_repeat", "unlock_refused", "unlock_failed"}
        ],
    }
    return {
        "public": public,
        "seals": len(seals),
        "unlocks": len(unlocks),
        "repeats": len(repeats),
        "results": len(results),
        "oos": oos,
    }


def _oos_moments(returns: Any) -> dict[str, Any] | None:
    if not isinstance(returns, list):
        return None
    moments = estimate_sharpe([float(item) for item in returns])
    per_period = moments.sharpe_per_period
    return {
        "n_obs": moments.n_obs,
        "sharpe_per_period": per_period if math.isfinite(per_period) else None,
    }


def _integrity_gates(
    spec: PreRegistration,
    registration: dict[str, Any],
    trials: TrialSummary,
    holdout: dict[str, Any],
    seal_present: bool,
    seal_id: str | None,
    seal_plaintext_sha256: str | None,
) -> list[dict[str, Any]]:
    criteria = spec.pass_criteria
    public = holdout["public"]
    gates = [
        _gate(
            "working_copy",
            state="pass" if registration["active_hash"] == spec.content_hash() else "fail",
            actual=spec.content_hash(),
            threshold=registration["active_hash"],
            detail="The YAML working copy matches the latest logged declaration.",
            integrity=True,
        )
    ]
    if criteria.reject_post_result_amendments:
        count = registration["post_result"]
        gates.append(
            _gate(
                "post_result_amendments",
                state="pass" if count == 0 else "fail",
                actual=count,
                threshold=0,
                detail="Declarations changed after a seal, a trial, or an unlock fail closed.",
                integrity=True,
            )
        )
    if criteria.reject_trials_outside_grid:
        gates.append(
            _gate(
                "trials_outside_grid",
                state="pass" if trials.outside_grid == 0 else "fail",
                actual=trials.outside_grid,
                threshold=0,
                detail="Every recorded configuration stays inside the declared search space.",
                integrity=True,
            )
        )
    if criteria.reject_conflicting_reruns:
        gates.append(
            _gate(
                "conflicting_reruns",
                state="pass" if trials.conflicting_reruns == 0 else "fail",
                actual=trials.conflicting_reruns,
                threshold=0,
                detail="The same configuration produced more than one return series.",
                integrity=True,
            )
        )
    if trials.attempts == 0:
        gates.append(
            _gate(
                "trials_recorded",
                state="blocked",
                actual=0,
                threshold=1,
                detail="No configurations have been recorded yet.",
            )
        )
    elif criteria.require_return_series:
        gates.append(
            _gate(
                "return_series",
                state="pass" if trials.missing_returns == 0 else "fail",
                actual=trials.missing_returns,
                threshold=0,
                detail="Each configuration stores the returns the Sharpe ratio is computed from.",
                integrity=True,
            )
        )
    gates.append(
        _gate(
            "trial_declaration",
            state="pass" if trials.foreign_prereg == 0 else "fail",
            actual=trials.foreign_prereg,
            threshold=0,
            detail="Every trial was recorded under the active declaration hash.",
            integrity=True,
        )
    )
    seal_state, seal_detail = _seal_consistent(
        public, seal_present, seal_id, seal_plaintext_sha256, holdout["seals"]
    )
    gates.append(
        _gate(
            "seal",
            state=seal_state,
            actual=seal_present,
            threshold=True,
            detail=seal_detail,
            integrity=seal_state == "fail",
        )
    )
    if holdout["repeats"]:
        gates.append(
            _gate(
                "repeat_unlock",
                state="fail",
                actual=holdout["repeats"],
                threshold=0,
                detail="The holdout was decrypted more than once and the repeat fails the study.",
                integrity=True,
            )
        )
    if holdout["results"] > 1:
        gates.append(
            _gate(
                "holdout_results",
                state="fail",
                actual=holdout["results"],
                threshold=1,
                detail="More than one out-of-sample result was recorded.",
                integrity=True,
            )
        )
    selected_hash = None if trials.selected is None else trials.selected.params_hash
    oos_hash = public["oos_params_hash"]
    if holdout["results"] == 1 and oos_hash != selected_hash:
        gates.append(
            _gate(
                "holdout_params",
                state="fail",
                actual=oos_hash,
                threshold=selected_hash,
                detail="Out-of-sample returns do not match the selected trial.",
                integrity=True,
            )
        )
    return gates


def _seal_consistent(
    public: dict[str, Any],
    seal_present: bool,
    seal_id: str | None,
    seal_plaintext_sha256: str | None,
    n_seals: int,
) -> tuple[str, str]:
    if n_seals == 0:
        return "blocked", "The holdout has not been sealed."
    if n_seals != 1:
        return "fail", "The study has more than one seal event."
    if not seal_present:
        return "fail", "The seal file is missing."
    if seal_id != public["seal_id"] or seal_plaintext_sha256 != public["plaintext_sha256"]:
        return "fail", "The seal file does not match the seal event."
    return "pass", "The seal file matches the single seal event."


def _scientific_gates(
    spec: PreRegistration,
    trials: TrialSummary,
    holdout: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    criteria = spec.pass_criteria
    selected = trials.selected
    gates: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "n_obs": None,
        "n_trials": len(trials.unique),
        "sharpe_per_period": None,
        "sharpe_annualized": None,
        "skewness": None,
        "kurtosis": None,
        "probabilistic_sharpe": None,
        "deflated_sharpe": None,
        "sharpe_benchmark": None,
        "trial_sharpe_std": trials.trial_sharpe_std,
        "dispersion_source": trials.dispersion_source,
        "sr_units": "per_period",
        "kurtosis_definition": "non_excess_plug_in",
    }
    if selected is None or selected.returns is None or selected.sharpe_per_period is None:
        gates.append(
            _gate(
                "in_sample_sharpe",
                state="blocked",
                actual=None,
                threshold=None,
                detail="No selected in-sample return series with a finite Sharpe ratio.",
            )
        )
    else:
        dispersion = trials.trial_sharpe_std
        source = trials.dispersion_source
        if (
            dispersion is None
            and len(trials.unique) > 1
            and selected.n_obs is not None
            and selected.n_obs >= 2
        ):
            dispersion = null_sharpe_dispersion(selected.n_obs)
            source = "null_fallback"
        try:
            result = deflated_sharpe_ratio(
                selected.sharpe_per_period,
                n_obs=selected.n_obs or 0,
                n_trials=max(1, len(trials.unique)),
                skewness=0.0 if selected.skewness is None else selected.skewness,
                kurtosis=3.0 if selected.kurtosis is None else selected.kurtosis,
                trial_sharpe_std=0.0 if dispersion is None else dispersion,
            )
        except ValueError:
            gates.append(
                _gate(
                    "deflated_sharpe",
                    state="blocked",
                    actual=None,
                    threshold=criteria.min_deflated_sharpe,
                    detail="The deflated Sharpe estimator rejected these moments.",
                )
            )
            _holdout_gates(spec, statistics, holdout, gates)
            return statistics, [gate for gate in gates if gate is not None]
        annualized = annualized_sharpe(selected.sharpe_per_period, spec.periods_per_year)
        statistics.update(
            {
                "n_obs": selected.n_obs,
                "sharpe_per_period": selected.sharpe_per_period,
                "sharpe_annualized": annualized,
                "skewness": selected.skewness,
                "kurtosis": selected.kurtosis,
                "probabilistic_sharpe": _number(result.probabilistic_sharpe),
                "deflated_sharpe": _number(result.deflated_sharpe),
                "sharpe_benchmark": result.sharpe_benchmark,
                "trial_sharpe_std": result.trial_sharpe_std,
                "dispersion_source": source,
            }
        )
        gates.append(
            _probability_gate(
                "deflated_sharpe",
                actual=result.deflated_sharpe,
                threshold=criteria.min_deflated_sharpe,
                detail="Probability the true Sharpe exceeds the expected maximum of the trials.",
            )
        )
        gates.append(
            _probability_gate(
                "probabilistic_sharpe",
                actual=result.probabilistic_sharpe,
                threshold=criteria.min_probabilistic_sharpe,
                detail="Probability the true Sharpe exceeds zero, before deflation.",
            )
        )
    _holdout_gates(spec, statistics, holdout, gates)
    return statistics, [gate for gate in gates if gate is not None]


def _holdout_gates(
    spec: PreRegistration,
    statistics: dict[str, Any],
    holdout: dict[str, Any],
    gates: list[dict[str, Any]],
) -> None:
    criteria = spec.pass_criteria
    public = holdout["public"]
    if criteria.require_single_holdout_unlock:
        unlocks = holdout["unlocks"]
        if unlocks == 1 and holdout["repeats"] == 0 and holdout["results"] == 1:
            state = "pass"
            detail = "The holdout was decrypted once and an out-of-sample result was recorded."
        elif unlocks == 0 and holdout["repeats"] == 0:
            state = "blocked"
            detail = "The holdout has not been decrypted, so there is no out-of-sample result."
        elif holdout["repeats"]:
            state = "fail"
            detail = "A repeat unlock already failed the study."
        else:
            state = "blocked"
            detail = "The holdout unlock count is not a single completed evaluation."
        gates.append(
            _gate(
                "holdout_unlock",
                state=state,
                actual=unlocks,
                threshold=1,
                detail=detail,
                integrity=False,
            )
        )
    oos = holdout["oos"]
    if oos is None:
        if (
            criteria.min_oos_observations is not None
            or criteria.min_oos_sharpe_annualized is not None
        ):
            gates.append(
                _gate(
                    "oos_sample",
                    state="blocked",
                    actual=None,
                    threshold=criteria.min_oos_observations,
                    detail="Out-of-sample returns were not recorded.",
                )
            )
        return
    annualized = None
    if oos["sharpe_per_period"] is not None:
        annualized = annualized_sharpe(oos["sharpe_per_period"], spec.periods_per_year)
    public["oos_sharpe_annualized"] = annualized
    oos["sharpe_annualized"] = annualized
    if criteria.min_oos_observations is not None:
        enough = oos["n_obs"] >= criteria.min_oos_observations
        gates.append(
            _gate(
                "oos_sample",
                state="pass" if enough else "blocked",
                actual=oos["n_obs"],
                threshold=criteria.min_oos_observations,
                detail="Too few out-of-sample rows is incomplete, not a statistical rejection.",
            )
        )
    if criteria.min_oos_sharpe_annualized is not None:
        if annualized is None or (
            criteria.min_oos_observations is not None
            and oos["n_obs"] < criteria.min_oos_observations
        ):
            gates.append(
                _gate(
                    "oos_sharpe",
                    state="blocked",
                    actual=annualized,
                    threshold=criteria.min_oos_sharpe_annualized,
                    detail="Out-of-sample Sharpe is undefined or the sample is too small.",
                )
            )
        else:
            gates.append(
                _gate(
                    "oos_sharpe",
                    state="pass" if annualized >= criteria.min_oos_sharpe_annualized else "fail",
                    actual=annualized,
                    threshold=criteria.min_oos_sharpe_annualized,
                    detail="Annualized out-of-sample Sharpe versus the declared floor.",
                )
            )
    if criteria.max_sharpe_decay_annualized is not None:
        in_sample = statistics.get("sharpe_annualized")
        if in_sample is None or annualized is None:
            gates.append(
                _gate(
                    "sharpe_decay",
                    state="blocked",
                    actual=None,
                    threshold=criteria.max_sharpe_decay_annualized,
                    detail="Decay needs both an in-sample and an out-of-sample Sharpe.",
                )
            )
        else:
            decay = float(in_sample) - float(annualized)
            gates.append(
                _gate(
                    "sharpe_decay",
                    state="pass" if decay <= criteria.max_sharpe_decay_annualized else "fail",
                    actual=decay,
                    threshold=criteria.max_sharpe_decay_annualized,
                    detail="In-sample annualized Sharpe minus out-of-sample annualized Sharpe.",
                )
            )


def _probability_gate(
    name: str, *, actual: float, threshold: float | None, detail: str
) -> dict[str, Any] | None:
    if threshold is None:
        return None
    if not math.isfinite(actual):
        return _gate(
            name,
            state="blocked",
            actual=None,
            threshold=threshold,
            detail=detail + " The estimator was undefined.",
        )
    state = "pass" if actual >= threshold else "fail"
    return _gate(name, state=state, actual=actual, threshold=threshold, detail=detail)


def _gate(
    name: str,
    *,
    state: str,
    actual: Any,
    threshold: Any,
    detail: str,
    integrity: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "state": state,
        "actual": _number(actual),
        "threshold": _number(threshold),
        "detail": detail,
        "integrity": integrity,
    }


def _number(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        number = float(value) if isinstance(value, float) else value
        if isinstance(number, float) and not math.isfinite(number):
            return None
        return number
    return value


def _trials_public(trials: TrialSummary) -> dict[str, Any]:
    selected = trials.selected
    return {
        "unique_configs": len(trials.unique),
        "attempts": trials.attempts,
        "outside_grid": trials.outside_grid,
        "conflicting_reruns": trials.conflicting_reruns,
        "missing_returns": trials.missing_returns,
        "selected_params_hash": None if selected is None else selected.params_hash,
        "selected_params": None if selected is None else selected.params,
        "selected_metrics": None if selected is None else selected.metrics,
        "configurations": [
            {
                "params_hash": trial.params_hash,
                "params": trial.params,
                "primary": trial.primary,
                "sharpe_per_period": trial.sharpe_per_period,
                "outside_grid": trial.outside_grid,
                "n_obs": trial.n_obs,
            }
            for trial in trials.unique
        ],
    }
