# backtest-receipt

`backtest-receipt` is a research discipline layer for backtests. It does not
try to be another statistics toolkit. Purged and combinatorial cross-validation
come from [purgedcv](https://github.com/eslazarev/purged-cross-validation) (MIT).
The probabilistic Sharpe ratio (PSR) and the deflated Sharpe ratio (DSR) are
implemented in this repository from Bailey and López de Prado (2012, 2014).

The library is for the part those packages do not cover: write the claim down
before you see the result, lock the out-of-sample rows, count every
configuration you actually run, and produce a receipt a CI job can reject.

This is not a backtest engine, not a data client, and not a probability of
backtest overfitting (PBO) implementation. It will not stop a researcher who is
willing to rewrite a consistent receipt by hand. It will make that rewrite
visible when the study directory is what `check --study` recomputes.

License: MIT. Python 3.11+.

## Install

The package is not published on PyPI.

```bash
pip install "git+https://github.com/tupils1/backtest-receipt.git"
```

For development:

```bash
pip install -e ".[dev]"
ruff check src tests examples
pytest
```

## Five minutes

```bash
python examples/synthetic_study.py --mode overfit --out /tmp/overfit-study
backtest-receipt check /tmp/overfit-study/receipt.json
echo $?   # 1: a noise search does not pass

python examples/synthetic_study.py --mode edge --out /tmp/edge-study
backtest-receipt check /tmp/edge-study/receipt.json
echo $?   # 0: one pre-declared configuration on a drifted series
```

Both commands use an artificial price path. The crossover in the example lags
its signal by one bar. It is a demonstration of the protocol, not a strategy
result.

## Protocol

1. **Pre-register.** Hypothesis, universe, parameter grid, metrics, and pass
   criteria live in one YAML document or one Python object. The content hash is
   SHA-256 of canonical JSON, so key order does not matter. The prose is stored
   and hashed. It is not an input to the Sharpe math.
2. **Seal the holdout.** Rows at or after the declared time go into a Fernet
   ciphertext. The key is a separate file created with mode `0600` and is
   gitignored. An embargo of N rows immediately before the boundary is dropped
   from the in-sample set. You only get the in-sample frame back.
3. **Record every configuration** on that in-sample frame. Unique parameter
   sets are the trial count `N` that enters the deflated Sharpe ratio. The same
   parameters with a different return series are a conflict. A configuration
   outside the declared grid is stored anyway, and it fails the receipt.
4. **Unlock once** and record the out-of-sample returns of the selected
   configuration. A second unlock is refused and logged. Passing
   `acknowledge_repeat=True` returns the rows and fails the receipt.
5. **Issue a receipt** (`receipt.json` and `receipt.md`). `backtest-receipt check`
   exits 0 only when that artifact is internally consistent and says pass.

```python
from backtest_receipt import PreRegistration, Study

spec = PreRegistration.from_mapping(
    {
        "study_id": "demo",
        "hypothesis": "A lagged moving-average sign has a positive Sharpe on this series.",
        "universe": "one daily series",
        "parameters": {"fast": [10, 20], "slow": [40, 80]},
        "metrics": ["sharpe", "mean_return"],
        "primary_metric": "sharpe",
        "periods_per_year": 252,
        "notes": "Notes are audit text. They do not enter the estimator.",
        "holdout": {"time_column": "date", "start": "2024-01-01", "embargo_periods": 5},
        "pass_criteria": {
            "min_deflated_sharpe": 0.95,
            "min_oos_sharpe_annualized": 0.0,
            "min_oos_observations": 30,
        },
    }
)
study = Study.create(spec, "studies/demo")
in_sample = study.seal_holdout(frame)
study.record_trial({"fast": 10, "slow": 40}, returns=run(in_sample, fast=10, slow=40))
study.evaluate_holdout(lambda holdout, params: run(holdout, **params))
receipt = study.issue_receipt()
```

`sharpe` is the annualized Sharpe used only to rank trials and to apply the
out-of-sample floor. PSR and DSR are computed on the per-period Sharpe
(`mean / sample_std`). Annualizing those inputs would make the probability
meaningless. `periods_per_year` does not enter the probability.

Higher primary metric wins. If you care about a loss statistic, pass its
negation under a name you declared.

### Pass criteria

Omitted numeric hurdles receive a library default and are written into the
normalized document, which is what the hash covers. Explicit `null` turns a
numeric hurdle off.

| Field | Default | Meaning |
|---|---|---|
| `min_deflated_sharpe` | `0.95` | DSR hurdle. `null` disables it. |
| `min_probabilistic_sharpe` | off | PSR versus a zero benchmark. |
| `min_oos_sharpe_annualized` | `0.0` | Floor on the annualized out-of-sample Sharpe. |
| `max_sharpe_decay_annualized` | off | Maximum `in-sample minus out-of-sample` annualized Sharpe. |
| `min_oos_observations` | `30` | Fewer rows is `blocked`, not a statistical fail. |
| `require_single_holdout_unlock` | `true` | The holdout must be decrypted once and scored. |
| `reject_post_result_amendments` | `true` | Edits after a seal, trial, or unlock fail the study. |
| `reject_trials_outside_grid` | `true` | Undeclared configurations fail the study. |
| `reject_conflicting_reruns` | `true` | Same parameters, different returns. |
| `require_return_series` | `true` | Sharpe is recomputed from stored returns. |

`blocked` means the protocol is unfinished or the sample is too small to judge.
`fail` means it was judged and rejected, or an integrity rule fired. `check`
exits 1 for both. It exits 2 when the file is unreadable or disagrees with its
own gates and fingerprint (for example `passed` was flipped and nothing else
was updated).

```bash
backtest-receipt hash prereg.yaml
backtest-receipt issue studies/demo
backtest-receipt check studies/demo/receipt.json
backtest-receipt check studies/demo/receipt.json --study studies/demo
```

`check` alone confirms the artifact is internally consistent. `check --study`
recomputes the decision from the logs and the seal file. A fully rewritten but
self-consistent receipt still needs the study directory to be caught. Keep the
study in version control. The hash chain detects an edited line. It does not,
by itself, detect someone deleting the tail of a log and every copy of it.

Do not commit `holdout.key`. The seal file is ciphertext plus a plaintext hash.

### Cross-validation inside the in-sample window

```python
from backtest_receipt import make_combinatorial_purged_cv, make_purged_kfold

cv = make_purged_kfold(prediction_times, evaluation_times, n_splits=5, embargo="2D")
cpcv = make_combinatorial_purged_cv(
    prediction_times, evaluation_times, n_splits=6, n_test_groups=2
)
```

Folds are not trials. If you keep a configuration because of a cross-validation
score, record that configuration yourself. The splitters never open the holdout.

An Optuna callback is available without importing Optuna:

```python
def objective(trial):
    lookback = trial.suggest_int("lookback", 5, 40)
    returns = run(in_sample, lookback)
    trial.set_user_attr("returns", returns)
    return float(sharpe(returns))

optuna_study.optimize(objective, callbacks=[study.optuna_callback()])
```

Put the return series on `user_attrs["returns"]`. A callback that only forwards
the objective value is logged, and the receipt then fails the return-series gate.

## Estimators

Per period, with sample standard deviation in the Sharpe ratio and plug-in
central moments (divisor `n`) for skewness and non-excess kurtosis (a normal
distribution has kurtosis 3):

```text
PSR(SR*) = Φ( (SR - SR*) * sqrt(n - 1) / sqrt(1 - skew * SR + ((kurtosis - 1) / 4) * SR^2) )
```

DSR is that probability with `SR*` equal to the expected maximum Sharpe across
`N` independent trials:

```text
SR* = σ_trials * ( (1 - γ) Φ^{-1}(1 - 1/N) + γ Φ^{-1}(1 - 1/(N e)) )
```

`γ` is the Euler–Mascheroni constant. `σ_trials` is the sample standard
deviation of the per-period Sharpes across the unique recorded configurations.
One trial has benchmark 0, so DSR equals PSR. `N` is the count of unique
parameter sets. That treats them as independent, which over-penalizes clustered
searches. The library does not silently replace `N` with an effective-trial
estimate.

References:

- Bailey, D. H. and López de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk, 15(2), 3–44.
- Bailey, D. H. and López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality.* The Journal of Portfolio Management, 40(5), 94–107.

## Continuous integration

Copy [`examples/github/workflows/oos-gate.yml`](examples/github/workflows/oos-gate.yml)
into the repository that commits the receipt. This library's own CI
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff and pytest.
It does not publish the package, and it does not create a release.

## Limitations

- The threat model is an honest researcher, or an agent that would otherwise
  peek and keep tuning. It is not a control against someone who owns the files
  and is willing to rewrite both the log and a consistent receipt.
- A crash after a successful unlock and before `record_holdout_result` burns
  the one clean decrypt. Recovery uses `acknowledge_repeat=True` and fails the
  receipt. Score the holdout inside `evaluate_holdout` so both steps share one
  call.
- The sealed table is JSON inside the ciphertext. It is meant for research
  frames, not for a market-data warehouse.
- Datetime columns must be timezone-naive.
- Default hurdles are part of the normalized document. Two library versions
  with different defaults hash an incomplete YAML file differently. `Study.create`
  writes the resolved YAML; commit that file.

中文说明见 [README.zh-CN.md](README.zh-CN.md).
