"""Content hashes and declaration validation."""

from __future__ import annotations

import pytest

from backtest_receipt import PreRegistration, PreRegistrationError, content_hash
from backtest_receipt.prereg import dump_preregistration, load_preregistration, params_within_grid
from tests.conftest import declaration


def test_hash_ignores_key_order_and_rejects_non_finite() -> None:
    assert content_hash({"b": 1, "a": [1, True]}) == content_hash({"a": [1, True], "b": 1})
    with pytest.raises(ValueError):
        content_hash({"x": float("nan")})
    with pytest.raises(TypeError):
        content_hash({"x": object()})


def test_yaml_roundtrip_preserves_hash(tmp_path) -> None:
    spec = declaration()
    path = tmp_path / "prereg.yaml"
    path.write_text(dump_preregistration(spec), encoding="utf-8")
    loaded = load_preregistration(path)
    assert loaded.content_hash() == spec.content_hash()
    assert loaded.hypothesis == spec.hypothesis
    assert loaded.pass_criteria.min_probabilistic_sharpe is None
    assert loaded.pass_criteria.min_deflated_sharpe == pytest.approx(0.5)


def test_unquoted_yaml_date_matches_string_boundary(tmp_path) -> None:
    spec = declaration()
    text = dump_preregistration(spec).replace("start: '2024-06-03'", "start: 2024-06-03")
    text = text.replace('start: "2024-06-03"', "start: 2024-06-03")
    path = tmp_path / "prereg.yaml"
    path.write_text(text, encoding="utf-8")
    loaded = load_preregistration(path)
    assert loaded.holdout.start == "2024-06-03"
    assert loaded.content_hash() == spec.content_hash()


def test_explicit_null_disables_a_default_hurdle() -> None:
    spec = declaration(
        pass_criteria={"min_deflated_sharpe": None, "min_oos_sharpe_annualized": None}
    )
    assert spec.pass_criteria.min_deflated_sharpe is None
    assert spec.pass_criteria.min_oos_sharpe_annualized is None
    assert spec.pass_criteria.reject_post_result_amendments is True


def test_grid_membership() -> None:
    space = declaration().parameters
    assert params_within_grid({"lookback": 10}, space)
    assert params_within_grid({"lookback": 10.0}, space)
    assert not params_within_grid({"lookback": 7}, space)
    assert not params_within_grid({"lookback": 10, "extra": 1}, space)
    assert not params_within_grid({}, space)
    ranged = declaration(parameters={"band": {"min": 0.0, "max": 1.0}}).parameters
    assert params_within_grid({"band": 0.0}, ranged)
    assert params_within_grid({"band": 1}, ranged)
    assert not params_within_grid({"band": 1.1}, ranged)
    assert not params_within_grid({"band": True}, ranged)


@pytest.mark.parametrize(
    "raw",
    [
        {"study_id": "bad id"},
        {"hypothesis": "too short"},
        {"universe": "ok", "mystery": 1},
        {"parameters": {}},
        {"parameters": {"lookback": []}},
        {"parameters": {"lookback": {"min": 2}}},
        {"parameters": {"lookback": {"min": 3, "max": 1}}},
        {"metrics": ["sharpe"], "primary_metric": "other"},
        {"holdout": {"time_column": "date"}},
        {"periods_per_year": 0},
        {"pass_criteria": {"min_deflated_sharpe": 1.5}},
        {"pass_criteria": {"require_return_series": "yes"}},
    ],
)
def test_invalid_declarations(raw: dict) -> None:
    base = {
        "study_id": "sample-study",
        "hypothesis": "The synthetic series has a positive mean after costs of zero.",
        "universe": "synthetic daily series",
        "parameters": {"lookback": [5]},
        "metrics": ["sharpe"],
        "primary_metric": "sharpe",
        "holdout": {"time_column": "date", "start": "2024-06-03", "embargo_periods": 0},
    }
    base.update(raw)
    with pytest.raises(PreRegistrationError):
        PreRegistration.from_mapping(base)


def test_universe_mapping_and_duplicate_metrics() -> None:
    spec = declaration(universe={"frequency": "daily", "rows": 100})
    assert spec.universe["frequency"] == "daily"
    with pytest.raises(PreRegistrationError):
        declaration(metrics=["sharpe", "sharpe"])
    with pytest.raises(PreRegistrationError):
        declaration(universe="x")
    with pytest.raises(PreRegistrationError):
        declaration(notes=1)  # type: ignore[arg-type]
