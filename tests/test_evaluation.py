"""Tests for probabilistic evaluation metrics and the backtest harness."""
import numpy as np
import pandas as pd
import pytest

from tsforecast.evaluation import (
    crps_gaussian, interval_coverage, interval_width, mae, mape, rmse,
    rolling_origin_backtest,
)


def test_crps_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    crps = crps_gaussian(y, y, np.full(3, 1e-9))
    assert (crps >= 0).all()
    assert crps.max() < 1e-6


def test_crps_worse_forecast_is_larger():
    y = np.array([0.0])
    good = crps_gaussian(y, np.array([0.1]), np.array([1.0]))
    bad = crps_gaussian(y, np.array([5.0]), np.array([1.0]))
    assert bad[0] > good[0]


def test_crps_reduces_to_mae_for_degenerate_distribution():
    # As sigma -> 0, CRPS(F, y) -> |y - mu| (point mass limit).
    y = np.array([1.0, 2.0, -1.0])
    mu = np.array([1.5, 1.0, -0.5])
    crps = crps_gaussian(y, mu, np.full(3, 1e-9))
    assert np.allclose(crps, np.abs(y - mu), atol=1e-6)


def test_interval_coverage_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert interval_coverage(y, np.array([0, 0, 0, 0]), np.array([2.5, 2.5, 2.5, 2.5])) == 0.5
    assert interval_width(np.array([0.0]), np.array([2.0])) == 2.0


def test_point_metrics():
    y = np.array([1.0, 2.0, 3.0])
    mu = np.array([1.0, 3.0, 3.0])
    assert mae(y, mu) == pytest.approx(1 / 3)
    assert rmse(y, mu) == pytest.approx(np.sqrt(1 / 3))
    assert mape(np.array([100.0, 200.0]), np.array([110.0, 180.0])) == pytest.approx(10.0)


def test_backtest_has_no_lookahead():
    """A probe model records the training data it receives; a spike placed
    AFTER the origin must never leak into any fit."""
    seen = []

    def probe(train: pd.Series, horizon: int):
        seen.append(train.copy())
        return np.zeros(horizon), np.ones(horizon)

    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    y = pd.Series(np.random.default_rng(0).normal(0, 1, 200), index=idx)
    y.iloc[150:] += 1000.0  # massive regime shift AFTER all origins
    origins = list(pd.date_range("2020-05-01", periods=5, freq="7D"))
    out = rolling_origin_backtest(y, origins, horizon=7, model_fns={"probe": probe})
    assert len(out) == 5
    for train, origin in zip(seen, origins):
        assert train.index.max() == origin, "model saw data past the origin!"
        assert (train.values < 100).all(), "future spike leaked into training data"


def test_backtest_output_schema():
    def flat(train: pd.Series, horizon: int):
        m = float(train.mean())
        return np.full(horizon, m), np.full(horizon, 1.0)

    idx = pd.date_range("2021-01-01", periods=120, freq="D")
    y = pd.Series(np.sin(np.arange(120) / 7 * 2 * np.pi) + 10, index=idx)
    origins = list(pd.date_range("2021-04-01", periods=3, freq="7D"))
    out = rolling_origin_backtest(y, origins, horizon=7, model_fns={"flat": flat})
    expected = {"origin", "model", "mae", "rmse", "mape", "crps",
                "cov80", "cov95", "width80", "width95", "h"}
    assert expected.issubset(out.columns)
    assert (out["cov80"] >= 0).all() and (out["cov80"] <= 1).all()
