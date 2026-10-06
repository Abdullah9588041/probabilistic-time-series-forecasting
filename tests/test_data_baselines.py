"""Tests for data loading and baseline models."""
import numpy as np
import pandas as pd

from tsforecast import data
from tsforecast.baselines import gbm_lag, seasonal_naive


def test_load_daily_shape_and_quality():
    y = data.load_daily()
    assert isinstance(y, pd.Series)
    assert len(y) > 1000, f"expected >1000 days, got {len(y)}"
    assert y.isna().sum() == 0
    assert (y > 0).all(), "daily energy must be positive"
    assert y.index.freq == "D" or (y.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    info = data.describe(y)
    assert info["start"] < info["end"]


def test_seasonal_naive_repeats_last_week():
    train = pd.Series(np.arange(30, dtype=float))
    mean, sd = seasonal_naive(train, horizon=10, s=7)
    assert len(mean) == 10 and len(sd) == 10
    # last 7 values are 23..29; horizon 10 -> [23..29, 23, 24, 25]
    assert list(mean[:7]) == [23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0]
    assert list(mean[7:]) == [23.0, 24.0, 25.0]
    assert (sd > 0).all()


def test_gbm_lag_runs_on_small_series():
    rng = np.random.default_rng(0)
    t = np.arange(300)
    y = pd.Series(10 + 2 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, 300))
    mean, sd = gbm_lag(y, horizon=7, seed=0)
    assert mean.shape == (7,) and sd.shape == (7,)
    assert np.isfinite(mean).all() and (sd > 0).all()
