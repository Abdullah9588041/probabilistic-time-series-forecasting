"""Baseline forecasting models, each exposing ``fit_forecast(train, horizon)``.

Every function returns ``(mean, sd)`` — a point forecast and a predictive
standard deviation per horizon step — so all models can be compared on
probabilistic metrics (CRPS, interval coverage), not just point accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASONAL_PERIOD = 7


def seasonal_naive(train: pd.Series, horizon: int, s: int = SEASONAL_PERIOD):
    """Repeat the last ``s`` observations; sd from seasonal-difference residuals."""
    hist = train.values[-s:]
    mean = np.tile(hist, int(np.ceil(horizon / s)))[:horizon]
    resid = train.values[s:] - train.values[:-s]
    sd = np.full(horizon, max(float(np.std(resid, ddof=1)), 1e-6))
    return mean, sd


def sarima(train: pd.Series, horizon: int, s: int = SEASONAL_PERIOD, max_train: int = 365):
    """Seasonal ARIMA(2,1,2)x(1,1,1,s); refit on the last ``max_train`` obs for speed."""
    sub = train.iloc[-max_train:]
    model = SARIMAX(
        sub.values, order=(2, 1, 2), seasonal_order=(1, 1, 1, s),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = np.asarray(fc.predicted_mean)
    sd = np.asarray(fc.se_mean)
    sd = np.maximum(sd, 1e-6)
    return mean, sd


def _lag_features(values: np.ndarray, lags: range = range(1, 15)):
    """Build supervised matrix: lags + calendar features -> next value."""
    n = len(values)
    max_lag = max(lags)
    X, idx = [], []
    for t in range(max_lag, n):
        row = [values[t - l] for l in lags]
        X.append(row)
        idx.append(t)
    X = np.array(X)
    # rolling statistics add trend/cycle context
    roll7 = pd.Series(values).rolling(7).mean().values
    roll28 = pd.Series(values).rolling(28).mean().values
    X = np.column_stack([X, roll7[idx], roll28[idx]])
    return X, np.array(idx)


def gbm_lag(train: pd.Series, horizon: int, seed: int = 0):
    """Gradient boosting on lag/rolling features (recursive multi-step).

    Predictive sd is the in-sample residual std (homoscedastic Gaussian
    assumption — stated explicitly, see limitations in the README).
    """
    values = train.values.astype(float)
    X, idx = _lag_features(values)
    y = values[idx]
    # drop rows with NaN rolling features (first 28 observations)
    mask = ~np.isnan(X).any(axis=1)
    X, y = X[mask], y[mask]
    model = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, random_state=seed,
    )
    model.fit(X, y)
    resid_sd = max(float(np.std(y - model.predict(X), ddof=1)), 1e-6)

    # recursive forecasting
    extended = list(values)
    means = []
    for _ in range(horizon):
        arr = np.array(extended)
        Xf, _ = _lag_features(arr)
        last = Xf[[-1]]
        if np.isnan(last).any():  # not enough history yet (never in practice here)
            means.append(extended[-7])
            extended.append(extended[-7])
            continue
        pred = float(model.predict(last)[0])
        means.append(pred)
        extended.append(pred)
    return np.array(means), np.full(horizon, resid_sd)


def structural_tsm(train: pd.Series, horizon: int, s: int = SEASONAL_PERIOD, seed: int = 0):
    """Our from-scratch structural model: MLE + Kalman forecast with analytic variance."""
    from tsforecast.statespace import forecast as kf_forecast
    fc = kf_forecast(train.values, horizon, s=s, seed=seed, n_restarts=2)
    return fc["mean"], fc["sd"]


MODELS = {
    "SeasonalNaive": seasonal_naive,
    "SARIMA": sarima,
    "GBM-Lag": gbm_lag,
    "StructuralTSM": structural_tsm,
}
