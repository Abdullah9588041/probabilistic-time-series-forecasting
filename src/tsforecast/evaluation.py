"""Evaluation: point metrics, probabilistic metrics (CRPS, interval coverage),
and a rolling-origin backtest harness with no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------- point metrics
def mae(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.mean(np.abs(y - mu)))


def rmse(y: np.ndarray, mu: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - mu) ** 2)))


def mape(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - mu[mask]) / y[mask])) * 100)


# ------------------------------------------------------- probabilistic metrics
def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution.

    CRPS(F, y) = sigma * [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ],
    z = (y - mu) / sigma.  A *proper* scoring rule: minimised in expectation
    by reporting the true predictive distribution (see docs/math_notes.md).
    """
    y, mu, sigma = map(lambda a: np.asarray(a, dtype=float), (y, mu, sigma))
    sigma = np.maximum(sigma, 1e-12)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def interval_coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Empirical fraction of actuals falling inside [lower, upper]."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)))


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean(upper - lower))


# ------------------------------------------------------------------- backtest
def rolling_origin_backtest(
    y: pd.Series,
    origins: list[pd.Timestamp],
    horizon: int,
    model_fns: dict,
    seed: int = 0,
) -> pd.DataFrame:
    """Rolling-origin evaluation. Each model is fit ONLY on ``y[:origin]``.

    Returns one row per (origin, model) with point + probabilistic metrics.
    """
    rows = []
    for origin in origins:
        train = y.loc[:origin]
        actual = y.loc[origin + pd.Timedelta(days=1):].iloc[:horizon].values
        if len(actual) < horizon:
            continue
        for name, fn in model_fns.items():
            mean, sd = fn(train, horizon)  # all fns use fixed internal seeds
            sd = np.maximum(np.asarray(sd, dtype=float), 1e-9)
            mean = np.asarray(mean, dtype=float)
            lo80, hi80 = mean - norm.ppf(0.9) * sd, mean + norm.ppf(0.9) * sd
            lo95, hi95 = mean - norm.ppf(0.975) * sd, mean + norm.ppf(0.975) * sd
            rows.append({
                "origin": origin, "model": name, "h": horizon,
                "mae": mae(actual, mean), "rmse": rmse(actual, mean),
                "mape": mape(actual, mean),
                "crps": float(np.mean(crps_gaussian(actual, mean, sd))),
                "cov80": interval_coverage(actual, lo80, hi80),
                "cov95": interval_coverage(actual, lo95, hi95),
                "width80": interval_width(lo80, hi80),
                "width95": interval_width(lo95, hi95),
            })
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per model across origins."""
    agg = results.groupby("model").agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), mape=("mape", "mean"),
        crps=("crps", "mean"), cov80=("cov80", "mean"), cov95=("cov95", "mean"),
        width80=("width80", "mean"), width95=("width95", "mean"),
        n_origins=("origin", "nunique"),
    )
    return agg.round(4)
