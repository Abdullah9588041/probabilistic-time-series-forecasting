"""Tests for the Kalman filter / structural time-series model."""
import numpy as np
import pytest

from tsforecast.statespace import (
    build_system, fit, forecast, forecast_from_state, kalman_filter, rts_smoother,
)


def simulate_structural(n: int, s: int, variances: np.ndarray, seed: int = 0):
    """Simulate from the structural model with known parameters."""
    rng = np.random.default_rng(seed)
    F, Q, H, R = build_system(s, variances)
    d = F.shape[0]
    x = np.zeros(d)
    P0 = np.eye(d)
    x = rng.multivariate_normal(np.zeros(d), P0)
    ys, xs = [], []
    for _ in range(n):
        x = rng.multivariate_normal(F @ x, Q)
        ys.append(H @ x + rng.normal(0, np.sqrt(R)))
        xs.append(x.copy())
    return np.array(ys), np.array(xs)


def test_build_system_shapes():
    F, Q, H, R = build_system(7, np.array([1.0, 0.1, 0.01, 0.1]))
    assert F.shape == (8, 8)
    assert Q.shape == (8, 8)
    assert H.shape == (8,)
    assert R == 1.0
    # seasonal row sums to -1 on seasonal block (sum-to-zero constraint)
    assert F[2, 2:].sum() == pytest.approx(-6.0)


def test_kalman_filter_runs_and_loglik_finite():
    y, _ = simulate_structural(300, 7, np.array([1.0, 0.2, 0.02, 0.3]), seed=1)
    out = kalman_filter(y, 7, np.array([1.0, 0.2, 0.02, 0.3]))
    assert np.isfinite(out["loglik"])
    assert out["xf"].shape == (300, 8)
    assert (out["innov_var"] > 0).all()


def test_filter_recovers_true_level_and_smoother_improves():
    variances = np.array([0.5, 0.3, 0.02, 0.2])
    y, xs_true = simulate_structural(500, 7, variances, seed=2)
    out = kalman_filter(y, 7, variances)
    xs, _ = rts_smoother(out)
    filt_rmse = np.sqrt(np.mean((out["xf"][:, 0] - xs_true[:, 0]) ** 2))
    sm_rmse = np.sqrt(np.mean((xs[:, 0] - xs_true[:, 0]) ** 2))
    # level scale is O(sqrt of variances); errors should be well under signal scale
    assert filt_rmse < 2.0
    assert sm_rmse <= filt_rmse + 1e-9  # smoothing cannot hurt (in expectation)


def test_mle_recovers_variances_order_of_magnitude():
    true = np.array([1.0, 0.4, 0.03, 0.25])
    y, _ = simulate_structural(800, 7, true, seed=3)
    res = fit(y, s=7, n_restarts=2, seed=0)
    est = res["variances"]
    # order-of-magnitude recovery (variances are only weakly identified individually)
    assert np.all(np.abs(np.log(est) - np.log(true)) < 2.5)


def test_forecast_shapes_and_interval_nesting():
    y = np.random.default_rng(0).normal(10, 1, size=400)
    fc = forecast(y, horizon=7, s=7, seed=0, n_restarts=1)
    assert fc["mean"].shape == (7,)
    assert fc["sd"].shape == (7,)
    assert (fc["sd"] > 0).all()
    lo50, hi50 = fc["intervals"][0.5]
    lo80, hi80 = fc["intervals"][0.8]
    lo95, hi95 = fc["intervals"][0.95]
    assert (lo95 <= lo80).all() and (lo80 <= lo50).all()
    assert (hi50 <= hi80).all() and (hi80 <= hi95).all()
    # predictive variance is essentially non-decreasing with horizon (uncertainty
    # accumulates); tiny periodic wiggles can arise from the seasonal rotation
    # of the terminal covariance, so allow a small relative tolerance
    assert (np.diff(fc["var"]) >= -1e-3 * fc["var"][0]).all()
