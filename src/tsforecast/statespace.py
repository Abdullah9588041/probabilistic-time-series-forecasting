"""Structural time-series model estimated with the Kalman filter.

Model (local linear trend + sum-to-zero seasonal, period ``s``):

    y_t      = mu_t + gamma_t + eps_t,    eps_t   ~ N(0, sigma_eps^2)
    mu_{t+1} = mu_t + beta_t + eta_t,     eta_t   ~ N(0, sigma_level^2)
    beta_{t+1} = beta_t + zeta_t,         zeta_t  ~ N(0, sigma_trend^2)
    gamma_{t+1} = -sum_{j=0}^{s-2} gamma_{t-j} + omega_t,
                                             omega_t ~ N(0, sigma_seas^2)

State vector x_t = [mu_t, beta_t, g_t, g_{t-1}, ..., g_{t-s+2}]  (dim d = s+1).

The Kalman recursions are derived from the Gaussian conditioning formulas;
see docs/math_notes.md for the full derivation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def build_system(s: int, variances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (F, Q, H, R) for the structural model.

    ``variances`` = [sigma_eps^2, sigma_level^2, sigma_trend^2, sigma_seas^2].
    """
    s_eps2, s_level2, s_trend2, s_seas2 = variances
    d = s + 1
    F = np.zeros((d, d))
    F[0, 0] = 1.0
    F[0, 1] = 1.0          # mu' = mu + beta
    F[1, 1] = 1.0          # beta' = beta
    F[2, 2:] = -1.0        # gamma' = -sum(gamma)  (sum-to-zero seasonal)
    for k in range(3, d):  # shift register for past seasonals
        F[k, k - 1] = 1.0
    Q = np.diag([s_level2, s_trend2, s_seas2] + [0.0] * (d - 3))
    H = np.zeros(d)
    H[0] = 1.0
    H[2] = 1.0             # y = mu + gamma_t
    return F, Q, H, float(s_eps2)


def kalman_filter(
    y: np.ndarray, s: int, variances: np.ndarray, burn: int | None = None
) -> dict:
    """Run the Kalman filter. Returns filtered states, covariances and log-likelihood.

    The log-likelihood is the prediction-error decomposition; the first
    ``burn`` (default 2*s) observations are excluded so the diffuse
    initialisation does not dominate the likelihood.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if burn is None:
        burn = 2 * s
    F, Q, H, R = build_system(s, variances)
    d = F.shape[0]

    x = np.zeros(d)            # diffuse-ish prior mean
    P = np.eye(d) * 1e6        # diffuse prior covariance

    xf = np.zeros((n, d))
    Pf = np.zeros((n, d, d))
    innov = np.zeros(n)
    innov_var = np.zeros(n)
    loglik = 0.0

    for t in range(n):
        # predict
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # update
        v = y[t] - H @ x_pred          # innovation
        S = H @ P_pred @ H + R         # innovation variance
        K = (P_pred @ H) / S           # Kalman gain
        x = x_pred + K * v
        P = P_pred - np.outer(K, H) @ P_pred
        P = (P + P.T) / 2              # keep symmetric against round-off
        xf[t], Pf[t] = x, P
        innov[t], innov_var[t] = v, S
        if t >= burn and S > 0:
            loglik += -0.5 * (np.log(2 * np.pi * S) + v * v / S)

    return {
        "xf": xf, "Pf": Pf, "innov": innov, "innov_var": innov_var,
        "loglik": loglik, "F": F, "Q": Q, "H": H, "R": R, "s": s,
    }


def rts_smoother(filt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rauch-Tung-Striebel smoother. Returns smoothed states and covariances."""
    xf, Pf = filt["xf"], filt["Pf"]
    F, Q = filt["F"], filt["Q"]
    n, d = xf.shape
    xs = xf.copy()
    Ps = Pf.copy()
    for t in range(n - 2, -1, -1):
        P_pred = F @ Pf[t] @ F.T + Q
        G = Pf[t] @ F.T @ np.linalg.pinv(P_pred)  # smoother gain
        xs[t] = xf[t] + G @ (xs[t + 1] - F @ xf[t])
        Ps[t] = Pf[t] + G @ (Ps[t + 1] - P_pred) @ G.T
        Ps[t] = (Ps[t] + Ps[t].T) / 2
    return xs, Ps


def neg_loglik(log_vars: np.ndarray, y: np.ndarray, s: int) -> float:
    variances = np.exp(log_vars)
    try:
        out = kalman_filter(y, s, variances)
    except Exception:
        return 1e12
    if not np.isfinite(out["loglik"]):
        return 1e12
    return -out["loglik"]


def fit(y: np.ndarray, s: int = 7, n_restarts: int = 3, seed: int = 0) -> dict:
    """MLE of the four variance parameters (optimised in log space)."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    v0 = np.var(np.diff(y)) if len(y) > 1 else 1.0
    base = np.log([v0 * 0.5, v0 * 0.1, v0 * 0.01, v0 * 0.1])
    best = {"fun": np.inf}
    for r in range(n_restarts):
        x0 = base + rng.normal(0, 0.75, size=4) if r else base
        res = minimize(
            neg_loglik, x0, args=(y, s), method="L-BFGS-B",
            options={"maxiter": 500},
        )
        if res.fun < best["fun"]:
            best = {"fun": res.fun, "x": res.x, "success": res.success}
    variances = np.exp(best["x"])
    filt = kalman_filter(y, s, variances)
    return {
        "variances": variances,
        "var_names": ["sigma_eps^2", "sigma_level^2", "sigma_trend^2", "sigma_seas^2"],
        "loglik": filt["loglik"],
        "n": len(y),
        "s": s,
        "filter": filt,
    }


def forecast_from_state(
    x_last: np.ndarray, P_last: np.ndarray, s: int,
    variances: np.ndarray, horizon: int, alpha_levels: tuple = (0.5, 0.8, 0.95),
) -> dict:
    """Propagate the terminal filtered state ``horizon`` steps ahead.

    Returns predictive means, variances and central prediction intervals.
    """
    F, Q, H, R = build_system(s, variances)
    x, P = x_last.copy(), P_last.copy()
    means = np.zeros(horizon)
    variances_pred = np.zeros(horizon)
    for h in range(horizon):
        x = F @ x
        P = F @ P @ F.T + Q
        means[h] = H @ x
        variances_pred[h] = H @ P @ H + R
    sd = np.sqrt(np.maximum(variances_pred, 1e-12))
    intervals = {}
    for a in alpha_levels:
        z = norm.ppf(0.5 + a / 2)
        intervals[a] = (means - z * sd, means + z * sd)
    return {"mean": means, "var": variances_pred, "sd": sd, "intervals": intervals}


def forecast(y: np.ndarray, horizon: int, s: int = 7, seed: int = 0, **fit_kwargs) -> dict:
    """Fit on ``y`` (MLE) and forecast ``horizon`` steps. Convenience wrapper."""
    res = fit(y, s=s, seed=seed, **fit_kwargs)
    filt = res["filter"]
    fc = forecast_from_state(filt["xf"][-1], filt["Pf"][-1], s, res["variances"], horizon)
    fc["fit"] = res
    return fc
