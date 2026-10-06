"""Visualisation: fan charts, coverage reliability, residual diagnostics."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def fan_chart(y: pd.Series, origin: pd.Timestamp, horizon: int,
              fc: dict, path: str, title: str = "StructuralTSM forecast — fan chart"):
    """Plot history + forecast mean with 50/80/95% prediction intervals."""
    hist = y.loc[:origin].iloc[-120:]
    future_idx = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
    actual = y.reindex(future_idx)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist.index, hist.values, color="black", lw=1.2, label="history")
    if actual.notna().all():
        ax.plot(future_idx, actual.values, color="black", lw=1.5, ls="--", label="actual")
    mean, sd = fc["mean"], fc["sd"]
    for alpha, color, lab in [(0.95, "#cfe8ff", "95%"), (0.8, "#9ecfff", "80%"), (0.5, "#5aa9ff", "50%")]:
        z = norm.ppf(0.5 + alpha / 2)
        ax.fill_between(future_idx, mean - z * sd, mean + z * sd, color=color, alpha=0.8, label=f"{lab} interval")
    ax.plot(future_idx, mean, color="#0b5fff", lw=2, label="forecast mean")
    ax.axvline(origin, color="grey", ls=":", label="origin")
    ax.set_title(title)
    ax.set_ylabel("daily energy (kWh)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def coverage_plot(summary: pd.DataFrame, path: str):
    """Nominal vs empirical coverage per model (80% and 95% intervals)."""
    models = list(summary.index)
    x = np.arange(len(models))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 1.5 * w, summary["cov80"], w, label="empirical 80%")
    ax.bar(x - 0.5 * w, summary["cov95"], w, label="empirical 95%")
    ax.bar(x + 0.5 * w, [0.8] * len(models), w, label="nominal 80%", hatch="//", edgecolor="black", fill=False)
    ax.bar(x + 1.5 * w, [0.95] * len(models), w, label="nominal 95%", hatch="\\\\", edgecolor="black", fill=False)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Prediction-interval calibration: nominal vs empirical coverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_diagnostics(filt: dict, y: np.ndarray, path: str):
    """Standardised innovations: time plot + histogram + ACF."""
    innov = filt["innov"]
    ivar = filt["innov_var"]
    burn = 2 * filt["s"]
    std_resid = innov[burn:] / np.sqrt(ivar[burn:])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].plot(std_resid, lw=0.7)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("standardised innovations")
    axes[1].hist(std_resid, bins=40, density=True, alpha=0.7)
    xs = np.linspace(-4, 4, 200)
    axes[1].plot(xs, norm.pdf(xs), "r-", lw=1.5, label="N(0,1)")
    axes[1].set_title("histogram vs N(0,1)")
    axes[1].legend(fontsize=8)
    max_lag = 28
    acf = [1.0] + [np.corrcoef(std_resid[:-l], std_resid[l:])[0, 1] for l in range(1, max_lag + 1)]
    axes[2].stem(range(max_lag + 1), acf)
    axes[2].axhline(1.96 / np.sqrt(len(std_resid)), color="r", ls="--", lw=1)
    axes[2].axhline(-1.96 / np.sqrt(len(std_resid)), color="r", ls="--", lw=1)
    axes[2].set_title("ACF of standardised innovations")
    fig.suptitle("StructuralTSM residual diagnostics (in-sample, post burn-in)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def backtest_schematic(y: pd.Series, origins: list, horizon: int, path: str):
    """Diagram of the rolling-origin design (train windows + forecast horizons)."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(y.index, y.values, color="lightgrey", lw=0.8)
    for i, o in enumerate(origins):
        ax.axvline(o, color="C0", lw=1)
        ax.hlines(y.max() * 1.02, o, o + pd.Timedelta(days=horizon), colors="C3", lw=3)
    ax.set_title(f"rolling-origin backtest: {len(origins)} origins, horizon {horizon} days")
    ax.set_ylabel("daily energy (kWh)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
