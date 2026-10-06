"""End-to-end analysis: rolling-origin backtest of 4 forecasters on daily
household energy data, with point AND probabilistic evaluation.

Outputs:
  results/metrics.json      — all computed metrics (nothing hand-written)
  results/figures/*.png     — fan chart, coverage plot, diagnostics, schematic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tsforecast import baselines, data, evaluation, statespace, visualization

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
FIGURES = RESULTS / "figures"
HORIZON = 7
SEED = 0


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    y = data.load_daily()
    info = data.describe(y)
    print("data:", info)

    # --- rolling-origin design: 13 weekly origins over the last ~98 days ---
    origins = list(pd.date_range(y.index[-98], y.index[-8], freq="7D"))
    print(f"origins: {len(origins)} ({origins[0].date()} -> {origins[-1].date()}), horizon={HORIZON}d")
    visualization.backtest_schematic(y, origins, HORIZON, str(FIGURES / "backtest_design.png"))

    # --- backtest ---
    results = evaluation.rolling_origin_backtest(y, origins, HORIZON, baselines.MODELS, seed=SEED)
    summary = evaluation.summarise(results)
    print(summary.to_string())

    # --- final StructuralTSM fit on full sample: diagnostics + fan chart ---
    fit = statespace.fit(y.values, s=7, n_restarts=3, seed=SEED)
    print("MLE variances:", dict(zip(fit["var_names"], np.round(fit["variances"], 6))),
          "loglik:", round(fit["loglik"], 1))
    visualization.residual_diagnostics(fit["filter"], y.values, str(FIGURES / "residual_diagnostics.png"))

    last_origin = origins[-1]
    fc = statespace.forecast(y.loc[:last_origin].values, HORIZON, s=7, seed=SEED, n_restarts=3)
    visualization.fan_chart(y, last_origin, HORIZON, fc, str(FIGURES / "fan_chart.png"))
    visualization.coverage_plot(summary, str(FIGURES / "coverage.png"))

    metrics = {
        "seed": SEED,
        "data": info,
        "design": {
            "horizon_days": HORIZON,
            "n_origins": len(origins),
            "origin_step_days": 7,
            "first_origin": str(origins[0].date()),
            "last_origin": str(origins[-1].date()),
        },
        "structural_tsm_mle": {
            name: float(v) for name, v in zip(fit["var_names"], fit["variances"])
        },
        "backtest_summary": summary.to_dict(orient="index"),
        "per_origin": results.to_dict(orient="records"),
    }
    # JSON needs string keys for timestamps
    for row in metrics["per_origin"]:
        row["origin"] = str(row["origin"].date())
    with open(RESULTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote results/metrics.json and figures/")
    return metrics


if __name__ == "__main__":
    main()
