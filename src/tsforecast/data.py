"""Data loading and preprocessing.

Source: UCI Machine Learning Repository — "Individual household electric power
consumption" (Hebrail & Bérard). Minute-level measurements from one household
in Sceaux, France, Dec 2006 – Nov 2010, accessed 2026-10-06 from
https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip

We aggregate Global_active_power (kW, 1-minute resolution) to DAILY total energy
(kWh), which exhibits strong weekly seasonality (period 7) plus trend — ideal
for structural time-series modelling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
RAW_PATH = BASE / "data" / "raw" / "household_power_consumption.txt"
PROCESSED_PATH = BASE / "data" / "processed" / "daily_kwh.csv"

SOURCE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/"
    "household_power_consumption.zip"
)
ACCESS_DATE = "2026-10-06"


def load_daily(force: bool = False) -> pd.Series:
    """Return daily total active energy (kWh) as a DatetimeIndex series.

    Caches the processed series to ``data/processed/daily_kwh.csv`` so the raw
    133 MB file is parsed only once.
    """
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists() and not force:
        s = pd.read_csv(PROCESSED_PATH, parse_dates=["date"], index_col="date")["kwh"]
        return s.asfreq("D").rename("kwh")

    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        usecols=["Date", "Time", "Global_active_power"],
        dtype={"Global_active_power": str},
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    gap = pd.to_numeric(df["Global_active_power"].replace("?", np.nan))
    ts = pd.Series(gap.values, index=dt).sort_index()
    # kW averaged over each minute -> kWh per day
    daily = ts.resample("D").sum(min_count=1) / 60.0
    # the raw file has one multi-day gap (Aug 2010); linear interpolation is
    # adequate for a smooth daily-energy series and is documented here
    daily = daily.interpolate(limit_direction="both").dropna()
    daily.name = "kwh"
    daily.index.name = "date"
    daily.to_csv(PROCESSED_PATH, header=True)
    return daily.asfreq("D").rename("kwh")


def train_test_split(y: pd.Series, test_days: int = 91) -> tuple[pd.Series, pd.DataFrame]:
    """Split into train and a trailing test block of ``test_days`` days."""
    return y.iloc[:-test_days], y.iloc[-test_days:]


def describe(y: pd.Series) -> dict:
    return {
        "start": str(y.index[0].date()),
        "end": str(y.index[-1].date()),
        "n_days": int(len(y)),
        "mean_kwh": float(y.mean()),
        "std_kwh": float(y.std()),
        "min_kwh": float(y.min()),
        "max_kwh": float(y.max()),
        "missing": int(y.isna().sum()),
    }
