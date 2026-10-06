# Data

## Source (REAL data — no synthetic generation)

**UCI Machine Learning Repository: "Individual household electric power
consumption"** (G. Hebrail, A. Bérard, EDF R&D).

- Download URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- Accessed: 2026-10-06
- Contents: 2,075,259 minute-level measurements from a single household in
  Sceaux, France, 16 Dec 2006 – 26 Nov 2010 (`;`-separated, `?` = missing).

## Processing (`src/tsforecast/data.py::load_daily`)

1. Parse `Date;Time;Global_active_power` (kW, 1-minute resolution).
2. Aggregate to **daily total active energy (kWh)**: `sum(minute kW) / 60`.
3. The raw file has one multi-day gap (Aug 2010); filled by linear
   interpolation (documented choice — the series is smooth at daily scale).
4. Cached to `data/processed/daily_kwh.csv` (1,442 days, no missing values).

`data/raw/` (20 MB zip, 133 MB txt) is **git-ignored**; the small processed
CSV is committed so results are reproducible without re-downloading.

## Why this series

Daily household energy shows a strong **weekly seasonal pattern** (period 7:
weekday vs weekend routines) plus slow trend — exactly the structure the
local-linear-trend + seasonal state-space model is designed for, while still
being hard enough that naive methods are visibly worse.
