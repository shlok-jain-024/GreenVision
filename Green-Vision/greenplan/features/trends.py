"""Trend and feature extraction on the H3 panel.

NDVI (and the other metrics) get a Theil-Sen slope: the median of all
pairwise slopes, which is robust to the outlier months that are common in
satellite composites (cloud contamination) and sensor feeds.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd

METRICS = ["traffic", "aqi", "ndvi"]
METRIC_BOUNDS = {"traffic": (0.0, 100.0), "aqi": (10.0, 500.0), "ndvi": (0.01, 0.9)}


def pairwise_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen slope with explicit x positions."""
    if len(y) < 3:
        return 0.0
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(y))
        for j in range(i + 1, len(y))
        if x[j] != x[i]
    ]
    return float(np.median(slopes)) if slopes else 0.0


def trend_seasonal_estimate(
    points: list[tuple[int, int, float]], target_month: int, target_cal: int
) -> float | None:
    """Naive statistical forecast: Theil-Sen trend + detrended seasonal
    profile. `points` are (month_index, calendar_month, value) triples.
    Serves both as the mock model's core and as the memory-free baseline the
    backtest report measures skill against."""
    if len(points) < 3:
        return None
    x = np.array([p[0] for p in points], dtype=float)
    cal = np.array([p[1] for p in points])
    y = np.array([p[2] for p in points], dtype=float)
    slope = pairwise_slope(x, y)
    detrended = y - slope * x
    level = float(np.mean(detrended))
    same_season = detrended[cal == target_cal]
    seasonal = float(np.mean(same_season) - level) if len(same_season) else 0.0
    return slope * target_month + level + seasonal


def theil_sen(y: np.ndarray) -> float:
    """Theil-Sen slope per month index. NaNs are dropped; needs >= 3 points."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(y) < 3:
        return 0.0
    slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(len(y)), 2)]
    return float(np.median(slopes))


def normalize(values: pd.Series) -> pd.Series:
    """Robust 0-1 scaling using the 5th-95th percentile band."""
    v = values.astype(float)
    lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    if math.isclose(hi, lo):
        return pd.Series(np.full(len(v), 0.5), index=values.index)
    return ((v - lo) / (hi - lo)).clip(0.0, 1.0)


def panel_stats(panel: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean/std per metric across the whole panel (for z-scoring features)."""
    out: dict[str, dict[str, float]] = {}
    for m in METRICS:
        col = panel[m].dropna()
        std = float(col.std()) if len(col) else float("nan")
        # Both degenerate spreads fall back to 1.0. The old one-liner
        # (`float(col.std()) or 1.0 if len(col) else 1.0`) caught a std of 0.0,
        # which is falsy, but NOT a std of NaN — and NaN is TRUTHY, so a metric
        # with a single observation put NaN into every downstream z-score. The
        # precedence was also easy to misread: the conditional binds looser than
        # `or`, so it parsed as `(x or 1.0) if len(col) else 1.0`.
        out[m] = {
            "mean": float(col.mean()) if len(col) else 0.0,
            "std": std if math.isfinite(std) and std != 0.0 else 1.0,
        }
    return out


def zone_features(
    panel: pd.DataFrame, trend_window: int = 24
) -> pd.DataFrame:
    """One row per zone: latest values, Theil-Sen slopes (per month) over the
    trailing `trend_window` months, and a plantable-space proxy.

    NOTE: `plantable_space` is a PLACEHOLDER proxy — it assumes zones with
    less existing canopy have more unplanted ground (1 - NDVI, rescaled).
    Replace it with real land-use / open-space data by extending an adapter
    before using recommendations operationally.
    """
    rows = []
    for zone, g in panel.groupby("zone"):
        g = g.sort_values("month")
        tail = g.tail(trend_window)
        row: dict[str, float | str] = {"zone": zone, "latest_month": int(g["month"].max())}
        for m in METRICS:
            series = g[m].dropna()
            row[f"{m}_latest"] = float(series.iloc[-1]) if len(series) else np.nan
            row[f"{m}_slope"] = theil_sen(tail[m].to_numpy())
        ndvi_latest = row.get("ndvi_latest")
        # No NDVI for this cell means the 1-NDVI proxy has nothing to stand on,
        # so there is no plantable-space estimate to give. It used to fall back
        # to 0.5, which is indistinguishable on the map from a measured "half
        # this cell is plantable" — a substituted number wearing a measurement's
        # clothes. NaN instead, which propagates to a null in the outputs. These
        # cells already score NaN (c_low_green_cover = 1 - normalize(NaN)), so
        # nothing that was ranked before stops being ranked.
        row["plantable_space"] = (
            float(np.clip((0.6 - ndvi_latest) / 0.6, 0.0, 1.0))
            if ndvi_latest is not None and not math.isnan(ndvi_latest)
            else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)
