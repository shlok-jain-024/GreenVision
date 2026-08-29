"""Feature engineering for the trained forecaster.

One vector per (zone, cutoff, metric), built from the SAME prediction context
dict that `build_predict_context` hands the language model — so the trained
model and the LLM see identical information and their skill numbers are
directly comparable.

The first feature is the Theil-Sen + seasonality baseline estimate itself:
the network's job is to learn *corrections* to the honest baseline, not to
rediscover the trend from scratch. With ~1,700 training samples, that framing
is what makes a small model work at all.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..features.trends import METRICS, trend_seasonal_estimate

FEATURE_NAMES = [
    "baseline_estimate",   # trend + seasonal profile projected to the target month
    "latest",              # last observed value of this metric
    "slope",               # Theil-Sen robust slope (per month)
    "mean_last3",
    "mean_last6",
    "std_last6",
    "seasonal_lag",        # value 12 months before the target, if recorded
    "season_sin",          # target calendar month on the unit circle
    "season_cos",
    "horizon",             # months past the cutoff being predicted
    "aqi_latest",          # cross-metric context
    "ndvi_latest",
    "traffic_latest",
]


def _f(value: Any, fallback: float) -> float:
    try:
        v = float(value)
        return fallback if math.isnan(v) else v
    except (TypeError, ValueError):
        return fallback


def feature_vector(ctx: dict[str, Any], metric: str) -> np.ndarray:
    """Build the feature vector for one metric from a prediction context."""
    hist = [
        (h["month"], h["cal_month"], h[metric])
        for h in ctx["history"]
        if h[metric] is not None
    ]
    vals = [v for _, _, v in hist]
    latest = _f(ctx["inputs"].get(f"{metric}_latest"), vals[-1] if vals else 0.0)
    slope = _f(ctx["inputs"].get(f"{metric}_slope"), 0.0)

    base = trend_seasonal_estimate(hist, ctx["target_month"], ctx["season"])
    base = _f(base, latest)

    lag_month = ctx["target_month"] - 12
    seasonal_lag = next((v for mth, _, v in hist if mth == lag_month), latest)

    last3 = vals[-3:] or [latest]
    last6 = vals[-6:] or [latest]
    angle = 2.0 * math.pi * (ctx["season"] - 1) / 12.0

    cross = {
        m: _f(ctx["inputs"].get(f"{m}_latest"), 0.0) for m in METRICS
    }

    return np.array(
        [
            base,
            latest,
            slope,
            float(np.mean(last3)),
            float(np.mean(last6)),
            float(np.std(last6)),
            float(seasonal_lag),
            math.sin(angle),
            math.cos(angle),
            float(ctx["horizon"]),
            cross.get("aqi", 0.0),
            cross.get("ndvi", 0.0),
            cross.get("traffic", 0.0),
        ],
        dtype=np.float32,
    )
