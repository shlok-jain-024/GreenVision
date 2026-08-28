"""The backtesting training loop.

Each iteration:
  1. sample a (zone, cutoff-month) from the PAST,
  2. show the model only history strictly before the cutoff,
  3. ask it to predict traffic / AQI / NDVI at cutoff + horizon,
  4. look up what really happened (still inside recorded history) and score it,
  5. ask the model for a one-line lesson,
  6. write inputs, prediction, actual, error and lesson to the memory store,
     which future prompts draw on (in-context learning — no weight updates).

At the end we compare the model's SKILL in the first half of iterations vs
the second half. Skill is measured against a memory-free naive baseline
(Theil-Sen trend + seasonal profile) computed for the SAME tasks, so
differences in task difficulty between the halves cancel out: if the memory
loop is helping, skill should rise as memory grows.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..features.trends import (
    METRIC_BOUNDS,
    METRICS,
    panel_stats,
    theil_sen,
    trend_seasonal_estimate,
)
from .memory import MemoryStore, build_feature_vector

log = logging.getLogger(__name__)


def max_validatable_horizon(panel: pd.DataFrame, min_history_months: int) -> int:
    """Largest horizon H for which a backtest check is honest: there must be
    at least one cutoff with `min_history_months` behind it AND a real
    observation H months after it inside the data."""
    n_months = int(panel["month"].max()) - int(panel["month"].min()) + 1
    return max(0, n_months - 1 - min_history_months)


def _calendar_month(month_idx: int, start_month: int) -> int:
    """1-12 calendar month for an integer month index (0 = oldest)."""
    return (start_month - 1 + month_idx) % 12 + 1


def build_predict_context(
    panel: pd.DataFrame,
    zone: str,
    cutoff: int,
    horizon: int,
    context_window: int,
    start_month: int,
    stats: dict,
    memory: MemoryStore,
    memory_k: int,
) -> dict[str, Any]:
    g = panel[(panel["zone"] == zone) & (panel["month"] < cutoff)].sort_values("month")
    tail = g.tail(context_window)
    inputs: dict[str, float] = {"season": _calendar_month(cutoff + horizon, start_month)}
    for m in METRICS:
        series = g[m].dropna()
        inputs[f"{m}_latest"] = float(series.iloc[-1]) if len(series) else float("nan")
        inputs[f"{m}_slope"] = theil_sen(tail[m].to_numpy())
    feat = build_feature_vector(inputs, stats)
    history = [
        {
            "month": int(r.month),
            "cal_month": _calendar_month(int(r.month), start_month),
            "traffic": None if pd.isna(r.traffic) else round(float(r.traffic), 1),
            "aqi": None if pd.isna(r.aqi) else round(float(r.aqi), 1),
            "ndvi": None if pd.isna(r.ndvi) else round(float(r.ndvi), 3),
        }
        for r in tail.itertuples()
    ]
    return {
        "zone": zone,
        "cutoff": int(cutoff),
        "target_month": int(cutoff + horizon),
        "horizon": int(horizon),
        "season": inputs["season"],
        "history": history,
        "inputs": inputs,
        "feat": feat,
        "memories": memory.retrieve(zone, feat, memory_k),
        "stats": stats,
    }


def run_backtest(
    panel: pd.DataFrame,
    model: Any,
    memory: MemoryStore,
    *,
    iterations: int,
    horizon: int,
    min_history: int,
    context_window: int,
    memory_k: int,
    start_month: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    stats = panel_stats(panel)
    last_month = int(panel["month"].max())

    max_h = max_validatable_horizon(panel, min_history)
    if horizon > max_h:
        raise ValueError(
            f"horizon={horizon} cannot be validated: with {last_month + 1} months of "
            f"history and min_history={min_history}, the largest honest horizon is {max_h}. "
            "Lower --horizon, add data, or use --project for an UNVALIDATED long-range forecast."
        )

    zones = sorted(panel["zone"].unique())
    per_iter: list[dict[str, Any]] = []
    attempts, max_attempts = 0, iterations * 6

    while len(per_iter) < iterations and attempts < max_attempts:
        attempts += 1
        zone = zones[int(rng.integers(len(zones)))]
        cutoff = int(rng.integers(min_history, last_month - horizon + 1))
        target = panel[(panel["zone"] == zone) & (panel["month"] == cutoff + horizon)]
        if target.empty or target[METRICS].isna().any(axis=None):
            continue
        hist = panel[(panel["zone"] == zone) & (panel["month"] < cutoff)]
        if hist["month"].nunique() < min_history:
            continue

        ctx = build_predict_context(
            panel, zone, cutoff, horizon, context_window, start_month, stats, memory, memory_k
        )
        try:
            pred = model.predict(ctx)
        except Exception as exc:  # model gave up after its own retries
            log.warning("prediction failed for %s@%d: %s", zone, cutoff, exc)
            continue

        actual = {m: float(target.iloc[0][m]) for m in METRICS}
        signed = {m: float(pred[m]) - actual[m] for m in METRICS}
        abs_err = {m: abs(e) for m, e in signed.items()}
        norm_err = float(np.mean([abs_err[m] / stats[m]["std"] for m in METRICS]))

        # memory-free naive baseline on the same task -> per-task skill
        base_errs = []
        for m in METRICS:
            pts = [
                (h["month"], h["cal_month"], h[m]) for h in ctx["history"] if h[m] is not None
            ]
            est = trend_seasonal_estimate(pts, ctx["target_month"], ctx["season"])
            if est is None:
                est = ctx["inputs"][f"{m}_latest"]
            est = float(np.clip(est, *METRIC_BOUNDS[m]))
            base_errs.append(abs(est - actual[m]) / stats[m]["std"])
        base_norm_err = float(np.mean(base_errs))
        skill = base_norm_err - norm_err  # positive = beat the naive baseline

        try:
            lesson = model.lesson(ctx, pred, actual)
        except Exception as exc:
            log.warning("lesson failed for %s@%d: %s", zone, cutoff, exc)
            lesson = "(no lesson: model call failed)"

        memory.append(
            {
                "zone": zone,
                "cutoff": cutoff,
                "horizon": horizon,
                "inputs": {k: round(v, 4) for k, v in ctx["inputs"].items()},
                "feat": ctx["feat"],
                "prediction": {m: round(float(pred[m]), 4) for m in METRICS},
                "actual": {m: round(actual[m], 4) for m in METRICS},
                "signed_error": {m: round(signed[m], 4) for m in METRICS},
                "abs_error": {m: round(abs_err[m], 4) for m in METRICS},
                "norm_error": round(norm_err, 4),
                "baseline_norm_error": round(base_norm_err, 4),
                "skill": round(skill, 4),
                "lesson": lesson,
            }
        )
        per_iter.append(
            {
                "zone": zone,
                "cutoff": cutoff,
                "abs_error": abs_err,
                "norm_error": norm_err,
                "skill": skill,
            }
        )
        if len(per_iter) % 10 == 0:
            log.info("backtest %d/%d iterations done", len(per_iter), iterations)

    if not per_iter:
        raise RuntimeError("backtest produced no scored iterations; check data coverage")

    half = len(per_iter) // 2
    first, second = per_iter[:half], per_iter[half:]
    report = {
        "iterations": len(per_iter),
        "horizon": horizon,
        "mae": {
            m: float(np.mean([r["abs_error"][m] for r in per_iter])) for m in METRICS
        },
        "norm_error_overall": float(np.mean([r["norm_error"] for r in per_iter])),
        "norm_error_first_half": float(np.mean([r["norm_error"] for r in first])) if first else None,
        "norm_error_second_half": float(np.mean([r["norm_error"] for r in second])) if second else None,
        # skill = naive-baseline error minus model error on the SAME tasks,
        # so halves are comparable even if one half drew harder tasks
        "skill_vs_baseline_first_half": float(np.mean([r["skill"] for r in first])) if first else None,
        "skill_vs_baseline_second_half": float(np.mean([r["skill"] for r in second])) if second else None,
    }
    if first and second:
        gain = report["skill_vs_baseline_second_half"] - report["skill_vs_baseline_first_half"]
        # "helping" = by the second half of training, the model WITH memory
        # beats the memory-free baseline on the same tasks
        report["memory_helped"] = bool(report["skill_vs_baseline_second_half"] > 0)
        report["skill_gain_first_to_second_half"] = round(float(gain), 4)
    return report
