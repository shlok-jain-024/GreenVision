"""Prompt builders for the reasoning model.

Design rules baked into every prompt:
  * the model only sees history BEFORE the cutoff — never the answer;
  * retrieved memory records (its own past predictions, errors, lessons) are
    re-injected so it can pattern-match on prior hits and misses;
  * strict-JSON replies only;
  * in recommendation, the model explains and selects from provided options —
    it never invents data values and never overrides the numeric ranking.
"""

from __future__ import annotations

import json
from typing import Any

PREDICT_SYSTEM = (
    "You are the forecasting engine of GreenGrid, an urban-planning AI. "
    "You predict monthly zone-level metrics for an Indian city: traffic congestion "
    "index (0-100), AQI (0-500), and mean NDVI green-cover (0-1). "
    "Study the zone's history, its trend and seasonality (calendar months given), and "
    "especially your own PAST CASES below: they show what you predicted before, what "
    "actually happened, your error, and the lesson you recorded. Correct for the biases "
    "those lessons reveal. Reply with STRICT JSON only, no prose, no code fences: "
    '{"traffic": <number>, "aqi": <number>, "ndvi": <number>, "rationale": "<one line>"}'
)

LESSON_SYSTEM = (
    "You are reviewing one of your own forecasts against reality. "
    "State the single most useful pattern you missed or confirmed, as ONE line "
    "(max 25 words) that would improve a future forecast for a similar zone/season. "
    'Reply with STRICT JSON only: {"lesson": "<one line>"}'
)

RECOMMEND_SYSTEM = (
    "You are the recommendation writer of GreenGrid. City zones have ALREADY been "
    "ranked for tree-planting priority by a numeric multi-criteria score computed from "
    "predicted AQI worsening, predicted traffic worsening, declining green cover, low "
    "current canopy, and plantable space. Each zone may also carry a `soil` object "
    "(pH, pH class, texture, moisture). Your job: (1) for each zone, in the GIVEN "
    "order, write a short planner-facing justification grounded ONLY in the numbers "
    "provided and the lessons learned; (2) select 3-5 species per zone STRICTLY from "
    "the provided species table, matching pollution tolerance, water need, canopy size "
    "and planting context to the zone's conditions AND respecting soil fit — do NOT "
    "pick a species whose soil_ph range excludes the zone's pH or whose soil_texture "
    "excludes the zone's texture (when soil data is given). You must NOT reorder zones, "
    "invent data values, invent species, or alter species traits. Reply with STRICT "
    'JSON only: {"zones": [{"zone": "<id>", "justification": "<2-3 sentences>", '
    '"species": ["<name from table>", ...]}]}'
)

PROJECT_SYSTEM = (
    "You are annotating a LONG-RANGE, UNVALIDATED trend projection produced by "
    "damped Theil-Sen extrapolation. No observed data exists that far in the future, "
    "so it cannot be checked. Write 2-3 sentences of caveats a city planner should "
    "read before using it (compounding uncertainty, policy/climate breaks, land-use "
    'change). Reply with STRICT JSON only: {"note": "<2-3 sentences>"}'
)


def _fmt_history(history: list[dict[str, Any]]) -> str:
    lines = ["month,calendar_month,traffic,aqi,ndvi"]
    for h in history:
        lines.append(
            f"{h['month']},{h['cal_month']},{h['traffic']},{h['aqi']},{h['ndvi']}"
        )
    return "\n".join(lines)


def _fmt_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "(none yet — this is one of your first forecasts)"
    lines = []
    for r in memories:
        lines.append(
            f"- zone {r['zone']}, cutoff m{r['cutoff']}, horizon {r['horizon']}: "
            f"predicted {r['prediction']}, actual {r['actual']}, "
            f"abs_error {r['abs_error']}. Lesson: {r.get('lesson', '-')}"
        )
    return "\n".join(lines)


def predict_user(ctx: dict[str, Any]) -> str:
    return (
        f"ZONE: {ctx['zone']}\n"
        f"TASK: predict month {ctx['target_month']} "
        f"(calendar month {ctx['season']}), i.e. {ctx['horizon']} months after the last "
        f"row below. You only see history BEFORE month {ctx['cutoff']}.\n\n"
        f"HISTORY (oldest first):\n{_fmt_history(ctx['history'])}\n\n"
        f"ROBUST TRENDS (Theil-Sen, per month): "
        f"traffic {ctx['inputs']['traffic_slope']:+.3f}, "
        f"aqi {ctx['inputs']['aqi_slope']:+.3f}, "
        f"ndvi {ctx['inputs']['ndvi_slope']:+.5f}\n\n"
        f"PAST CASES from your memory (predictions, actuals, errors, lessons):\n"
        f"{_fmt_memories(ctx['memories'])}\n\n"
        "Return the strict JSON now."
    )


def lesson_user(ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
    return (
        f"Zone {ctx['zone']}, target month {ctx['target_month']} "
        f"(calendar month {ctx['season']}), horizon {ctx['horizon']}.\n"
        f"Your prediction: {json.dumps({k: round(float(pred[k]), 3) for k in ('traffic', 'aqi', 'ndvi')})}\n"
        f"Actual values:  {json.dumps({k: round(float(actual[k]), 3) for k in ('traffic', 'aqi', 'ndvi')})}\n"
        "Return the strict JSON lesson now."
    )


def recommend_user(
    ranked_rows: list[dict[str, Any]], lessons: list[str], kb_table: str
) -> str:
    zone_lines = [json.dumps(r, ensure_ascii=False) for r in ranked_rows]
    lesson_lines = "\n".join(f"- {l}" for l in lessons) if lessons else "(none)"
    return (
        "RANKED ZONES (already ordered by numeric priority score — do NOT reorder):\n"
        + "\n".join(zone_lines)
        + "\n\nLESSONS LEARNED during backtesting (context for your justifications):\n"
        + lesson_lines
        + "\n\nSPECIES TABLE (choose ONLY exact `common` names from this table):\n"
        + kb_table
        + "\n\nReturn the strict JSON now, zones in the same order as given."
    )


def project_user(city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
    return (
        f"City: {city}. Projection length: {n_months} months. "
        f"Sample of projected zone endpoints:\n"
        + "\n".join(json.dumps(r) for r in sample_rows[:8])
        + "\nReturn the strict JSON note now."
    )
