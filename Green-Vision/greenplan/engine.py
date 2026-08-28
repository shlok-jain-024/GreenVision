"""GreenGrid orchestration: data -> H3 panel -> backtest training ->
prediction -> MCDA ranking -> model recommendation -> files.

Single entry point:  engine.run(cfg, ...)
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .adapters import base as adapter_base
from .adapters.csvfile import CsvMetricAdapter
from .adapters.mock import (
    MockAQIAdapter,
    MockCityDataset,
    MockGreenCoverAdapter,
    MockTrafficAdapter,
)
from .config import Config
from .features.h3grid import (
    aggregate_to_h3,
    cell_boundary_lonlat,
    cell_center,
    latlng_to_cell,
)
from .features.sites import CandidateSites, load_candidate_sites
from .features.soil import SoilProfile, load_soil
from .features.trends import METRICS, normalize, panel_stats, zone_features
from .reasoning.client import BOUNDS, build_model
from .training.backtest import (
    build_predict_context,
    max_validatable_horizon,
    run_backtest,
)
from .training.memory import MemoryStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_adapters(cfg: Config, mock: bool) -> list[adapter_base.DataAdapter]:
    """Instantiate one adapter per stream, honoring config unless --mock
    forces every stream to its mock implementation."""
    ds: MockCityDataset | None = None

    def mock_ds() -> MockCityDataset:
        nonlocal ds
        if ds is None:
            ds = MockCityDataset(
                bbox=cfg.city.bbox,
                n_zones=cfg.data.n_mock_zones,
                months=cfg.data.months_history,
                start_month=cfg.data.start_month,
                seed=cfg.run.seed,
            )
        return ds

    plan = {
        "green_cover": (cfg.adapters.green_cover, MockGreenCoverAdapter, adapter_base.GreenCoverAdapter, "ndvi"),
        "traffic": (cfg.adapters.traffic, MockTrafficAdapter, adapter_base.TrafficAdapter, "traffic"),
        "aqi": (cfg.adapters.aqi, MockAQIAdapter, adapter_base.AQIAdapter, "aqi"),
    }
    adapters: list[adapter_base.DataAdapter] = []
    for name, (kind, mock_cls, real_cls, metric) in plan.items():
        if mock or kind == "mock":
            adapters.append(mock_cls(mock_ds()))
        elif kind.startswith("csv:"):
            adapters.append(CsvMetricAdapter(cfg.resolve(kind[4:].strip()), metric))
        else:
            adapters.append(real_cls())
        log.info("adapter %-12s -> %s", name, type(adapters[-1]).__name__)
    return adapters


def load_panel(cfg: Config, mock: bool) -> pd.DataFrame:
    """Fetch all three streams, merge, snap onto the H3 grid."""
    adapters = _build_adapters(cfg, mock)
    frames, geometry = [], {}
    for a in adapters:
        frames.append(a.validate(a.fetch()))
        geo = a.zone_geometry()
        if geo:
            for k, v in geo.items():
                geometry.setdefault(k, v)
    panel = aggregate_to_h3(frames, geometry or None, cfg.grid.h3_resolution)
    log.info(
        "panel: %d H3 cells (res %d), months %d..%d",
        panel["zone"].nunique(),
        cfg.grid.h3_resolution,
        panel["month"].min(),
        panel["month"].max(),
    )
    return panel


# ---------------------------------------------------------------------------
# Training (with retrain policy)
# ---------------------------------------------------------------------------


def _meta_path(cfg: Config) -> Path:
    return cfg.resolve(cfg.training.memory_path).with_suffix(".meta.json")


def train(cfg: Config, panel: pd.DataFrame, model: Any, memory: MemoryStore) -> dict | None:
    t = cfg.training
    policy = t.retrain.policy
    meta_path = _meta_path(cfg)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    max_month = int(panel["month"].max())

    if policy == "never" and len(memory):
        log.info("retrain policy 'never': reusing %d memory records", len(memory))
        return None
    if (
        policy == "if_new_months"
        and len(memory)
        and max_month - int(meta.get("max_month", -(10**9))) < t.retrain.new_months_threshold
    ):
        log.info(
            "retrain policy 'if_new_months': only %d new month(s) since last training "
            "(< %d); reusing %d memory records. Set policy 'always' to force.",
            max_month - int(meta.get("max_month", 0)),
            t.retrain.new_months_threshold,
            len(memory),
        )
        return None

    report = run_backtest(
        panel,
        model,
        memory,
        iterations=t.iterations,
        horizon=t.horizon_months,
        min_history=t.min_history_months,
        context_window=t.context_window,
        memory_k=t.memory_k,
        start_month=cfg.data.start_month,
        seed=cfg.run.seed,
    )
    meta_path.write_text(
        json.dumps({"max_month": max_month, "trained_at": datetime.now(timezone.utc).isoformat(),
                    "memory_records": len(memory), "last_report": report}, indent=2),
        encoding="utf-8",
    )
    return report


# ---------------------------------------------------------------------------
# Forward prediction + recommendation
# ---------------------------------------------------------------------------


def predict_future(cfg: Config, panel: pd.DataFrame, model: Any, memory: MemoryStore) -> pd.DataFrame:
    """Predict each zone `horizon` months past the end of the data, using the
    full history plus retrieved memory (the learned patterns)."""
    t = cfg.training
    stats = panel_stats(panel)
    last_month = int(panel["month"].max())
    rows = []
    for zone in sorted(panel["zone"].unique()):
        ctx = build_predict_context(
            panel, zone, last_month + 1, t.horizon_months, t.context_window,
            cfg.data.start_month, stats, memory, t.memory_k,
        )
        try:
            pred = model.predict(ctx)
        except Exception as exc:
            log.warning("future prediction failed for %s: %s", zone, exc)
            continue
        rows.append({"zone": zone, **{f"{m}_pred": pred[m] for m in METRICS}})
    return pd.DataFrame(rows)


def _load_sites(cfg: Config) -> CandidateSites | None:
    """Bare-ground patches for the site finder, if configured."""
    if not cfg.sites.enabled or not cfg.sites.candidates_csv:
        return None
    return load_candidate_sites(
        cfg.resolve(cfg.sites.candidates_csv),
        cfg.grid.h3_resolution,
        cfg.sites.min_patch_m2,
    )


def _load_plantable_fraction(cfg: Config) -> dict[str, float] | None:
    """Legacy pre-computed 0..1 plantable-fraction CSV (sites.plantable_csv),
    averaged per H3 cell. Used only when no candidate-sites CSV is supplied."""
    if not cfg.sites.plantable_csv:
        return None
    path = cfg.resolve(cfg.sites.plantable_csv)
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    value_col = next((c for c in ("plantable", "value", "mean") if c in df.columns), None)
    if value_col is None:
        raise ValueError(f"{path}: need a 'plantable'/'value'/'mean' column")
    if "lat" in df.columns and "lon" in df.columns:
        df["cell"] = [
            latlng_to_cell(la, lo, cfg.grid.h3_resolution)
            for la, lo in zip(df["lat"], df["lon"])
        ]
    elif "zone" in df.columns:
        df["cell"] = df["zone"].astype(str)
    else:
        raise ValueError(f"{path}: need 'lat'+'lon' or 'zone' columns")
    df = df.dropna(subset=[value_col])
    out = df.groupby("cell")[value_col].mean().clip(0.0, 1.0).to_dict()
    log.info("plantable-land data: %d cells from %s", len(out), path.name)
    return out


def rank_zones(
    cfg: Config,
    panel: pd.DataFrame,
    preds: pd.DataFrame,
    sites: CandidateSites | None = None,
) -> pd.DataFrame:
    """Numeric MCDA ranking — this ordering is authoritative; the reasoning
    model only explains it and picks species.

    Predicted deltas are YEAR-OVER-YEAR (prediction vs the observed value at
    the same calendar month one year earlier) so "worsening" measures real
    deterioration, not the seasonal cycle between the latest month and the
    target month. Falls back to the latest value if that month is missing.

    `plantable_space` comes from real bare-patch area when candidate sites are
    supplied (cells with no detected bare ground -> 0), else a legacy fraction
    CSV, else the 1-NDVI placeholder proxy from zone_features.
    """
    feats = zone_features(panel, trend_window=cfg.training.context_window)
    df = feats.merge(preds, on="zone", how="inner")

    df["n_sites"] = 0
    if sites is not None and not sites.empty:
        frac = sites.plantable_fraction_by_cell()
        counts = sites.sites.groupby("cell").size().to_dict()
        # a comprehensive bare-ground export covers the bbox, so a cell absent
        # from it has ~no detected open ground -> 0, not the optimistic proxy
        df["plantable_space"] = df["zone"].map(frac).fillna(0.0)
        df["n_sites"] = df["zone"].map(counts).fillna(0).astype(int)
    else:
        frac = _load_plantable_fraction(cfg)
        if frac is not None:
            df["plantable_space"] = df["zone"].map(frac).fillna(df["plantable_space"])

    base_month = int(panel["month"].max()) + cfg.training.horizon_months - 12
    base = (
        panel[panel["month"] == base_month].set_index("zone")
        if base_month >= int(panel["month"].min())
        else pd.DataFrame(columns=METRICS)
    )
    for m in METRICS:
        baseline = df["zone"].map(base[m]) if not base.empty else pd.Series(np.nan, index=df.index)
        df[f"{m}_baseline"] = baseline.fillna(df[f"{m}_latest"])
        df[f"{m}_pred_delta"] = df[f"{m}_pred"] - df[f"{m}_baseline"]

    w = cfg.mcda.weights
    df["c_aqi_worsening"] = normalize(df["aqi_pred_delta"])
    df["c_traffic_worsening"] = normalize(df["traffic_pred_delta"])
    df["c_ndvi_decline"] = normalize(-df["ndvi_slope"])
    df["c_low_green_cover"] = 1.0 - normalize(df["ndvi_latest"])
    df["c_plantable_space"] = df["plantable_space"].astype(float)
    df["score"] = (
        w["aqi_worsening"] * df["c_aqi_worsening"]
        + w["traffic_worsening"] * df["c_traffic_worsening"]
        + w["ndvi_decline"] * df["c_ndvi_decline"]
        + w["low_green_cover"] * df["c_low_green_cover"]
        + w["plantable_space"] * df["c_plantable_space"]
    )
    df = df.sort_values("score", ascending=False, ignore_index=True)
    df["rank"] = df.index + 1
    return df


def _ranked_rows_for_model(
    df: pd.DataFrame, top_n: int, soil: dict[str, SoilProfile] | None = None
) -> list[dict[str, Any]]:
    rows = []
    for r in df.head(top_n).itertuples():
        row = {
            "rank": int(r.rank),
            "zone": r.zone,
            "score": round(float(r.score), 3),
            "traffic_latest": round(float(r.traffic_latest), 1),
            "aqi_latest": round(float(r.aqi_latest), 1),
            "ndvi_latest": round(float(r.ndvi_latest), 3),
            "traffic_pred_delta": round(float(r.traffic_pred_delta), 1),
            "aqi_pred_delta": round(float(r.aqi_pred_delta), 1),
            "ndvi_slope": round(float(r.ndvi_slope), 5),
            "plantable_space": round(float(r.plantable_space), 2),
            "n_sites": int(getattr(r, "n_sites", 0)),
        }
        prof = (soil or {}).get(r.zone)
        row["soil"] = prof.as_dict() if prof is not None else None
        rows.append(row)
    return rows


def recommend(
    cfg: Config, panel: pd.DataFrame, model: Any, memory: MemoryStore, train_report: dict | None
) -> dict[str, Any]:
    preds = predict_future(cfg, panel, model, memory)
    if preds.empty:
        raise RuntimeError("no zone predictions available; cannot rank")
    sites = _load_sites(cfg)
    soil = load_soil(
        cfg.resolve(cfg.soil.soilgrids_csv) if cfg.soil.soilgrids_csv else None,
        cfg.resolve(cfg.soil.moisture_csv) if cfg.soil.moisture_csv else None,
        cfg.grid.h3_resolution,
    )
    ranked = rank_zones(cfg, panel, preds, sites)
    top_rows = _ranked_rows_for_model(ranked, cfg.mcda.top_n, soil)

    lessons = []
    for r in reversed(memory.records):  # newest lessons first, deduplicated
        l = r.get("lesson", "").strip()
        if l and l not in lessons:
            lessons.append(l)
        if len(lessons) >= 12:
            break

    recs = model.recommend(top_rows, lessons)
    out_dir = cfg.resolve(cfg.run.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_outputs(cfg, ranked, recs, train_report, lessons, out_dir, sites, soil)
    return {"ranked": ranked, "recommendations": recs, "outputs_dir": str(out_dir)}


# ---------------------------------------------------------------------------
# Output writers (GeoJSON + CSV + plain-text brief; no rendering)
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with None, recursively.

    Python's json.dumps happily writes bare `NaN` / `Infinity`. That is
    valid *Python*, and json.load reads it back, but it is NOT valid JSON:
    every browser's JSON.parse rejects it outright.

    It went unnoticed because the HTTP server sanitises on the way out, so
    the API was always clean. The FILE on disk was not — which broke exactly
    the two paths that have no server in front of them: opening index.html
    from the filesystem, and any static deploy. A cell with no NDVI coverage
    is enough to poison the whole document, because JSON.parse is all-or-
    nothing.

    A missing measurement becomes null. It never becomes a substituted
    number."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def _write_outputs(
    cfg: Config,
    ranked: pd.DataFrame,
    recs: list[dict[str, Any]],
    train_report: dict | None,
    lessons: list[str],
    out_dir: Path,
    sites: CandidateSites | None = None,
    soil: dict[str, SoilProfile] | None = None,
) -> None:
    rec_by_zone = {r["zone"]: r for r in recs}
    soil = soil or {}
    has_sites = sites is not None and not sites.empty

    n_sites_col = ["n_sites"] if "n_sites" in ranked.columns else []
    csv_cols = [
        "rank", "zone", "score", "traffic_latest", "aqi_latest", "ndvi_latest",
        "traffic_pred", "aqi_pred", "ndvi_pred", "traffic_pred_delta", "aqi_pred_delta",
        "ndvi_pred_delta", "ndvi_slope", "plantable_space", *n_sites_col, "c_aqi_worsening",
        "c_traffic_worsening", "c_ndvi_decline", "c_low_green_cover", "c_plantable_space",
    ]
    csv_df = ranked[csv_cols].copy()
    csv_df["species"] = csv_df["zone"].map(
        lambda z: "; ".join(rec_by_zone[z]["species"]) if z in rec_by_zone else ""
    )
    csv_df["soil_ph"] = csv_df["zone"].map(lambda z: getattr(soil.get(z), "ph", None))
    csv_df["soil_texture"] = csv_df["zone"].map(
        lambda z: getattr(soil.get(z), "texture_class", None)
    )
    csv_df["soil_moisture"] = csv_df["zone"].map(
        lambda z: getattr(soil.get(z), "moisture", None)
    )
    csv_path = out_dir / "recommendations.csv"
    csv_df.to_csv(csv_path, index=False, float_format="%.4f")

    features = []
    geo_n = getattr(cfg.mcda, "geojson_n", 0) or len(ranked)
    for r in ranked.head(geo_n).itertuples():
        rec = rec_by_zone.get(r.zone, {})
        prof = soil.get(r.zone)
        try:
            geom = {"type": "Polygon", "coordinates": [cell_boundary_lonlat(r.zone)]}
        except Exception:  # grid isn't H3 (custom adapter without geometry)
            geom = None
        # 14 of the 146 cells are outside the NDVI source's coverage. Everything
        # derived from NDVI is therefore unmeasured for them, and must ship as
        # null — the non-finite ones already do via _json_safe, but theil_sen()
        # returns a REAL 0.0 for an empty series ("no trend estimable", its
        # legitimate default inside MCDA scoring), and that 0.0 reached the map
        # as a genuine reading of "flat NDVI trend". Same for plantable_source,
        # which named a proxy that was never computed for these cells.
        has_ndvi = math.isfinite(float(r.ndvi_latest))
        props = {
            "zone": r.zone,
            "rank": int(r.rank),
            # True only for the top_n cells the model actually reasoned about.
            "reasoned": r.zone in rec_by_zone,
            "priority_score": round(float(r.score), 4),
            "aqi_latest": round(float(r.aqi_latest), 1),
            "aqi_pred_delta": round(float(r.aqi_pred_delta), 1),
            "traffic_latest": round(float(r.traffic_latest), 1),
            "traffic_pred_delta": round(float(r.traffic_pred_delta), 1),
            "ndvi_latest": round(float(r.ndvi_latest), 3),
            "ndvi_trend_per_year": round(float(r.ndvi_slope) * 12, 4) if has_ndvi else None,
            # forecast yr-on-yr (yellow signal)
            "ndvi_pred_delta": round(float(r.ndvi_pred_delta), 4) if has_ndvi else None,
            "plantable_space": round(float(r.plantable_space), 2),
            "plantable_source": (
                ("bare_ground_sites" if has_sites else "proxy_1_minus_ndvi")
                if has_ndvi else None
            ),
            "n_sites": int(getattr(r, "n_sites", 0)),
            "species": rec.get("species", []),
            "justification": rec.get("justification", ""),
        }
        if prof is not None:
            props["soil"] = prof.as_dict()
        features.append({"type": "Feature", "geometry": geom, "properties": props})
    geojson_path = out_dir / "recommendations.geojson"
    geojson_path.write_text(
        json.dumps(_json_safe({"type": "FeatureCollection", "features": features}), indent=2),
        encoding="utf-8",
    )

    # planting_sites.geojson — the point-level deliverable: one pin per bare
    # patch inside each top zone, with soil + matched species. Only written
    # when a candidate-sites layer is supplied.
    sites_path = None
    if has_sites:
        pt_features = []
        for r in ranked.head(cfg.mcda.top_n).itertuples():
            rec = rec_by_zone.get(r.zone, {})
            prof = soil.get(r.zone)
            for s in sites.sites_for_zone(r.zone, cfg.sites.max_sites_per_zone):
                sp = {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                    "properties": {
                        "zone": r.zone,
                        "zone_rank": int(r.rank),
                        "priority_score": round(float(r.score), 4),
                        "patch_area_m2": s["patch_area_m2"],
                        "confidence": s["confidence"],
                        "ndvi_latest": round(float(r.ndvi_latest), 3),
                        "species": rec.get("species", []),
                        "positional_accuracy_m": 5,
                        "note": "confirm exact pit on a sub-metre basemap before digging",
                    },
                }
                if prof is not None:
                    sp["properties"]["soil"] = prof.as_dict()
                pt_features.append(sp)
        sites_path = out_dir / "planting_sites.geojson"
        sites_path.write_text(
            json.dumps(_json_safe({"type": "FeatureCollection", "features": pt_features}), indent=2),
            encoding="utf-8",
        )
        log.info("wrote %s (%d candidate points)", sites_path.name, len(pt_features))

    brief = [
        f"GREENGRID PLANTING BRIEF — {cfg.city.name}",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Grid: H3 resolution {cfg.grid.h3_resolution} | "
        f"Horizon: {cfg.training.horizon_months} months | "
        f"Zones ranked: {len(ranked)} | Top listed: {cfg.mcda.top_n}",
        "",
    ]
    if train_report:
        verdict = "beat" if train_report.get("memory_helped") else "did NOT beat"
        brief += [
            f"Training: {train_report['iterations']} backtest iterations at horizon "
            f"{train_report['horizon']}. MAE — traffic {train_report['mae']['traffic']:.1f}, "
            f"AQI {train_report['mae']['aqi']:.1f}, NDVI {train_report['mae']['ndvi']:.3f}.",
            f"With its memory the model {verdict} a memory-free naive baseline on the "
            f"same tasks (skill first half {train_report['skill_vs_baseline_first_half']:+.3f} "
            f"-> second half {train_report['skill_vs_baseline_second_half']:+.3f}; "
            f"raw error {train_report['norm_error_first_half']:.3f} -> "
            f"{train_report['norm_error_second_half']:.3f}).",
            "",
        ]
    brief.append("TOP PRIORITY ZONES")
    brief.append("=" * 60)
    for r in ranked.head(cfg.mcda.top_n).itertuples():
        rec = rec_by_zone.get(r.zone, {})
        prof = soil.get(r.zone)
        lat, lon = (float("nan"), float("nan"))
        try:
            lat, lon = cell_center(r.zone)
        except Exception:
            pass
        brief += [
            f"#{int(r.rank)}  {r.zone}  (lat {lat:.4f}, lon {lon:.4f})  "
            f"score {float(r.score):.3f}",
            f"    AQI {float(r.aqi_latest):.0f} ({float(r.aqi_pred_delta):+.0f} predicted yr-on-yr), "
            f"traffic {float(r.traffic_latest):.0f} ({float(r.traffic_pred_delta):+.0f} predicted yr-on-yr), "
            f"NDVI {float(r.ndvi_latest):.2f} ({float(r.ndvi_slope) * 12:+.3f}/yr)",
        ]
        if prof is not None:
            brief.append(f"    Soil: {prof.summary()}")
        brief += [
            f"    Plant: {', '.join(rec.get('species', [])) or '-'}",
            f"    Why: {rec.get('justification', '-')}",
        ]
        if has_sites:
            zone_sites = sites.sites_for_zone(r.zone, cfg.sites.max_sites_per_zone)
            if zone_sites:
                brief.append(
                    f"    Planting sites ({len(zone_sites)} bare patches, "
                    "confirm on sub-metre basemap before digging):"
                )
                for i, s in enumerate(zone_sites[:5], 1):
                    brief.append(
                        f"      {i}. lat {s['lat']:.5f}, lon {s['lon']:.5f}  "
                        f"(~{s['patch_area_m2']:.0f} m²)"
                    )
            else:
                brief.append("    Planting sites: none detected — survey on foot")
        brief.append("")
    if lessons:
        brief.append("LESSONS THE SYSTEM LEARNED WHILE BACKTESTING")
        brief += [f"  - {l}" for l in lessons[:8]]
        brief.append("")
    brief += ["CAVEATS"]
    if has_sites:
        brief += [
            "  - Planting-site coordinates come from 10 m bare-ground land cover:",
            "    accurate to ~±5 m (a plot, not an individual pit). Confirm each spot",
            "    visually on a sub-metre basemap (Esri World Imagery) and on the ground",
            "    before digging — ownership, utilities and access are not modelled.",
        ]
    else:
        brief += [
            "  - 'Plantable space' is a placeholder proxy (1 - NDVI); supply a",
            "    bare-ground sites CSV (sites.candidates_csv) for real open-space data.",
        ]
    brief += [
        "  - Soil pH/texture are SoilGrids MODELLED estimates (250 m); moisture is",
        "    from SMAP. Good for screening, not a substitute for a soil lab test.",
        "  - Species traits (incl. soil tolerances) come from a draft table; verify",
        "    against a horticulture source (see greenplan/reasoning/species.py).",
    ]
    (out_dir / "planting_brief.txt").write_text("\n".join(brief), encoding="utf-8")
    outs = [csv_path.name, geojson_path.name, "planting_brief.txt"]
    if sites_path is not None:
        outs.append(sites_path.name)
    log.info("wrote %s", ", ".join(outs))


# ---------------------------------------------------------------------------
# Long-range UNVALIDATED projection
# ---------------------------------------------------------------------------


def project(
    cfg: Config, panel: pd.DataFrame, model: Any, n_months: int, out_dir: Path
) -> Path:
    """Damped Theil-Sen extrapolation far past the data. Explicitly
    UNVALIDATED: no observation exists to score it against."""
    feats = zone_features(panel, trend_window=cfg.training.context_window)
    last_month = int(panel["month"].max())
    tau = 120.0  # damping: trend contribution saturates over ~a decade
    rows = []
    for r in feats.itertuples():
        for step in range(12, n_months + 1, 12):
            row: dict[str, Any] = {
                "zone": r.zone,
                "months_ahead": step,
                "years_ahead": step // 12,
                "month_index": last_month + step,
                "validated": False,
            }
            for m in METRICS:
                latest = getattr(r, f"{m}_latest")
                slope = getattr(r, f"{m}_slope")
                drift = slope * tau * (1.0 - np.exp(-step / tau))
                row[m] = float(np.clip(latest + drift, *BOUNDS[m]))
            rows.append(row)
    proj = pd.DataFrame(rows)

    sample = (
        proj[proj["months_ahead"] == proj["months_ahead"].max()]
        .head(8)[["zone", "traffic", "aqi", "ndvi"]]
        .round(2)
        .to_dict("records")
    )
    try:
        note = model.project_note(cfg.city.name, n_months, sample)
    except Exception as exc:
        log.warning("projection note failed: %s", exc)
        note = "Model commentary unavailable."

    path = out_dir / f"projection_UNVALIDATED_{n_months}mo.csv"
    header = (
        f"# UNVALIDATED LONG-RANGE PROJECTION — {cfg.city.name}, {n_months} months "
        f"(~{n_months // 12} years) past the last observed month.\n"
        "# There is no real data this far out, so these numbers CANNOT be checked\n"
        "# against reality (unlike the backtested horizon). Damped Theil-Sen trend\n"
        "# extrapolation only; seasonality, policy and land-use change not modeled.\n"
        f"# Model note: {note}\n"
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(header)
        proj.to_csv(f, index=False, float_format="%.3f")
    log.info("wrote %s", path.name)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    cfg: Config,
    *,
    mock: bool = False,
    do_recommend: bool = False,
    project_months: int | None = None,
) -> dict[str, Any]:
    np.random.seed(cfg.run.seed)
    panel = load_panel(cfg, mock)
    n_months = int(panel["month"].max()) - int(panel["month"].min()) + 1
    max_h = max_validatable_horizon(panel, cfg.training.min_history_months)
    log.info(
        "%d months of history -> largest honestly validatable horizon: %d months "
        "(requested horizon: %d)",
        n_months, max_h, cfg.training.horizon_months,
    )

    model = build_model(cfg.model, mock)
    memory = MemoryStore(cfg.resolve(cfg.training.memory_path))
    report = train(cfg, panel, model, memory)

    summary: dict[str, Any] = {
        "city": cfg.city.name,
        "zones": int(panel["zone"].nunique()),
        "months_of_history": n_months,
        "max_validatable_horizon": max_h,
        "memory_records": len(memory),
        "train_report": report,
    }

    if do_recommend:
        result = recommend(cfg, panel, model, memory, report)
        summary["outputs_dir"] = result["outputs_dir"]
        summary["top_zones"] = [
            {"zone": r["zone"], "species": r["species"]} for r in result["recommendations"][:3]
        ]

    if project_months:
        out_dir = cfg.resolve(cfg.run.outputs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary["projection_file"] = str(project(cfg, panel, model, project_months, out_dir))

    return summary
