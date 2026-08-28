"""Local HTTP bridge that puts the *real* trainable GreenGrid engine behind the
Front_End.html map — no framework, stdlib only, runs on the same venv.

On startup it loads the city panel, builds the model (offline MockModel unless a
provider key is set), runs the backtest/memory TRAINING loop (this is the
"trained on all the data" step), and caches the ranked zones + recommendations.

Endpoints (all CORS-open so a file:// page can call them):
  GET  /api/health      -> engine status, city, model, training skill
  GET  /api/zones       -> recommendations.geojson (trained-engine top zones)
  POST /api/recommend   -> body {lat,lon,aqi,ndvi|green,plantable}; returns the
                           engine's species pick + justification for that point,
                           using the curated species KB + soil/pollution logic.

The frontend keeps its API-key slots empty and separate; this server needs no
key to run (mock/offline). Set NVIDIA_API_KEY to train/predict with live NVIDIA
reasoning instead — the per-click species selection stays fast + deterministic.

Run:  python -m greenplan.server --config config/city.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import numpy as np


def _clean(obj: Any) -> Any:
    """Recursively replace NaN/Inf floats with None so output is valid JSON
    (json.dumps emits bare NaN/Infinity, which browsers' JSON.parse reject)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj

from .config import apply_env_overrides, load_config
from .engine import load_panel, recommend, train
from .features.h3grid import cell_boundary_lonlat, cell_center, latlng_to_cell
from .features.soil import load_soil, species_soil_ok
from .reasoning import i18n
from .reasoning.assistant import Assistant
from .reasoning.client import MockModel, build_model
from .reasoning.species import SPECIES_KB, kb_by_name, validate_selection
from .training.memory import MemoryStore

log = logging.getLogger(__name__)


class Engine:
    """Loads + trains the pipeline once, then answers per-point queries."""

    def __init__(self, config_path: str) -> None:
        self.cfg = load_config(config_path)
        # MODEL_PROVIDER lets a deployment (the Dockerfile ships
        # `ENV MODEL_PROVIDER=mock`) pick the provider without editing the
        # tracked YAML. Shared with the CLI so both entry points behave
        # identically — see config.apply_env_overrides for the reasoning.
        apply_env_overrides(self.cfg)
        # Two independent "mock" decisions:
        #  * adapters ALWAYS follow config (so real csv: streams like AQI are
        #    honored) — pass mock=False to load_panel.
        #  * the reasoning MODEL falls back to MockModel only when no LLM key is
        #    set, so the engine still runs fully offline.
        # `openvino` is a real provider that needs no key at all — it runs the
        # INT4 weights on this CPU — so it must not be mistaken for "no model
        # configured". Hosted providers still require their key before we trust
        # them; without one we fall back to the offline engine.
        self.mock_model = self.cfg.model.provider != "openvino" and not (
            os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        )
        self.kb = kb_by_name()
        # Per-click reasoning engine: the real LLM when a provider key is set,
        # the deterministic offline engine otherwise. This is the ONLY place an
        # LLM is used — it answers the handful of clicks a user actually reads.
        # _fallback catches rate-limits/timeouts so a click never hard-fails.
        self._picker = build_model(self.cfg.model, self.mock_model, strict=False)
        self._fallback = MockModel()
        # build_model may have degraded to the offline engine (runtime missing,
        # weights not fetched). health() must report what is ACTUALLY answering,
        # not what the config asked for, so re-derive the flag from the object.
        if isinstance(self._picker, MockModel):
            self.mock_model = True
        self.assistant = Assistant(self)
        self._build_and_train()

    def _build_and_train(self) -> None:
        cfg = self.cfg
        log.info("loading panel for %s (adapters from config; mock_model=%s)…",
                 cfg.city.name, self.mock_model)
        self.panel = load_panel(cfg, mock=False)  # honor configured adapters
        # Soil is loaded HERE, not only inside recommend(), because per-click
        # species matching needs it too. Without this the clicked-point pick
        # silently ignored pH and texture while the README said it did not.
        self.soil = load_soil(
            cfg.resolve(cfg.soil.soilgrids_csv) if cfg.soil.soilgrids_csv else None,
            cfg.resolve(cfg.soil.moisture_csv) if cfg.soil.moisture_csv else None,
            cfg.grid.h3_resolution,
        )
        # Batch forecasting/training is ALWAYS the offline deterministic model:
        # predict_future runs once per zone (146 cells) plus ~120 training calls.
        # Routing that through the LLM would mean ~270 requests at startup, which
        # this account answers with HTTP 429. The LLM is reserved for per-click
        # reasoning (self._picker) — the part a user actually reads.
        self.model = build_model(cfg.model, True)
        self.memory = MemoryStore(cfg.resolve(cfg.training.memory_path))
        log.info("training (backtest + memory) …")
        self.train_report = train(cfg, self.panel, self.model, self.memory)
        log.info("running recommendation pass …")
        result = recommend(cfg, self.panel, self.model, self.memory, self.train_report)
        self.ranked = result["ranked"]
        self.recommendations = result["recommendations"]
        # cache the trained-engine geojson for the map overlay
        geo_path = cfg.resolve(cfg.run.outputs_dir) / "recommendations.geojson"
        try:
            self.zones_geojson = json.loads(geo_path.read_text(encoding="utf-8"))
        except Exception:
            self.zones_geojson = {"type": "FeatureCollection", "features": []}
        self.lessons = [
            r.get("lesson", "").strip()
            for r in reversed(self.memory.records)
            if r.get("lesson", "").strip()
        ][:12]
        self._build_greenloss()
        log.info(
            "engine ready: %d zones, %d memory records, model=%s | greenloss %s",
            len(self.ranked), len(self.memory), type(self.model).__name__,
            self.greenloss["count"],
        )

    # Zone status thresholds on real NDVI (documented, not magic):
    #   GREEN_NDVI  — at/above this the cell is currently vegetated
    #   LOSS_DELTA  — forecast yr-on-yr NDVI change at/below this = meaningful loss
    GREEN_NDVI = 0.28
    LOSS_DELTA = -0.02

    def _build_greenloss(self) -> None:
        """All ranked cells as H3 polygons tagged green / yellow / red, from the
        trained engine's forecast. YELLOW (currently green but predicted to
        decline) is the whole point — it comes from ndvi_pred_delta, not a
        snapshot. GREEN/RED here are the engine's NDVI-based view; the browser
        refines them per-pixel (VARI) and with OSM (buildings/roads/water)."""
        feats = []
        for r in self.ranked.itertuples():
            nl, dd = float(r.ndvi_latest), float(r.ndvi_pred_delta)
            if not math.isfinite(nl) or not math.isfinite(float(r.score)):
                continue  # cell has no real NDVI coverage — omit, don't guess a colour
            if nl >= self.GREEN_NDVI and dd <= self.LOSS_DELTA:
                status = "yellow"          # has green, forecast to lose it
            elif nl >= self.GREEN_NDVI:
                status = "green"           # vegetated, stable/improving
            else:
                status = "red"             # low vegetation (plantable candidate)
            try:
                geom = {"type": "Polygon", "coordinates": [cell_boundary_lonlat(r.zone)]}
            except Exception:
                geom = None
            feats.append({
                "type": "Feature", "geometry": geom,
                "properties": {
                    "zone": r.zone, "rank": int(r.rank),
                    "score": round(float(r.score), 4),
                    "ndvi_latest": round(nl, 3),
                    "ndvi_slope_per_year": round(float(r.ndvi_slope) * 12, 4),
                    "ndvi_pred_delta": round(dd, 4),
                    "status": status,
                },
            })
        count = {s: sum(1 for f in feats if f["properties"]["status"] == s)
                 for s in ("green", "yellow", "red")}
        self.greenloss = {
            "type": "FeatureCollection",
            "features": feats,
            "thresholds": {"green_ndvi": self.GREEN_NDVI, "loss_delta": self.LOSS_DELTA},
            "count": count,
            "total_zones": len(feats),
        }

    def _reasoning_label(self) -> str:
        """What is actually answering clicks — named honestly. The old label
        said "nvidia-llm" whatever the provider was, which is wrong the moment
        inference is local."""
        if self.mock_model:
            return "offline-engine"
        return {
            "openvino": "openvino-local",
            "nvidia": "nvidia-llm",
            "openrouter": "openrouter-llm",
        }.get(self.cfg.model.provider, self.cfg.model.provider)

    def health(self) -> dict[str, Any]:
        tr = self.train_report or {}
        return {
            "ok": True,
            "city": self.cfg.city.name,
            # Two models, deliberately. `forecast_model` does the 146-zone
            # batch numerically; `reasoning_model` answers the clicks a human
            # reads. Reporting only one of them misrepresents the system.
            "model": type(self._picker).__name__,
            "forecast_model": type(self.model).__name__,
            "mock_model": self.mock_model,
            "aqi_source": self.cfg.adapters.aqi,
            "zones": int(len(self.ranked)),
            "memory_records": int(len(self.memory)),
            "trained": int(len(self.memory)) > 0,
            "retrained_this_session": self.train_report is not None,
            "memory_helped": tr.get("memory_helped"),
            "ndvi_source": self.cfg.adapters.green_cover,
            "reasoning": self._reasoning_label(),
            "reasoning_model": (
                None if self.mock_model
                else self.cfg.model.model_dir if self.cfg.model.provider == "openvino"
                else self.cfg.model.name
            ),
            "reasoning_device": (
                self.cfg.model.device if self.cfg.model.provider == "openvino" else None
            ),
            "greenloss": self.greenloss["count"],
            "assistant": True,
            "species_kb": len(SPECIES_KB),
            "soil_cells": int(len(self.soil)),
            "languages": [l["code"] for l in i18n.available()],
        }

    def recommend_point(self, body: dict[str, Any]) -> dict[str, Any]:
        """Species pick + justification for one clicked point, from the engine."""
        aqi = _num(body.get("aqi"), 90.0)
        # accept ndvi (0..1) directly, or a 0..100 "green"/canopy percentage
        ndvi = body.get("ndvi")
        if ndvi is None and body.get("green") is not None:
            ndvi = _num(body.get("green"), 40.0) / 100.0
        if ndvi is None and body.get("vegPct") is not None:
            ndvi = _num(body.get("vegPct"), 40.0) / 100.0
        ndvi = float(np.clip(_num(ndvi, 0.4), 0.0, 1.0))
        plantable = float(np.clip(_num(body.get("plantable"), 1.0 - ndvi), 0.0, 1.0))
        lat, lon = body.get("lat"), body.get("lon")
        try:
            zone = latlng_to_cell(float(lat), float(lon), self.cfg.grid.h3_resolution)
        except Exception:
            zone = "point"

        # crude single-point MCDA-ish score just for the justification text
        score = float(np.clip(0.3 * (aqi / 300.0) + 0.35 * (1.0 - ndvi) + 0.1 * plantable, 0, 1))
        # Real values wherever the trained panel covers this cell. The forecast
        # delta and the NDVI slope are the two things a browser click cannot
        # know; pinning them at 0.0 made every justification read "predicted
        # AQI change +0.0" no matter where the user clicked. Soil likewise:
        # it was hardcoded None while the README said species respected pH.
        prof = self.soil.get(zone)
        hist = self._panel_row(zone)
        row = {
            "zone": zone, "score": round(score, 3),
            "aqi_latest": round(aqi, 1),
            "aqi_pred_delta": hist.get("aqi_pred_delta") or 0.0,
            "traffic_latest": 0.0, "traffic_pred_delta": 0.0,
            "ndvi_latest": round(ndvi, 3),
            "ndvi_slope": hist.get("ndvi_slope") or 0.0,
            "plantable_space": round(plantable, 2),
            "soil": prof.as_dict() if prof is not None else None,
        }
        used = self._reasoning_label()
        try:
            recs = self._picker.recommend([row], self.lessons)
        except Exception as exc:  # 429 / timeout / invalid JSON from the provider
            log.warning("LLM failed (%s); using offline engine", exc)
            recs = self._fallback.recommend([row], self.lessons)
            used = "offline-engine (LLM unavailable)"
        names = validate_selection(recs[0]["species"]) if recs else []
        species = [self._species_card(n) for n in names]
        return {
            "source": used,
            "zone": zone,
            "justification": recs[0]["justification"] if recs else "",
            "species": species,
            "score": row["score"],
        }

    # ------------------------------------------------------------------
    # Assistant-facing surface
    # ------------------------------------------------------------------
    # Everything the natural-language layer is allowed to read goes through
    # these methods. They return JSON-safe dicts and never raise: the
    # assistant must be able to say "I don't have that" rather than 500.

    def reasoning_label(self) -> str:
        return self._reasoning_label()

    def n_zones(self) -> int:
        return int(len(self.ranked))

    def _panel_row(self, zone: str) -> dict[str, Any]:
        """The trained panel's row for one H3 cell, or {} when the cell is
        outside the modelled city. A click in Mumbai is a legitimate action;
        it just has no 42-month history behind it."""
        try:
            hit = self.ranked[self.ranked["zone"] == zone]
            if hit.empty:
                return {}
            r = hit.iloc[0]
        except Exception:
            return {}

        def g(name: str, digits: int = 4):
            try:
                v = float(r[name])
            except Exception:
                return None
            return round(v, digits) if math.isfinite(v) else None

        slope = g("ndvi_slope", 6)
        return {
            "zone": zone,
            "rank": int(r["rank"]),
            "score": g("score"),
            "aqi_latest": g("aqi_latest", 1),
            "aqi_pred_delta": g("aqi_pred_delta", 1),
            "ndvi_latest": g("ndvi_latest", 3),
            "ndvi_slope": slope,
            "ndvi_trend_per_year": round(slope * 12, 4) if slope is not None else None,
            "ndvi_pred_delta": g("ndvi_pred_delta"),
            "plantable_space": g("plantable_space", 2),
        }

    def cell_report(self, lat: float, lon: float):
        """What the trained engine knows about the cell containing (lat, lon)."""
        try:
            zone = latlng_to_cell(float(lat), float(lon), self.cfg.grid.h3_resolution)
        except Exception:
            return None
        row = self._panel_row(zone)
        if not row:
            return None
        rec = next((r for r in self.recommendations if r["zone"] == zone), None)
        rank, n = row.get("rank"), self.n_zones()
        parts = []
        if rank:
            parts.append("This point falls in H3 cell `%s`, which the engine ranks "
                         "**#%d of %d** for planting priority" % (zone, rank, n))
            if row.get("score") is not None:
                parts.append(" (score %.3f)" % row["score"])
            parts.append(".")
        if row.get("ndvi_trend_per_year") is not None:
            parts.append(" Its MODIS NDVI trend over 42 months is **%+.4f/yr**"
                         % row["ndvi_trend_per_year"])
            if row.get("aqi_pred_delta") is not None:
                parts.append(", with forecast AQI change **%+.1f**" % row["aqi_pred_delta"])
            parts.append(".")
        row["sentence"] = "".join(parts) or ("This point is in H3 cell `%s`." % zone)
        if rec:
            row["species"] = rec.get("species", [])
            row["justification"] = rec.get("justification", "")
        return row

    def soil_report(self, lat: float, lon: float):
        try:
            zone = latlng_to_cell(float(lat), float(lon), self.cfg.grid.h3_resolution)
        except Exception:
            return None
        prof = self.soil.get(zone)
        if prof is None:
            return None
        return {
            "zone": zone, "ph": prof.ph, "ph_class": prof.ph_class,
            "texture": prof.texture_class, "texture_simple": prof.texture_simple,
            "sand": prof.sand, "silt": prof.silt, "clay": prof.clay,
            "organic_carbon": prof.soc, "nitrogen": prof.nitrogen,
            "moisture": prof.moisture,
        }

    def top_cells(self, n: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        rec_by_zone = {r["zone"]: r for r in self.recommendations}
        for r in self.ranked.head(max(1, min(100, int(n)))).itertuples():
            try:
                if not math.isfinite(float(r.score)):
                    continue
                lat, lon = cell_center(r.zone)
            except Exception:
                lat = lon = None
            rec = rec_by_zone.get(r.zone, {})
            out.append({
                "zone": r.zone, "rank": int(r.rank), "score": float(r.score),
                "lat": lat, "lon": lon,
                "aqi_latest": float(r.aqi_latest),
                "aqi_pred_delta": float(r.aqi_pred_delta),
                "ndvi_latest": float(r.ndvi_latest),
                "ndvi_trend_per_year": float(r.ndvi_slope) * 12,
                "plantable_space": float(r.plantable_space),
                "species": rec.get("species", []),
                "justification": rec.get("justification", ""),
            })
        return out

    def bare_cells(self, lat=None, lon=None, n=5) -> list[dict[str, Any]]:
        """Ranked cells with the most plantable space, nearest first.

        This is the "where is there actually room" question, which is not the
        same as the priority ranking: a cell can be top-ranked on air quality
        and have nowhere to put a tree."""
        rows = []
        for r in self.ranked.itertuples():
            try:
                ps = float(r.plantable_space)
                if not math.isfinite(ps) or ps < 0.45:
                    continue
                clat, clon = cell_center(r.zone)
            except Exception:
                continue
            d = None
            if lat is not None and lon is not None:
                # Equirectangular is accurate well past the 100 km2 perimeter.
                dx = (clon - float(lon)) * 111.32 * math.cos(math.radians(float(lat)))
                dy = (clat - float(lat)) * 110.57
                d = math.hypot(dx, dy)
            rows.append({
                "zone": r.zone, "rank": int(r.rank),
                "plantable_space": ps,
                "ndvi_latest": float(r.ndvi_latest),
                "score": float(r.score),
                "lat": clat, "lon": clon,
                "dist_km": d if d is not None else 0.0,
            })
        rows.sort(key=lambda x: (x["dist_km"], -x["plantable_space"]))
        return rows[:max(1, min(20, int(n)))]

    # -- species matching -------------------------------------------------
    #
    # The piece the browser cannot do on its own: the page's 16-row seed list
    # carries no soil column at all. Here the full KB is filtered against the
    # cell's real SoilGrids pH and texture first, then scored on the
    # conditions actually measured at the point.

    _GOAL_CONTEXT = {
        "park": ("park",), "greenbelt": ("park", "peri-urban", "belt"),
        "riverfront": ("riverbank", "lakefront"),
        "community": ("park", "campus", "garden"),
        "campus": ("campus", "courtyard"),
        "avenue": ("avenue", "median", "street"),
        "residential": ("avenue", "compact", "home", "garden"),
        "industrial": ("peri-urban", "boundary", "screen"),
        "wetland": ("riverbank", "lakefront"),
    }

    # Phrases in a KB row's `context` that mean the species needs real room.
    # Planting one of these on a small urban plot is a decision to fell it, or
    # the wall next to it, inside twenty years.
    _NEEDS_ROOM = ("large park", "open ground", "wide avenue", "peri-urban")

    def match_species(self, lat=None, lon=None, aqi=None, canopy_pct=None,
                      rain_mm_yr=None, goal="park", limit=5,
                      area_m2=None, lang="en") -> dict[str, Any]:
        """Rank the KB for one location. Deterministic and explainable - every
        point of every score is attributable to a measured input."""
        prof = None
        if lat is not None and lon is not None:
            try:
                zone = latlng_to_cell(float(lat), float(lon), self.cfg.grid.h3_resolution)
                prof = self.soil.get(zone)
            except Exception:
                prof = None

        dirty = aqi is not None and aqi >= 120
        filthy = aqi is not None and aqi >= 180
        dry = rain_mm_yr is not None and rain_mm_yr < 750
        very_dry = rain_mm_yr is not None and rain_mm_yr < 400
        bare = canopy_pct is not None and canopy_pct < 12
        wants = self._GOAL_CONTEXT.get(goal, ("park",))
        # Plot-size bands. 2 ha is roughly where a Banyan or a Rain Tree stops
        # being a centrepiece and starts being a problem; 0.5 ha is where even
        # a large crown crowds everything else out.
        small = area_m2 is not None and area_m2 < 20000
        tiny = area_m2 is not None and area_m2 < 5000

        compatible = [sp for sp in SPECIES_KB if species_soil_ok(sp, prof)]
        soil_filtered = len(compatible) < len(SPECIES_KB)
        use_compatible = len(compatible) >= 6
        pool = compatible if use_compatible else SPECIES_KB
        soil_relaxed = soil_filtered and not use_compatible

        def score(sp):
            s, why = 0.0, []
            tol = {"high": 2.0, "medium": 1.0, "low": 0.0}[sp["pollution_tolerance"]]
            if filthy:
                s += tol * 2.5
                if tol >= 2.0:
                    why.append(i18n.t("why.tolerant_at", lang, aqi="%.0f" % aqi))
            elif dirty:
                s += tol * 1.8
                if tol >= 2.0:
                    why.append(i18n.t("why.high_tolerance", lang))
            else:
                s += tol * 0.8
            if very_dry and sp["water_need"] == "low":
                s += 2.5
                why.append(i18n.t("why.survives_rain", lang, mm="%.0f" % rain_mm_yr))
            elif dry and sp["water_need"] == "low":
                s += 1.6
                why.append(i18n.t("why.low_water", lang))
            elif dry and sp["water_need"] == "high":
                s -= 1.5
            if bare and sp["canopy"] == "large" and not small:
                s += 1.2
                why.append(i18n.t("why.large_crown", lang))
            if sp["native_status"] == "native":
                s += 0.8
                why.append(i18n.t("why.native", lang))
            ctxs = sp.get("context", "").lower()
            if any(w in ctxs for w in wants):
                s += 1.5
                why.append(i18n.t("why.belongs", lang,
                                  goal=i18n.t("goal." + goal, lang)))
            if "aggressive" in ctxs and goal in (
                "avenue", "residential", "campus", "community"
            ):
                s -= 2.5
                why.append(i18n.t("why.aggressive", lang))
            # Fit to the ground actually available. Only species whose own
            # KB row says they need open ground are penalised by plot size -
            # a large tree is not a problem on a hectare, a Banyan is.
            if small and any(w in ctxs for w in self._NEEDS_ROOM):
                s -= 3.0
                why.append(i18n.t("why.needs_room", lang))
            if tiny and sp["canopy"] == "large":
                s -= 2.5
            if prof is not None and prof.ph is not None and use_compatible:
                why.append(i18n.t("why.tolerates_ph", lang, ph="%.1f" % prof.ph))
            return s, why

        scored = sorted(((score(sp), sp) for sp in pool),
                        key=lambda t: t[0][0], reverse=True)
        species = []
        for (sc, why), sp in scored[:max(1, min(10, int(limit)))]:
            card = self._species_card(sp["common"])
            card["score"] = round(sc, 2)
            uniq = list(dict.fromkeys(why))[:3]
            card["why"] = (", ".join(uniq).capitalize() + "."
                           if uniq else i18n.t("why.default", lang))
            species.append(card)

        cond = []
        if aqi is not None:
            cond.append(i18n.t("match.cond_aqi", lang, aqi="%.0f" % aqi))
        if rain_mm_yr is not None:
            cond.append(i18n.t("match.cond_rain", lang, mm="{:,.0f}".format(rain_mm_yr)))
        if canopy_pct is not None:
            cond.append(i18n.t("match.cond_canopy", lang, pct="%.0f" % canopy_pct))
        if prof is not None and prof.ph is not None:
            cond.append(i18n.t("match.cond_soil", lang, ph="%.1f" % prof.ph)
                        + (" (%s)" % prof.texture_class if prof.texture_class else ""))
        if area_m2 is not None:
            cond.append(i18n.t("match.cond_plot", lang, ha="%.2f" % (area_m2 / 10000.0)))
        headline = (i18n.t("match.headline", lang, conditions=", ".join(cond))
                    if cond else i18n.t("match.headline_none", lang))

        caveats = []
        if prof is None:
            caveats.append(i18n.t("match.caveat_nosoil", lang))
        elif soil_relaxed:
            caveats.append(i18n.t("match.caveat_relaxed", lang))
        caveats.append(i18n.t("match.caveat_always", lang))
        return {"species": species, "headline": headline,
                "caveats": " ".join(caveats),
                "soil": prof.as_dict() if prof is not None else None}

    # -- layout planning ---------------------------------------------------

    def layout_plan(self, area_m2, goal, species, rain_mm_yr=None, aqi=None,
                    lang="en"):
        """Turn an area and a species shortlist into a placeable plan.

        Spacing follows mature crown width, not a round number: a Banyan at
        8 m centres is a plan to fell half of them in year fifteen."""
        area_m2 = max(200.0, float(area_m2))
        crown = {"small": 5.0, "medium": 8.0, "large": 12.0}

        # A planting plan is a structure, not a leaderboard. Take the best
        # available species in each canopy stratum so the result has shade
        # trees, a body, and an edge - which is also how it stays diverse
        # without diversity being bolted on afterwards.
        #   (large, medium, small) targets by plot size
        if area_m2 < 2000:
            target = (0, 1, 2)
        elif area_m2 < 10000:
            target = (1, 2, 1)
        elif area_m2 < 50000:
            target = (2, 2, 2)
        else:
            target = (3, 3, 2)
        if goal == "avenue":
            target = (1, 2, 0)            # a street wants uniform crowns, not an edge

        by_size = {"large": [], "medium": [], "small": []}
        for sp in species:
            by_size.setdefault(sp.get("canopy", "medium"), []).append(sp)

        picks, seen = [], set()
        for size, want in zip(("large", "medium", "small"), target):
            for sp in by_size.get(size, [])[:want]:
                if sp["common"] not in seen:
                    picks.append(sp)
                    seen.add(sp["common"])
        # A stratum can come back empty once soil and pollution filters have
        # run; backfill from the ranking rather than shipping a thin mix.
        for sp in species:
            if len(picks) >= sum(target):
                break
            if sp["common"] not in seen:
                picks.append(sp)
                seen.add(sp["common"])
        if not picks:
            picks = [{"common": "Neem", "canopy": "large"}]

        # Spacing follows the crowns actually chosen.
        avg = sum(crown.get(s.get("canopy", "medium"), 8.0) for s in picks) / len(picks)
        if goal == "avenue":
            avg = max(avg, 10.0)          # a row of street trees, not a thicket
        # Canopy must not exceed the plot: reserve ground for paths, water and
        # the open space that makes a park usable rather than a woodlot.
        share = 0.55 if goal in ("park", "community", "campus") else 0.75
        plantable = area_m2 * share
        n_trees = int(max(3, min(600, plantable / (avg ** 2))))

        # Weight the split by crown: one Neem occupies what four Amlas do, so
        # an even split by count would still be a plan dominated by big trees.
        weights = [1.0 / (crown.get(s.get("canopy", "medium"), 8.0) ** 2) for s in picks]
        wsum = sum(weights) or 1.0
        # ...but cap any one species, or pure area-weighting hands 45% of a
        # park to whichever small tree ranked first. A stand where one species
        # is a third of the trees is the monoculture risk this mix exists to
        # avoid, so no species exceeds `cap_share` while others can absorb it.
        cap_share = 0.30 if len(picks) >= 4 else 0.45
        cap = max(1, int(n_trees * cap_share))
        raw = [max(1, round(n_trees * w / wsum)) for w in weights]
        capped = [min(r, cap) for r in raw]
        # Redistribute what the cap freed onto the species still under it.
        spare = n_trees - sum(capped)
        guard = 0
        while spare > 0 and guard < 1000:
            room = [i for i, c in enumerate(capped) if c < cap]
            if not room:
                break
            for i in room:
                if spare <= 0:
                    break
                capped[i] += 1
                spare -= 1
            guard += 1
        mix, left = [], n_trees
        for i, sp in enumerate(picks):
            take = min(capped[i], left)
            if i == len(picks) - 1:
                take = min(left, max(take, 0))
            if take <= 0:
                continue
            mix.append({"species": sp["common"], "count": int(take)})
            left -= take

        elements = []
        if goal in ("park", "community", "campus"):
            elements += [
                {"id": "path_gravel", "name": "Gravel trail", "unit": "m2",
                 "qty": round(area_m2 * 0.06)},
                {"id": "meadow", "name": "Native grass meadow", "unit": "m2",
                 "qty": round(area_m2 * 0.15)},
                {"id": "shrub", "name": "Shrub massing", "unit": "m2",
                 "qty": round(area_m2 * 0.08)},
                {"id": "bench", "name": "Bench", "unit": "each",
                 "qty": max(3, int(area_m2 / 1200))},
                {"id": "light", "name": "Solar path light", "unit": "each",
                 "qty": max(2, int(area_m2 / 2000))},
                {"id": "tap", "name": "Drinking water point", "unit": "each",
                 "qty": max(1, int(area_m2 / 8000))},
                {"id": "compost", "name": "Composting bay", "unit": "each",
                 "qty": 1},
            ]
            # A park nobody can sit in the shade of is a lawn. Above half a
            # hectare there is room for shelter and, on a campus, for play.
            if area_m2 >= 5000:
                elements.append({"id": "gazebo", "name": "Shade pavilion",
                                 "unit": "each", "qty": 1})
            if goal in ("campus", "community") and area_m2 >= 4000:
                elements.append({"id": "play", "name": "Play equipment",
                                 "unit": "each", "qty": 1})
        if goal in ("industrial", "avenue", "greenbelt"):
            elements.append({"id": "hedge", "name": "Hedge screen", "unit": "m",
                             "qty": round(2 * math.sqrt(area_m2))})
        if goal in ("riverfront", "wetland"):
            elements.append({"id": "swale", "name": "Bioswale", "unit": "m",
                             "qty": round(math.sqrt(area_m2))})
        if rain_mm_yr is not None and rain_mm_yr < 750:
            elements += [
                {"id": "drip", "name": "Drip irrigation", "unit": "m2",
                 "qty": round(plantable * 0.5)},
                {"id": "mulch", "name": "Mulched planting bed", "unit": "m2",
                 "qty": round(plantable * 0.25)},
            ]
        if area_m2 >= 300:
            elements.append({"id": "rwh", "name": "Rainwater recharge pit",
                             "unit": "each", "qty": max(1, int(area_m2 / 5000))})

        why = [i18n.t("plan.spacing", lang, spacing="%.0f" % avg),
               i18n.t("plan.mix", lang, n=len(mix))]
        if goal in ("park", "community", "campus"):
            why.append(i18n.t("plan.open", lang))
        if rain_mm_yr is not None and rain_mm_yr < 750:
            why.append(i18n.t("plan.drip", lang, mm="{:,.0f}".format(rain_mm_yr)))
        if aqi is not None and aqi >= 120:
            why.append(i18n.t("plan.tolerance", lang, aqi="%.0f" % aqi))
        if area_m2 >= 300:
            why.append(i18n.t("plan.recharge", lang))

        return {"n_trees": sum(m["count"] for m in mix), "spacing_m": round(avg, 1),
                "mix": mix, "elements": elements, "rationale": " ".join(why),
                "plantable_m2": round(plantable)}

    def _species_card(self, name: str) -> dict[str, Any]:
        k = self.kb.get(name, {})
        return {
            "common": name,
            "botanical": k.get("botanical", ""),
            "native_status": k.get("native_status", ""),
            "canopy": k.get("canopy", ""),
            "pollution_tolerance": k.get("pollution_tolerance", ""),
            "water_need": k.get("water_need", ""),
            "context": k.get("context", ""),
            "soil_ph": k.get("soil_ph", ""),
        }


def _num(v: Any, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GreenGridEngine/1.0"

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, obj: Any, code: int = 200) -> None:
            payload = json.dumps(_clean(obj)).encode("utf-8")
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # noqa: N802 (stdlib naming)
            self.send_response(204)
            self._cors()
            self.end_headers()

        # --- static files ---------------------------------------------------
        # The studio and the engine ship from ONE origin. That is not a
        # convenience: same-origin means the page calls /api/... with no CORS
        # negotiation and no hard-coded host, so the whole product survives
        # being moved behind a tunnel, a reverse proxy or a different port
        # without a rebuild.
        STATIC = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            # index.html loads this unconditionally. It lives in web/ and is
            # copied into dist/ by scripts/build_static.py, so without this
            # route a locally-served page 404s on it and logs a console error
            # that looks like a fault but is not.
            #
            # Serving it here is not merely cosmetic: it means that if this
            # engine dies mid-session, the page falls back to the in-page
            # planner instead of going dead, exactly as the static build does.
            "/gv-engine.js": ("web/gv-engine.js", "text/javascript; charset=utf-8"),
        }
        # Directories a page is allowed to read from, relative to the repo root.
        STATIC_DIRS = ("outputs", "data")
        # Answered inline so a browser's automatic request does not fill the
        # log with 404s that look like a real failure during a demo.
        NO_CONTENT = ("/favicon.ico", "/apple-touch-icon.png",
                      "/apple-touch-icon-precomposed.png")

        def _send_file(self, rel: str, ctype: str) -> bool:
            root = Path(__file__).resolve().parent.parent
            target = (root / rel).resolve()
            # Refuse anything that escapes the repo root, whatever the path said.
            # This has to be a real DIRECTORY-BOUNDARY test, not a text prefix:
            # str.startswith compares characters, so with the repo at
            # .../Green-Vision a sibling folder named .../Green-Vision-secrets
            # matched the prefix and sailed through — `/data/../../Green-Vision-
            # secrets/key.txt` resolved outside the repo and was served.
            # is_relative_to compares path COMPONENTS, so a sibling never counts.
            if not target.is_relative_to(root) or not target.is_file():
                return False
            body = target.read_bytes()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]

            if path in self.NO_CONTENT:
                self.send_response(204)
                self._cors()
                self.end_headers()
            elif path.startswith("/api/health"):
                self._json(engine.health())
            elif path.startswith("/api/greenloss"):
                self._json(engine.greenloss)
            elif path.startswith("/api/cells"):
                # Every ranked cell, numbers only - what the map needs to draw
                # the whole city rather than the top ten.
                qs = parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
                try:
                    n = int((qs.get("n") or ["0"])[0])
                except ValueError:
                    n = 0
                self._json({"cells": engine.top_cells(n or engine.n_zones()),
                            "total": engine.n_zones()})
            elif path.startswith("/api/languages"):
                self._json({"languages": i18n.available()})
            elif path.startswith("/api/zones"):
                self._json(engine.zones_geojson)
            elif path in self.STATIC:
                rel, ctype = self.STATIC[path]
                if not self._send_file(rel, ctype):
                    self._json({"error": f"{rel} not found on disk"}, 404)
            elif path.lstrip("/").split("/", 1)[0] in self.STATIC_DIRS:
                rel = path.lstrip("/")
                ctype = (
                    "application/geo+json" if rel.endswith(".geojson")
                    else "text/csv; charset=utf-8" if rel.endswith(".csv")
                    else "application/json" if rel.endswith(".json")
                    else "application/octet-stream"
                )
                if not self._send_file(rel, ctype):
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)

        # Body cap: a browser has no reason to post more than this, and an
        # unbounded read on a public-ish port is how a stdlib server dies.
        MAX_BODY = 1 << 20  # 1 MiB

        def _body(self) -> dict[str, Any]:
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                n = 0
            if n <= 0:
                return {}
            if n > self.MAX_BODY:
                raise ValueError("request body too large")
            obj = json.loads(self.rfile.read(n) or b"{}")
            return obj if isinstance(obj, dict) else {}

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            routes = {
                "/api/recommend": lambda b: engine.recommend_point(b),
                "/api/species": lambda b: engine.match_species(
                    lat=b.get("lat"), lon=b.get("lon"), aqi=b.get("aqi"),
                    canopy_pct=b.get("canopy_pct"), rain_mm_yr=b.get("rain_mm_yr"),
                    goal=str(b.get("goal") or "park"), limit=int(b.get("limit") or 5),
                    lang=i18n.normalise(b.get("lang")),
                ),
                "/api/assistant": lambda b: engine.assistant.handle(
                    str(b.get("message") or ""), b.get("context"), b.get("lang"),
                ),
            }
            fn = next((f for pre, f in routes.items() if path.startswith(pre)), None)
            if fn is None:
                self._json({"error": "not found"}, 404)
                return
            try:
                self._json(fn(self._body()))
            except Exception as exc:  # keep the server alive on any bad request
                log.warning("%s failed: %s", path, exc)
                self._json({"error": str(exc)}, 400)

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter logs
            log.info("%s - %s", self.address_string(), fmt % args)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description="GreenGrid local engine server")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    engine = Engine(args.config)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    log.info("GreenGrid engine serving on http://%s:%d  (Ctrl+C to stop)", args.host, args.port)
    log.info("  GET  /api/health  /api/zones  /api/greenloss  /api/cells")
    log.info("  POST /api/recommend  /api/species  /api/assistant")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
        httpd.server_close()


if __name__ == "__main__":
    main()
