"""Bake the trained engine down to static JSON for a serverless deploy.

Why this exists
---------------
`greenplan.server` is Python: pandas, numpy and h3 hold the 42-month panel and
the MCDA ranking. Cloudflare Pages serves files; Workers run JS/WASM. Neither
will run that stack.

But almost nothing the browser asks the engine for is actually *dynamic*. For a
fixed city the ranking, the forecast, the soil table and the species knowledge
base are constants — they change when you re-run training on new data, not when
a visitor clicks. So we compute them once, here, and ship them as files.

The one genuinely dynamic part is the assistant, and it is deterministic: no
model, no weights, just intent matching and planning. That is ported to
JavaScript in `web/gv-engine.js` and kept honest by `scripts/parity_check.py`,
which runs the same corpus through this Python and that JavaScript and diffs
the results.

What ships
----------
    dist/
      index.html                 the studio, unchanged
      data/i18n/*.json           interface + assistant strings (5 languages)
      engine/zones.geojson       146 ranked cells: score, species, justification
      engine/greenloss.json      the same cells as polygons, green/amber/red
      engine/cells.json          the ranked panel as rows
      engine/soil.json           SoilGrids pH + texture per cell
      engine/species.json        the 30-species knowledge base
      engine/meta.json           city, counts, thresholds, when it was baked
      gv-engine.js               the client-side engine + assistant

Point-in-polygon over 146 hexagons replaces h3 in the browser, so no H3
library is needed client-side — the polygons are already in greenloss.json.

Run:
    .venv/Scripts/python scripts/build_static.py --config config/city.yaml
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from greenplan.features.h3grid import cell_center  # noqa: E402
from greenplan.reasoning.species import SPECIES_KB  # noqa: E402
from greenplan.server import Engine  # noqa: E402

log = logging.getLogger("build_static")


def _finite(v):
    """JSON has no NaN. Anything not finite becomes null, never a guess."""
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _json_safe(obj):
    """Recursively replace non-finite floats with None.

    Belt and braces: engine.py now writes clean GeoJSON, but this build is
    the thing a public deploy is cut from, and a single bare NaN makes the
    WHOLE document unparseable in a browser — JSON.parse is all-or-nothing.
    Cheap insurance against a regression upstream."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    # separators: no wasted bytes; ensure_ascii=False keeps Indic text readable
    # in the file and is smaller over the wire once gzipped.
    # allow_nan=False turns a stray NaN into a build failure rather than a
    # site that loads for nobody.
    text = json.dumps(_json_safe(obj), ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def build(config: str, out_dir: Path) -> None:
    log.info("loading + training the engine (this is the slow part) …")
    eng = Engine(config)

    engine_dir = out_dir / "engine"
    sizes: dict[str, int] = {}

    # --- the ranked cells, exactly as /api/zones serves them ---------------
    sizes["zones.geojson"] = _write(engine_dir / "zones.geojson", eng.zones_geojson)

    # --- green / amber / red forecast, exactly as /api/greenloss ----------
    sizes["greenloss.json"] = _write(engine_dir / "greenloss.json", eng.greenloss)

    # --- the ranked panel as rows, plus the cell centre so the client can
    #     answer "nearest bare cell" without h3 -------------------------------
    cells = []
    for r in eng.ranked.itertuples():
        score = _finite(r.score)
        if score is None:
            continue                      # no real coverage — omit, don't guess
        try:
            lat, lon = cell_center(r.zone)
        except Exception:
            lat = lon = None
        cells.append({
            "zone": r.zone,
            "rank": int(r.rank),
            "score": round(score, 4),
            "lat": _finite(lat),
            "lon": _finite(lon),
            "aqi_latest": _finite(r.aqi_latest),
            "aqi_pred_delta": _finite(r.aqi_pred_delta),
            "ndvi_latest": _finite(r.ndvi_latest),
            "ndvi_trend_per_year": _finite(float(r.ndvi_slope) * 12),
            "ndvi_pred_delta": _finite(r.ndvi_pred_delta),
            "plantable_space": _finite(r.plantable_space),
        })
    rec_by_zone = {r["zone"]: r for r in eng.recommendations}
    for c in cells:
        rec = rec_by_zone.get(c["zone"], {})
        c["species"] = rec.get("species", [])
        c["justification"] = rec.get("justification", "")
    sizes["cells.json"] = _write(engine_dir / "cells.json", cells)

    # --- soil, per cell ----------------------------------------------------
    soil = {}
    for zone, prof in eng.soil.items():
        soil[zone] = {
            "ph": _finite(prof.ph),
            "ph_class": prof.ph_class,
            "texture": prof.texture_class,
            "texture_simple": prof.texture_simple,
            "sand": _finite(prof.sand),
            "silt": _finite(prof.silt),
            "clay": _finite(prof.clay),
            "organic_carbon": _finite(prof.soc),
            "nitrogen": _finite(prof.nitrogen),
            "moisture": _finite(prof.moisture),
        }
    sizes["soil.json"] = _write(engine_dir / "soil.json", soil)

    # --- species knowledge base -------------------------------------------
    sizes["species.json"] = _write(engine_dir / "species.json", SPECIES_KB)

    # --- what this build is ------------------------------------------------
    health = eng.health()
    meta = {
        "city": health["city"],
        "built_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "zones": health["zones"],
        "memory_records": health["memory_records"],
        "soil_cells": len(soil),
        "species_kb": len(SPECIES_KB),
        "greenloss": health["greenloss"],
        "languages": health["languages"],
        "h3_resolution": eng.cfg.grid.h3_resolution,
        "months_history": eng.cfg.data.months_history,
        "thresholds": eng.greenloss["thresholds"],
        "mcda_weights": dict(eng.cfg.mcda.weights),
        # Said plainly, in the payload itself, so the claim travels with the
        # data rather than living only in a README nobody opened.
        "reasoning": "deterministic",
        "note": (
            "Baked from the trained engine. The ranking, forecast, soil and "
            "species tables are computed by greenplan and are not recomputed "
            "in the browser. The assistant is deterministic: intent matching "
            "and planning, no model and no weights."
        ),
    }
    sizes["meta.json"] = _write(engine_dir / "meta.json", meta)

    # --- static assets the page already expects ---------------------------
    shutil.copy2(ROOT / "index.html", out_dir / "index.html")
    i18n_src, i18n_dst = ROOT / "data" / "i18n", out_dir / "data" / "i18n"
    if i18n_src.is_dir():
        i18n_dst.mkdir(parents=True, exist_ok=True)
        for f in i18n_src.glob("*.json"):
            shutil.copy2(f, i18n_dst / f.name)

    web = ROOT / "web" / "gv-engine.js"
    if web.is_file():
        shutil.copy2(web, out_dir / "gv-engine.js")
        sizes["gv-engine.js"] = web.stat().st_size
    else:
        log.warning("web/gv-engine.js missing — the static build will have no "
                    "assistant. Build it before deploying.")

    # Cloudflare Pages: long-cache the immutable engine payloads, never the
    # HTML. Without this the browser re-downloads 230 KB of unchanged JSON on
    # every visit, and worse, serves a stale index.html after a redeploy.
    (out_dir / "_headers").write_text(
        "/engine/*\n"
        "  Cache-Control: public, max-age=3600, must-revalidate\n"
        "/data/i18n/*\n"
        "  Cache-Control: public, max-age=3600, must-revalidate\n"
        "/gv-engine.js\n"
        "  Cache-Control: public, max-age=600, must-revalidate\n"
        "/index.html\n"
        "  Cache-Control: no-cache\n"
        "/\n"
        "  Cache-Control: no-cache\n",
        encoding="utf-8",
    )

    # Cloudflare Pages SPA fallback. The studio is one page that routes in
    # the browser, so a deep link must still serve index.html rather than
    # 404. This file was present in the deployed site but nothing here wrote
    # it, so every rebuild silently dropped it. Written here so the build
    # reproduces the deployment instead of quietly diverging from it.
    (out_dir / "_redirects").write_text("/*  /index.html  200\n", encoding="utf-8", newline="\n")

    total = sum(sizes.values())
    log.info("wrote %s", out_dir)
    for k, v in sorted(sizes.items(), key=lambda kv: -kv[1]):
        log.info("  %-18s %8.1f KB", k, v / 1024)
    log.info("  %-18s %8.1f KB", "TOTAL", total / 1024)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake the engine to static JSON")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--out", default="dist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    build(args.config, out)


if __name__ == "__main__":
    main()
