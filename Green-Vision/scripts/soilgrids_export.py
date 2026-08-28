"""Export REAL soil chemistry and texture for a city into GreenGrid's soil
adapter format — free, NO API key. Source: ISRIC SoilGrids v2.0 (250 m,
modelled) via the public REST API (https://rest.isric.org/soilgrids/v2.0/).

One query per H3 cell centre pulls pH, sand, silt, clay, organic carbon and
nitrogen for the top two depth slices. SoilGrids masks built-up land, so a
dense urban centre often returns nulls; when that happens we re-sample at four
offsets around the cell centre and take the first hit, which recovers most of
the urban cells from their unsealed surroundings.

Writes `zone,lat,lon,phh2o,sand,silt,clay,soc,nitrogen` keyed by H3 cell id so
it lands on the same grid the engine aggregates on. Values are written in
SoilGrids' native integer scaling (pH*10, g/kg, dg/kg, cg/kg);
`greenplan/features/soil.py` auto-unscales by magnitude.

Usage:
    python scripts/soilgrids_export.py --config config/city.yaml \
        --out data/ahmedabad_soilgrids.csv

Then in config/city.yaml:
    soil.soilgrids_csv: data/ahmedabad_soilgrids.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenplan.config import load_config  # noqa: E402
from greenplan.features.h3grid import cell_center, cells_covering_bbox  # noqa: E402

API = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ("phh2o", "sand", "silt", "clay", "soc", "nitrogen")
DEPTHS = ("0-5cm", "5-15cm")

# Offsets (km) tried when the cell centre itself is masked as built-up.
FALLBACK_KM = 1.5


def offsets(lat: float, lon: float, km: float) -> list[tuple[float, float]]:
    """Four points N/S/E/W of the centre, `km` away."""
    dlat = km / 110.574
    dlon = km / (111.320 * math.cos(math.radians(lat)) or 1)
    return [(lat + dlat, lon), (lat - dlat, lon), (lat, lon + dlon), (lat, lon - dlon)]


def query(lat: float, lon: float, retries: int = 3) -> dict[str, float] | None:
    """Mean of the available depth slices per property, or None if all masked."""
    q = [("lon", round(lon, 5)), ("lat", round(lat, 5))]
    q += [("property", p) for p in PROPS]
    q += [("depth", d) for d in DEPTHS]
    q += [("value", "mean")]
    url = API + "?" + urllib.parse.urlencode(q)

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            payload = json.loads(urllib.request.urlopen(req, timeout=120).read())
            break
        except Exception:
            if attempt == retries:
                return None
            time.sleep(3.0 * (attempt + 1))
    else:
        return None

    out: dict[str, float] = {}
    for layer in payload.get("properties", {}).get("layers", []):
        vals = [
            d["values"]["mean"]
            for d in layer.get("depths", [])
            if d.get("values", {}).get("mean") is not None
        ]
        if vals:
            out[layer["name"]] = sum(vals) / len(vals)
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description="Export SoilGrids soil profiles per H3 cell")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--out", default="data/soilgrids.csv")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--no-fallback", action="store_true", help="skip offset re-sampling")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cells = cells_covering_bbox(cfg.city.bbox, cfg.grid.h3_resolution)
    print(f"{len(cells)} H3 res-{cfg.grid.h3_resolution} cells over {cfg.city.name}")

    rows, direct, recovered, missed = [], 0, 0, 0
    for i, cell in enumerate(sorted(cells), 1):
        lat, lon = cell_center(cell)
        vals = query(lat, lon)
        source = "centre"

        if vals is None and not args.no_fallback:
            for olat, olon in offsets(lat, lon, FALLBACK_KM):
                time.sleep(args.sleep)
                vals = query(olat, olon)
                if vals:
                    source = "offset"
                    break

        if vals is None:
            missed += 1
            status = "masked"
        else:
            direct += source == "centre"
            recovered += source == "offset"
            status = source
            rows.append(
                {
                    "zone": cell,
                    "lat": round(lat, 5),
                    "lon": round(lon, 5),
                    **{p: round(vals[p], 1) for p in PROPS if p in vals},
                }
            )

        print(f"  [{i}/{len(cells)}] {cell} {status}", flush=True)
        time.sleep(args.sleep)

    if not rows:
        print("error: every cell came back masked — nothing written", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# Real soil chemistry + texture from ISRIC SoilGrids v2.0 (250 m, no key).\n")
        fh.write(f"# {cfg.city.name}. zone = H3 res-{cfg.grid.h3_resolution} cell id.\n")
        fh.write("# Native SoilGrids scaling: phh2o = pH*10, sand/silt/clay = g/kg,\n")
        fh.write("# soc = dg/kg, nitrogen = cg/kg. soil.py auto-unscales by magnitude.\n")
        w = csv.DictWriter(fh, fieldnames=["zone", "lat", "lon", *PROPS])
        w.writeheader()
        w.writerows(rows)

    print(
        f"\nwrote {out} — {len(rows)}/{len(cells)} cells "
        f"({direct} at centre, {recovered} recovered from offsets, {missed} masked)"
    )
    print(f"\nNow set in {args.config}:\n    soil:\n      soilgrids_csv: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
