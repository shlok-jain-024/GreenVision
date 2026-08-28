"""Export REAL monthly NDVI for a city into GreenGrid's csv: adapter format —
free, NO API key. Source: NASA/USGS MOD13Q1 (250 m, 16-day) via ORNL DAAC's
MODIS Web Service REST API (https://modis.ornl.gov/rst/api/v1/).

One BLOCK subset (kmAboveBelow/LeftRight) covers the whole bbox in a single
spatial call — thousands of pixels — so the whole city needs only a handful of
calls (the API caps each request at 10 sixteen-day composites, so time is
chunked). Each pixel is geolocated from the sinusoidal-projection corners the
API returns, snapped to its H3 cell (res from config), averaged per composite,
then composites averaged to monthly means.

Writes `zone,lat,lon,month,ndvi` keyed by H3 cell id so it lands on the same
grid the engine aggregates on.

Usage:
    python scripts/modis_ndvi_export.py --config config/city.yaml \
        --start 2023-01 --end 2026-06 --out data/ahmedabad_ndvi.csv

MOD13Q1: archive 2000-present, NDVI scaled x10000, fill -3000 (both handled).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenplan.config import load_config  # noqa: E402
from greenplan.features.h3grid import cells_covering_bbox, latlng_to_cell  # noqa: E402

API = "https://modis.ornl.gov/rst/api/v1/MOD13Q1/subset"
BAND = "250m_16_days_NDVI"
FILL = -3000
R = 6371007.181  # MODIS sinusoidal sphere radius (m)


def a_date(d: dt.date) -> str:
    return f"A{d.year}{d.timetuple().tm_yday:03d}"


def month_of(ym: str, start: str) -> int:
    y, m = map(int, ym.split("-")); ys, ms = map(int, start.split("-"))
    return (y - ys) * 12 + (m - ms)


def time_chunks(start: str, end: str, months_per: int = 4) -> list[tuple[dt.date, dt.date]]:
    """Windows of <=~8 composites each (API caps at 10)."""
    ys, ms = map(int, start.split("-")); ye, me = map(int, end.split("-"))
    out, y, m = [], ys, ms
    while (y, m) <= (ye, me):
        a = dt.date(y, m, 1)
        em = m + months_per - 1; ey = y + (em - 1) // 12; em = ((em - 1) % 12) + 1
        if (ey, em) > (ye, me):
            ey, em = ye, me
        b = dt.date(ey + (em == 12), (em % 12) + 1, 1) - dt.timedelta(days=1)
        out.append((a, b))
        m = em + 1; y = ey
        if m > 12:
            m, y = 1, y + 1
    return out


def sinu_to_latlon(x: float, y: float) -> tuple[float, float]:
    lat = math.degrees(y / R)
    lon = math.degrees(x / (R * math.cos(math.radians(lat))))
    return lat, lon


def fetch_block(lat: float, lon: float, km: int, a: dt.date, b: dt.date, retries: int = 3) -> dict:
    q = {"latitude": lat, "longitude": lon, "startDate": a_date(a), "endDate": a_date(b),
         "band": BAND, "kmAboveBelow": km, "kmLeftRight": km}
    url = API + "?" + urllib.parse.urlencode(q)
    last = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=120).read())
        except Exception as exc:
            last = repr(exc)[:140]
            time.sleep(2.5 * (attempt + 1))
    raise RuntimeError(f"block {a}..{b} failed: {last}")


def pixel_cells(meta: dict, res: int, grid: set[str]) -> dict[int, str]:
    """Map flat pixel index -> H3 cell (only cells inside the bbox grid)."""
    xll, yll = float(meta["xllcorner"]), float(meta["yllcorner"])
    cs, nrows, ncols = float(meta["cellsize"]), int(meta["nrows"]), int(meta["ncols"])
    out: dict[int, str] = {}
    for row in range(nrows):
        y = yll + (nrows - row - 0.5) * cs
        for col in range(ncols):
            x = xll + (col + 0.5) * cs
            la, lo = sinu_to_latlon(x, y)
            cell = latlng_to_cell(la, lo, res)
            if cell in grid:
                out[row * ncols + col] = cell
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Export real monthly NDVI (MODIS/ORNL, no key)")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--km", type=int, default=14, help="half-size of the block (km)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    res = cfg.grid.h3_resolution
    grid = cells_covering_bbox(cfg.city.bbox, res)
    centre = ((cfg.city.bbox[1] + cfg.city.bbox[3]) / 2, (cfg.city.bbox[0] + cfg.city.bbox[2]) / 2)
    chunks = time_chunks(args.start, args.end)
    print(f"{cfg.city.name}: {len(grid)} H3 res-{res} cells, {len(chunks)} time-chunks, "
          f"block ±{args.km} km at {centre}")

    # composite -> cell -> [pixel ndvi]; built across all chunks
    comp: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    pmap: dict[int, str] | None = None
    cell_centre = {c: xy for c, xy in grid.items()}
    for n, (a, b) in enumerate(chunks, 1):
        d = fetch_block(centre[0], centre[1], args.km, a, b)
        if pmap is None:
            pmap = pixel_cells(d, res, set(grid))
            print(f"  pixel->cell map: {len(pmap)} pixels land in {len(set(pmap.values()))} cells")
        for rec in d.get("subset", []):
            cal = rec.get("calendar_date", "")[:7]
            data = rec.get("data", [])
            if not cal:
                continue
            for idx, cell in pmap.items():
                v = data[idx] if idx < len(data) else None
                if v is not None and v > FILL:
                    comp[rec["calendar_date"]][cell].append(v / 10000.0)
        print(f"  [{n:>2}/{len(chunks)}] {a}..{b}: {len(d.get('subset',[]))} composites")
        time.sleep(0.5)

    # composite cell-means -> monthly cell-means
    monthly: dict[tuple[str, int], list[float]] = defaultdict(list)
    for cal_date, cells in comp.items():
        ym = cal_date[:7]; mi = month_of(ym, args.start)
        for cell, vals in cells.items():
            if vals:
                monthly[(cell, mi)].append(sum(vals) / len(vals))

    rows = []
    for (cell, mi), vals in monthly.items():
        if mi < 0:
            continue
        la, lo = cell_centre[cell]
        rows.append((cell, la, lo, mi, round(sum(vals) / len(vals), 4)))
    if not rows:
        sys.exit("No NDVI produced — check block coverage / date range.")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("# Real monthly NDVI from NASA MOD13Q1 250m 16-day via ORNL DAAC (no key).\n")
        f.write(f"# {cfg.city.name}, {args.start}..{args.end}. zone = H3 res-{res} cell id.\n")
        f.write("zone,lat,lon,month,ndvi\n")
        for cell, la, lo, mi, v in sorted(rows):
            f.write(f"{cell},{la:.5f},{lo:.5f},{mi},{v}\n")

    zones = {r[0] for r in rows}; months = sorted({r[3] for r in rows})
    print(f"\nWrote {out} — {len(rows)} rows, {len(zones)}/{len(grid)} cells with data, "
          f"months {months[0]}..{months[-1]} ({len(months)} distinct)")


if __name__ == "__main__":
    main()
