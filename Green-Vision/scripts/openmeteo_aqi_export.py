"""Export REAL monthly AQI for a city into GreenGrid's csv: adapter format —
free, no API key. Source: Open-Meteo Air-Quality archive (US AQI, hourly),
averaged to monthly means.

It samples AQI at the SAME zone coordinates the mock adapters use (so the real
AQI stream snaps onto the same H3 cells as the still-mock NDVI/traffic), writing
columns `zone, month, aqi` where zone = Z000.. matches the mock zone ids.

Usage:
    python scripts/openmeteo_aqi_export.py --config config/city.yaml \
        --start 2023-01 --end 2026-06 --out data/ahmedabad_aqi.csv

Then in config/city.yaml:
    data.start_month: 1                 # calendar month of --start
    data.months_history: <#months>      # printed at the end
    adapters.aqi: "csv:data/ahmedabad_aqi.csv"

Open-Meteo's air-quality archive is a rolling window (~2023-01 onward at time of
writing); pick --start within it or months come back empty.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenplan.config import load_config  # noqa: E402
from greenplan.features.h3grid import cells_covering_bbox  # noqa: E402

API = "https://air-quality-api.open-meteo.com/v1/air-quality"


def month_list(start: str, end: str) -> list[str]:
    """Inclusive 'YYYY-MM' list from start to end."""
    ys, ms = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    out = []
    y, m = ys, ms
    while (y, m) <= (ye, me):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def fetch_zone_monthly(lat: float, lon: float, start: str, end: str, retries: int = 2) -> dict[str, float]:
    """Monthly-mean US AQI for one point, keyed 'YYYY-MM'."""
    import json

    q = {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": "us_aqi",
        "start_date": start + "-01",
        "end_date": _last_day(end),
        "timezone": "auto",
    }
    url = API + "?" + urllib.parse.urlencode(q)
    last = ""
    for attempt in range(retries + 1):
        try:
            data = json.loads(urllib.request.urlopen(url, timeout=60).read())
            h = data.get("hourly", {})
            buckets: dict[str, list[float]] = defaultdict(list)
            for t, a in zip(h.get("time", []), h.get("us_aqi", [])):
                if a is not None:
                    buckets[t[:7]].append(float(a))
            return {k: sum(v) / len(v) for k, v in buckets.items()}
        except Exception as exc:  # transient network / rate limit
            last = repr(exc)[:120]
            time.sleep(1.5 * (attempt + 1))
    print(f"    ! failed ({last})", file=sys.stderr)
    return {}


def _last_day(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    nm_y, nm_m = (y + 1, 1) if m == 12 else (y, m + 1)
    import datetime
    last = datetime.date(nm_y, nm_m, 1) - datetime.timedelta(days=1)
    return last.isoformat()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export real monthly AQI (Open-Meteo, no key)")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--start", required=True, help="first month, YYYY-MM")
    ap.add_argument("--end", required=True, help="last month, YYYY-MM")
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    cfg = load_config(args.config)
    months = month_list(args.start, args.end)
    idx = {ym: i for i, ym in enumerate(months)}
    print(f"{cfg.city.name}: {len(months)} months {args.start}..{args.end}")

    # Sample the same H3 grid the engine aggregates on (res from config), one
    # Open-Meteo point per cell centre, so AQI lands on the same cells as NDVI.
    res = cfg.grid.h3_resolution
    grid = cells_covering_bbox(cfg.city.bbox, res)
    cells = sorted(grid.items())
    print(f"  {len(cells)} H3 res-{res} cells")

    rows: list[tuple[str, float, float, int, float]] = []
    for n, (zone, (lat, lon)) in enumerate(cells, 1):
        monthly = fetch_zone_monthly(lat, lon, args.start, args.end)
        hit = 0
        for ym, aqi in monthly.items():
            if ym in idx:
                rows.append((zone, lat, lon, idx[ym], round(aqi, 1)))
                hit += 1
        print(f"  [{n:>3}/{len(cells)}] {zone} ({lat:.3f},{lon:.3f}) -> {hit} months")
        time.sleep(0.3)  # be polite to the free API

    if not rows:
        sys.exit("No AQI data returned — is --start inside the archive window?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("# Real monthly US AQI from Open-Meteo Air-Quality archive (no key).\n")
        f.write(f"# {cfg.city.name}, {args.start}..{args.end}. zone = H3 res-{res} cell id.\n")
        f.write("zone,lat,lon,month,aqi\n")
        for zone, lat, lon, m, aqi in sorted(rows):
            f.write(f"{zone},{lat:.5f},{lon:.5f},{m},{aqi}\n")

    covered = sorted({m for _, _, _, m, _ in rows})
    print(f"\nWrote {out} — {len(rows)} rows, {len(set(r[0] for r in rows))} cells, "
          f"months {covered[0]}..{covered[-1]} ({len(covered)} distinct)")
    print(f"\nNow set in {args.config}:")
    print(f"  data.start_month: {int(args.start.split('-')[1])}")
    print(f"  data.months_history: {len(months)}")
    print(f'  adapters.aqi: "csv:{args.out}"')


if __name__ == "__main__":
    main()
