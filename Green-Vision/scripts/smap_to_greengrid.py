"""Reshape a NASA SMAP soil-moisture export into GreenGrid's soil-moisture CSV.

SMAP surface soil moisture (e.g. the 1 km downscaled product, or SPL3SMP_E)
sampled per point via NASA AppEEARS comes as a time series:
    Latitude, Longitude, Date, ..._soil_moisture, ...

GreenGrid's soil layer wants one representative value per location, so this
averages moisture over time (optionally over recent months) and writes:
    lat, lon, moisture            (volumetric, 0..1)

Point config/city.yaml at it:
    soil:
      moisture_csv: data/ahmedabad_smap.csv

Usage:
    python scripts/smap_to_greengrid.py in.csv out.csv [--since YYYY-MM]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def find_col(cols: list[str], *needles: str) -> str | None:
    for c in cols:
        low = c.lower()
        if all(n in low for n in needles):
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="SMAP point-sample CSV (AppEEARS or similar)")
    ap.add_argument("output", help="GreenGrid soil-moisture CSV to write")
    ap.add_argument("--since", help="only average observations on/after YYYY-MM")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = list(df.columns)
    lat_c = find_col(cols, "lat") or find_col(cols, "y")
    lon_c = find_col(cols, "lon") or find_col(cols, "x")
    moist_c = (
        find_col(cols, "soil", "moisture")
        or find_col(cols, "moisture")
        or find_col(cols, "smap")
        or find_col(cols, "mean")
        or find_col(cols, "value")
    )
    if not all((lat_c, lon_c, moist_c)):
        print(f"error: need lat/lon/moisture columns in {cols}", file=sys.stderr)
        return 1

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_c], errors="coerce").round(5),
            "lon": pd.to_numeric(df[lon_c], errors="coerce").round(5),
            "moisture": pd.to_numeric(df[moist_c], errors="coerce"),
        }
    )
    date_c = find_col(cols, "date")
    if args.since and date_c:
        keep = pd.to_datetime(df[date_c], errors="coerce") >= pd.Timestamp(args.since)
        out = out[keep.to_numpy()]
    # SMAP fill values are large negatives; keep only physical 0..1 moisture
    out = out[(out["moisture"] >= 0) & (out["moisture"] <= 1)].dropna()

    agg = (
        out.groupby(["lat", "lon"], as_index=False)["moisture"]
        .mean()
        .sort_values(["lat", "lon"])
    )
    agg.to_csv(args.output, index=False)
    print(f"wrote {args.output}: {len(agg)} points, "
          f"moisture {agg['moisture'].min():.2f}..{agg['moisture'].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
