"""Reshape an Esri NDVI export into GreenGrid's `csv:` adapter format.

Workflow (see README "Finding planting sites / NDVI via Esri"): in ArcGIS Pro,
load a Living Atlas Sentinel-2 / Landsat multispectral image service, compute
NDVI = (NIR-Red)/(NIR+Red) with Band Arithmetic per month, `Sample` onto a
point grid over the bbox, and export the table. This script turns that export
into:
    lon, lat, month, mean          (month = integer index, 0 = --start month)

Accepts either a date column (reshaped to a month index via --start) or an
existing integer `month` column. Multiple observations in one month are
averaged; NDVI stored ×10000 (integer) is auto-rescaled.

Usage:
    python scripts/esri_ndvi_to_greengrid.py in.csv out.csv --start 2016-01
    # then in config/city.yaml:
    #   adapters.green_cover: "csv:out.csv"

Note: Esri hosts the imagery, but the NDVI pixels are ESA Sentinel-2 / USGS
Landsat — the portal is Esri, the data provenance is Sentinel/Landsat.
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
    ap.add_argument("input", help="Esri NDVI point-sample CSV")
    ap.add_argument("output", help="GreenGrid-format CSV to write")
    ap.add_argument(
        "--start",
        help="YYYY-MM that becomes month index 0 (required if input has a date "
        "column rather than an integer month). Align with data.start_month and "
        "your traffic/AQI series.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = list(df.columns)
    lat_c = find_col(cols, "lat") or find_col(cols, "y")
    lon_c = find_col(cols, "lon") or find_col(cols, "x")
    ndvi_c = find_col(cols, "ndvi") or find_col(cols, "mean") or find_col(cols, "value")
    if not all((lat_c, lon_c, ndvi_c)):
        print(f"error: need lat/lon/NDVI columns in {cols}", file=sys.stderr)
        return 1

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df[lat_c], errors="coerce").round(5),
            "lon": pd.to_numeric(df[lon_c], errors="coerce").round(5),
            "ndvi": pd.to_numeric(df[ndvi_c], errors="coerce"),
        }
    ).dropna(subset=["lat", "lon", "ndvi"])

    month_c = find_col(cols, "month")
    date_c = find_col(cols, "date")
    if month_c and month_c != ndvi_c:
        out["month"] = pd.to_numeric(df.loc[out.index, month_c], errors="coerce").astype("Int64")
    elif date_c:
        if not args.start:
            print("error: input has a date column; pass --start YYYY-MM", file=sys.stderr)
            return 1
        start = pd.Period(args.start, freq="M")
        per = pd.to_datetime(df.loc[out.index, date_c]).dt.to_period("M")
        out["month"] = (per - start).map(lambda d: d.n)
    else:
        print("error: need a 'month' or 'date' column to index time", file=sys.stderr)
        return 1

    out = out.dropna(subset=["month"])
    out["month"] = out["month"].astype(int)
    out = out[out["month"] >= 0]
    # drop obvious nodata, then undo 10000x integer scaling if present
    out = out[out["ndvi"] > -1.0]
    if out["ndvi"].abs().median() > 1.5:
        out["ndvi"] = out["ndvi"] / 10000.0

    monthly = (
        out.groupby(["lat", "lon", "month"], as_index=False)["ndvi"]
        .mean()
        .rename(columns={"ndvi": "mean"})
        .sort_values(["lat", "lon", "month"])
    )
    monthly.to_csv(args.output, index=False)
    print(
        f"wrote {args.output}: {len(monthly)} rows, "
        f"{monthly[['lat', 'lon']].drop_duplicates().shape[0]} points, "
        f"months {monthly['month'].min()}..{monthly['month'].max()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
