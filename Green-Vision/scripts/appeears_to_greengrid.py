"""Convert a NASA AppEEARS point-sample CSV into GreenGrid's adapter format.

AppEEARS (https://appeears.earthdatacloud.nasa.gov/, free NASA Earthdata
login) exports per-point time series with columns like:
    Latitude, Longitude, Date, MOD13Q1_061__250m_16_days_NDVI, ...

This script reshapes that into the CSV the `csv:` adapter reads:
    lat, lon, month, mean          (month = integer index, 0 = --start month)

Usage:
    python scripts/appeears_to_greengrid.py in.csv out.csv --start 2016-01
    # then in config/city.yaml:
    #   data.start_month: 1
    #   adapters.green_cover: "csv:data/out.csv"

Multiple observations in the same month (MODIS is 16-day) are averaged.
Raw MODIS NDVI is stored scaled by 10000; if values look unscaled (>1.5),
they are divided by 10000 automatically.
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
    ap.add_argument("input", help="AppEEARS point-sample CSV")
    ap.add_argument("output", help="GreenGrid-format CSV to write")
    ap.add_argument(
        "--start",
        required=True,
        help="YYYY-MM that becomes month index 0 (align with data.start_month "
        "and your other data streams)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    cols = list(df.columns)
    lat_c = find_col(cols, "latitude") or find_col(cols, "lat")
    lon_c = find_col(cols, "longitude") or find_col(cols, "lon")
    date_c = find_col(cols, "date")
    ndvi_c = find_col(cols, "ndvi")
    if not all((lat_c, lon_c, date_c, ndvi_c)):
        print(
            f"error: could not find lat/lon/date/NDVI columns in {cols}",
            file=sys.stderr,
        )
        return 1

    out = pd.DataFrame(
        {
            "lat": df[lat_c].astype(float).round(5),
            "lon": df[lon_c].astype(float).round(5),
            "date": pd.to_datetime(df[date_c]),
            "ndvi": pd.to_numeric(df[ndvi_c], errors="coerce"),
        }
    ).dropna(subset=["ndvi"])

    # MODIS fill value is -3000 (scaled) / -0.3; drop obvious fill rows,
    # then undo the 10000x integer scaling if present
    out = out[out["ndvi"] > -0.5 * (10000 if out["ndvi"].abs().median() > 1.5 else 1)]
    if out["ndvi"].abs().median() > 1.5:
        out["ndvi"] = out["ndvi"] / 10000.0

    start = pd.Period(args.start, freq="M")
    out["month"] = (out["date"].dt.to_period("M") - start).map(lambda d: d.n)
    out = out[out["month"] >= 0]

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
