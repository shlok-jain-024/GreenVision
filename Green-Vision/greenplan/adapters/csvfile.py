"""CSV-backed adapter: feed any metric from a plain CSV file.

Expected columns (extras are ignored):
  * `month`                        — integer month index, 0 = oldest
  * a value column                 — named after the metric ("ndvi", "traffic",
                                     "aqi"), or "value", or "mean" (what Google
                                     Earth Engine exports)
  * either `zone`, or `lat` + `lon` (zone ids are then derived from the
                                     coordinates, which also enables H3 snapping)

Rows with missing values are dropped. Configure per stream in city.yaml, e.g.:

  adapters:
    green_cover: "csv:data/ahmedabad_ndvi_monthly.csv"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .base import SCHEMA, DataAdapter


class CsvMetricAdapter(DataAdapter):
    def __init__(self, path: str | Path, metric: str) -> None:
        if metric not in ("traffic", "aqi", "ndvi"):
            raise ValueError(f"unknown metric {metric!r}")
        self.path = Path(path)
        self.metric = metric
        self._geometry: dict[str, tuple[float, float]] | None = None

    def fetch(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(
                f"CSV for {self.metric} not found: {self.path}. "
                "Export it first (see scripts/gee_ndvi_export.js for NDVI)."
            )
        df = pd.read_csv(self.path, comment="#")
        df.columns = [c.strip().lower() for c in df.columns]

        value_col = next(
            (c for c in (self.metric, "value", "mean") if c in df.columns), None
        )
        if value_col is None or "month" not in df.columns:
            raise ValueError(
                f"{self.path}: need columns 'month' and one of "
                f"'{self.metric}'/'value'/'mean'; found {list(df.columns)}"
            )

        if "zone" in df.columns:
            df["zone"] = df["zone"].astype(str)
        elif "lat" in df.columns and "lon" in df.columns:
            df["zone"] = (
                df["lat"].round(4).astype(str) + "," + df["lon"].round(4).astype(str)
            )
        else:
            raise ValueError(f"{self.path}: need a 'zone' column or 'lat'+'lon' columns")

        df = df.dropna(subset=[value_col, "month"])
        out = pd.DataFrame(
            {
                "zone": df["zone"],
                "month": df["month"].astype(int),
                "traffic": np.nan,
                "aqi": np.nan,
                "ndvi": np.nan,
            }
        )
        out[self.metric] = df[value_col].astype(float).to_numpy()

        if "lat" in df.columns and "lon" in df.columns:
            firsts = df.drop_duplicates("zone")
            self._geometry = {
                str(r.zone): (float(r.lat), float(r.lon)) for r in firsts.itertuples()
            }
        return self.validate(out[SCHEMA])

    def zone_geometry(self) -> dict[str, tuple[float, float]] | None:
        return self._geometry
