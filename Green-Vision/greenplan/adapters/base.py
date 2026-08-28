"""Pluggable data-source interfaces.

Every adapter returns a pandas DataFrame with the columns

    zone   : str   — source-specific zone id (ward name, sensor id, tile id, ...)
    month  : int   — consecutive month index, 0 = oldest month, no gaps required
                     but months must be comparable ACROSS adapters (same month 0)
    traffic: float — congestion index (0-100); NaN if this adapter doesn't carry it
    aqi    : float — air quality index (0-500); NaN if not carried
    ndvi   : float — mean NDVI (0-1); NaN if not carried

Each concrete adapter fills only its own metric column and leaves the other
two as NaN; the feature layer merges all streams onto the common H3 grid.

If the adapter knows where its zones are, it should also implement
`zone_geometry()` returning {zone_id: (lat, lon)} so zones can be snapped to
H3 cells. Without geometry, zone ids are used as-is as the analysis grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

SCHEMA = ["zone", "month", "traffic", "aqi", "ndvi"]


class DataAdapter(ABC):
    """Abstract source of per-zone monthly city data."""

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Return rows following SCHEMA (see module docstring)."""

    def zone_geometry(self) -> dict[str, tuple[float, float]] | None:
        """Optional {zone_id: (lat, lon)} centroid map for H3 snapping."""
        return None

    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in SCHEMA if c not in df.columns]
        if missing:
            raise ValueError(f"adapter frame missing columns: {missing}")
        out = df[SCHEMA].copy()
        out["zone"] = out["zone"].astype(str)
        out["month"] = out["month"].astype(int)
        return out


# ---------------------------------------------------------------------------
# Fill these in with your real data sources. Each fetch() must return the
# SCHEMA columns with only your metric populated (others NaN). Keep month
# indices aligned with the other adapters (same month 0 across all three).
# ---------------------------------------------------------------------------


class GreenCoverAdapter(DataAdapter):
    """Plug in your satellite green-cover source here.

    Typical implementations: Sentinel-2 or Landsat NDVI composites via
    Google Earth Engine / Sentinel Hub, aggregated to monthly means per zone.
    Populate the `ndvi` column; leave `traffic` and `aqi` as NaN.
    """

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement GreenCoverAdapter.fetch() with your NDVI source, "
            "or run with adapters.green_cover: mock"
        )


class TrafficAdapter(DataAdapter):
    """Plug in your traffic-congestion source here.

    Typical implementations: Google Maps / TomTom / HERE congestion APIs, or
    municipal loop-detector feeds, reduced to a monthly 0-100 congestion
    index per zone. Populate `traffic`; leave `aqi` and `ndvi` as NaN.
    """

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement TrafficAdapter.fetch() with your traffic source, "
            "or run with adapters.traffic: mock"
        )


class AQIAdapter(DataAdapter):
    """Plug in your air-quality source here.

    Typical implementations: CPCB station data, OpenAQ, or private sensor
    networks, as monthly mean AQI per zone. Populate `aqi`; leave `traffic`
    and `ndvi` as NaN.
    """

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement AQIAdapter.fetch() with your AQI source, "
            "or run with adapters.aqi: mock"
        )
