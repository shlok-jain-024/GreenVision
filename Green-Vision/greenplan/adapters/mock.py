"""Synthetic city data so the whole pipeline runs before any real API exists.

A single seeded MockCityDataset generates correlated per-zone monthly series
(traffic pushes AQI up; dense zones lose NDVI), and the three mock adapters
each expose ONE metric from it — mirroring how three independent real sources
would later be merged by the feature layer.

Seasonality is calendar-realistic for Indian cities:
  * AQI peaks in winter (Nov-Jan), dips during monsoon rains.
  * NDVI peaks just after the monsoon (Aug-Oct).
  * Traffic dips slightly during monsoon months.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .base import SCHEMA, DataAdapter

# archetype: (traffic_base, traffic_trend/yr, aqi_base, ndvi_base, ndvi_trend/yr)
_ARCHETYPES: dict[str, tuple[float, float, float, float, float]] = {
    "dense_core":      (78.0,  1.5, 165.0, 0.18, -0.004),
    "growth_corridor": (55.0,  6.0, 120.0, 0.34, -0.020),
    "industrial":      (60.0,  2.0, 210.0, 0.15, -0.006),
    "residential":     (45.0,  2.5, 110.0, 0.38, -0.008),
    "green_stable":    (30.0,  1.0,  85.0, 0.55,  0.002),
    "periurban":       (25.0,  4.5,  95.0, 0.48, -0.015),
}


class MockCityDataset:
    """Shared synthetic generator; all three mock adapters read from it."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        n_zones: int,
        months: int,
        start_month: int = 1,
        seed: int = 42,
    ) -> None:
        self.months = months
        self.start_month = start_month
        rng = np.random.default_rng(seed)

        lon_min, lat_min, lon_max, lat_max = bbox
        names = list(_ARCHETYPES)
        self.zones: list[dict] = []
        for i in range(n_zones):
            self.zones.append(
                {
                    "zone": f"Z{i:03d}",
                    "lat": float(rng.uniform(lat_min, lat_max)),
                    "lon": float(rng.uniform(lon_min, lon_max)),
                    "archetype": names[i % len(names)],
                }
            )
        self._frame = self._generate(rng)

    def _generate(self, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for z in self.zones:
            tb, tt, ab, nb, nt = _ARCHETYPES[z["archetype"]]
            # per-zone jitter so zones of one archetype aren't identical
            tb += rng.normal(0, 4);  ab += rng.normal(0, 12);  nb += rng.normal(0, 0.04)
            tt *= rng.uniform(0.7, 1.3);  nt *= rng.uniform(0.7, 1.3)
            ar_t = ar_a = ar_n = 0.0  # AR(1) noise state per metric
            for m in range(self.months):
                cal = (self.start_month - 1 + m) % 12  # 0 = January
                monsoon = math.exp(-0.5 * ((cal - 7.0) / 1.6) ** 2)   # peaks Aug
                winter = math.cos(2 * math.pi * (cal - 0.5) / 12)     # peaks Dec-Jan
                post_monsoon = math.exp(-0.5 * ((cal - 8.5) / 1.8) ** 2)  # Aug-Oct

                ar_t = 0.6 * ar_t + rng.normal(0, 2.2)
                ar_a = 0.6 * ar_a + rng.normal(0, 9.0)
                ar_n = 0.5 * ar_n + rng.normal(0, 0.012)

                traffic = tb + tt * m / 12 - 5.0 * monsoon + ar_t
                aqi = (
                    ab
                    + 45.0 * winter
                    - 30.0 * monsoon
                    + 0.55 * (traffic - tb)  # congestion drags AQI with it
                    + ar_a
                )
                ndvi = nb + nt * m / 12 + 0.10 * post_monsoon + ar_n

                rows.append(
                    {
                        "zone": z["zone"],
                        "month": m,
                        "traffic": float(np.clip(traffic, 5, 100)),
                        "aqi": float(np.clip(aqi, 20, 500)),
                        "ndvi": float(np.clip(ndvi, 0.03, 0.85)),
                    }
                )
        return pd.DataFrame(rows)

    def metric_frame(self, metric: str) -> pd.DataFrame:
        out = self._frame[["zone", "month", metric]].copy()
        for col in ("traffic", "aqi", "ndvi"):
            if col != metric:
                out[col] = np.nan
        return out[SCHEMA]

    def geometry(self) -> dict[str, tuple[float, float]]:
        return {z["zone"]: (z["lat"], z["lon"]) for z in self.zones}


class _MockMetricAdapter(DataAdapter):
    metric: str = ""

    def __init__(self, dataset: MockCityDataset) -> None:
        self._ds = dataset

    def fetch(self) -> pd.DataFrame:
        return self.validate(self._ds.metric_frame(self.metric))

    def zone_geometry(self) -> dict[str, tuple[float, float]]:
        return self._ds.geometry()


class MockGreenCoverAdapter(_MockMetricAdapter):
    metric = "ndvi"


class MockTrafficAdapter(_MockMetricAdapter):
    metric = "traffic"


class MockAQIAdapter(_MockMetricAdapter):
    metric = "aqi"
