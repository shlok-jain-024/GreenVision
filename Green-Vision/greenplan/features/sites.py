"""Bare-land candidate sites: the bridge from hexagon-level ranking to actual
map points where trees can go.

Input is a CSV of bare-ground patches (e.g. centroids of the "Bare ground"
class from Esri's 10 m Land Cover, exported over the city bbox). Each patch is
snapped to its H3 cell so we can (a) report specific coordinates inside each
top-priority zone and (b) replace the crude ``1 - NDVI`` plantable-space proxy
with a real bare-area fraction per cell.

Accuracy note: 10 m land-cover locates a patch to ~±5 m. That is enough to send
a crew to the right plot, NOT to mark an individual planting pit — a human still
confirms the exact spot on a sub-metre basemap (Esri World Imagery). The brief
prints the coordinates to go inspect.

CSV columns (extras ignored, names case-insensitive):
  * ``lat`` + ``lon``            — patch centroid (required)
  * ``patch_area_m2`` / ``area`` / ``value`` / ``mean`` — bare area, optional
                                   (defaults to one 10 m pixel = 100 m²)
  * ``confidence``               — 0..1 classifier confidence, optional
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .h3grid import cell_area_m2, latlng_to_cell

log = logging.getLogger(__name__)

_DEFAULT_PIXEL_M2 = 100.0  # one 10 m Sentinel-2 / Esri land-cover pixel


class CandidateSites:
    """Bare-ground patches indexed by H3 cell."""

    def __init__(self, sites: pd.DataFrame, resolution: int) -> None:
        self.resolution = resolution
        self.sites = sites  # columns: lat, lon, cell, patch_area_m2, confidence

    @property
    def empty(self) -> bool:
        return self.sites.empty

    def zones(self) -> set[str]:
        return set(self.sites["cell"].unique())

    def plantable_fraction_by_cell(self) -> dict[str, float]:
        """bare patch area / hexagon area, per cell, clipped to [0, 1].

        A cell fully covered by bare patches -> 1.0; a sealed cell with no
        patches never appears here (callers fall back to the proxy)."""
        out: dict[str, float] = {}
        for cell, g in self.sites.groupby("cell"):
            area = cell_area_m2(cell)
            if area <= 0:
                continue
            out[cell] = float(np.clip(g["patch_area_m2"].sum() / area, 0.0, 1.0))
        return out

    def sites_for_zone(self, zone: str, k: int) -> list[dict[str, float]]:
        """The k biggest / most-confident patches in a zone, largest first."""
        g = self.sites[self.sites["cell"] == zone]
        if g.empty:
            return []
        g = g.sort_values(
            ["confidence", "patch_area_m2"], ascending=False
        ).head(k)
        return [
            {
                "lat": round(float(r.lat), 6),
                "lon": round(float(r.lon), 6),
                "patch_area_m2": round(float(r.patch_area_m2), 1),
                "confidence": round(float(r.confidence), 3),
            }
            for r in g.itertuples()
        ]


def load_candidate_sites(
    path: str | Path, resolution: int, min_patch_m2: float = 0.0
) -> CandidateSites:
    """Read a bare-patch CSV and snap every patch to its H3 cell."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"sites CSV not found: {path}. Export bare-ground patches (see the "
            "README 'Finding planting sites' section) or unset sites.candidates_csv."
        )
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "lat" not in df.columns or "lon" not in df.columns:
        raise ValueError(f"{path}: need 'lat' and 'lon' columns; found {list(df.columns)}")

    area_col = next(
        (c for c in ("patch_area_m2", "area", "value", "mean") if c in df.columns), None
    )
    area = (
        pd.to_numeric(df[area_col], errors="coerce")
        if area_col
        else pd.Series(_DEFAULT_PIXEL_M2, index=df.index)
    )
    conf = (
        pd.to_numeric(df["confidence"], errors="coerce")
        if "confidence" in df.columns
        else pd.Series(1.0, index=df.index)
    )

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df["lat"], errors="coerce"),
            "lon": pd.to_numeric(df["lon"], errors="coerce"),
            "patch_area_m2": area.fillna(_DEFAULT_PIXEL_M2),
            "confidence": conf.fillna(1.0).clip(0.0, 1.0),
        }
    ).dropna(subset=["lat", "lon"])
    out = out[out["patch_area_m2"] >= float(min_patch_m2)]
    out["cell"] = [
        latlng_to_cell(la, lo, resolution) for la, lo in zip(out["lat"], out["lon"])
    ]
    log.info(
        "candidate sites: %d patches in %d cells from %s",
        len(out), out["cell"].nunique(), path.name,
    )
    return CandidateSites(out.reset_index(drop=True), resolution)
