"""Snap heterogeneous source zones onto a common H3 hexagon grid.

Works with both the h3 v3 API (geo_to_h3) and v4 API (latlng_to_cell).
"""

from __future__ import annotations

import logging

import pandas as pd

import h3

log = logging.getLogger(__name__)


def latlng_to_cell(lat: float, lon: float, res: int) -> str:
    if hasattr(h3, "latlng_to_cell"):  # v4
        return h3.latlng_to_cell(lat, lon, res)
    return h3.geo_to_h3(lat, lon, res)  # v3


def cell_center(cell: str) -> tuple[float, float]:
    """(lat, lon) of the hexagon center."""
    if hasattr(h3, "cell_to_latlng"):  # v4
        return h3.cell_to_latlng(cell)
    return h3.h3_to_geo(cell)  # v3


def cells_covering_bbox(
    bbox: tuple[float, float, float, float], res: int
) -> dict[str, tuple[float, float]]:
    """Every H3 cell of `res` whose center falls in bbox=[lon_min,lat_min,
    lon_max,lat_max], mapped to its (lat, lon) center. Used to build a dense,
    gap-free sampling grid for real data instead of scattered random points."""
    lon_min, lat_min, lon_max, lat_max = bbox
    ring = [
        (lat_min, lon_min), (lat_min, lon_max),
        (lat_max, lon_max), (lat_max, lon_min),
    ]
    if hasattr(h3, "polygon_to_cells"):  # v4
        cells = h3.polygon_to_cells(h3.LatLngPoly(ring), res)
    else:  # v3
        geo = {"type": "Polygon", "coordinates": [[[lo, la] for la, lo in ring]]}
        cells = h3.polyfill(geo, res, geo_json_conformant=True)
    return {c: cell_center(c) for c in cells}


def cell_area_m2(cell: str) -> float:
    """Area of the hexagon in square metres (h3 v3 and v4)."""
    return float(h3.cell_area(cell, unit="m^2"))


def cell_boundary_lonlat(cell: str) -> list[list[float]]:
    """Closed GeoJSON-style ring: [[lon, lat], ...] with first == last."""
    if hasattr(h3, "cell_to_boundary"):  # v4
        ring = h3.cell_to_boundary(cell)
    else:  # v3
        ring = h3.h3_to_geo_boundary(cell)
    coords = [[float(lon), float(lat)] for lat, lon in ring]
    coords.append(coords[0])
    return coords


def aggregate_to_h3(
    frames: list[pd.DataFrame],
    geometry: dict[str, tuple[float, float]] | None,
    resolution: int,
) -> pd.DataFrame:
    """Merge adapter frames and aggregate them onto H3 cells.

    Returns a panel with columns zone (H3 cell id), month, traffic, aqi, ndvi
    — one row per (cell, month), metrics averaged across all source zones
    that fall in the cell (NaNs ignored, so each adapter contributes only the
    metric it actually carries).

    If no geometry is available the source zone ids are kept as the grid.
    """
    combined = pd.concat(frames, ignore_index=True)

    if geometry:
        known = combined["zone"].isin(geometry)
        dropped = combined.loc[~known, "zone"].unique()
        if len(dropped):
            log.warning("dropping %d zones without geometry: %s", len(dropped), dropped[:5])
        combined = combined[known].copy()
        cell_of = {
            z: latlng_to_cell(lat, lon, resolution) for z, (lat, lon) in geometry.items()
        }
        combined["zone"] = combined["zone"].map(cell_of)
    else:
        log.warning("no zone geometry provided; using source zone ids as the grid")

    panel = (
        combined.groupby(["zone", "month"], as_index=False)[["traffic", "aqi", "ndvi"]]
        .mean()
        .sort_values(["zone", "month"], ignore_index=True)
    )
    return panel
