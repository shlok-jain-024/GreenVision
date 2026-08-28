"""Soil intelligence per H3 cell, from SoilGrids (chemistry/texture) and NASA
SMAP (moisture).

Honest scope: **moisture** is a real satellite measurement; **pH, organic
carbon, nitrogen, sand/silt/clay** are SoilGrids' *modelled* estimates (250 m,
from ~240k profiles + covariates) — good enough to steer species choice, not a
substitute for a lab test. True mineralogy is not recoverable from these
sources. For urban tree survival, pH + texture + moisture are what matter.

Everything degrades gracefully: any missing stream leaves that field ``None``
and never blocks a run.

CSV conventions match ``adapters/csvfile.py`` — ``lat``+``lon`` (or ``zone``),
plus the named property columns. SoilGrids exports may be integer-scaled
(pH×10, sand/silt/clay in g/kg); values are auto-unscaled by magnitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .h3grid import latlng_to_cell

log = logging.getLogger(__name__)


@dataclass
class SoilProfile:
    ph: float | None = None
    soc: float | None = None          # organic carbon (g/kg, informational)
    nitrogen: float | None = None     # total N (g/kg, informational)
    sand: float | None = None         # percent
    silt: float | None = None         # percent
    clay: float | None = None         # percent
    moisture: float | None = None     # volumetric, 0..1 (SMAP)
    texture_class: str | None = None  # USDA 12-class
    texture_simple: str | None = None # sandy | loamy | clay
    ph_class: str | None = None       # acidic | neutral | alkaline

    def summary(self) -> str:
        bits = []
        if self.ph is not None:
            bits.append(f"pH {self.ph:.1f} ({self.ph_class})")
        if self.texture_class:
            bits.append(self.texture_class)
        if self.moisture is not None:
            bits.append(f"moisture {self.moisture:.2f}")
        if self.soc is not None:
            bits.append(f"SOC {self.soc:.0f} g/kg")
        return ", ".join(bits) or "no soil data"

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "ph": round(self.ph, 2) if self.ph is not None else None,
            "ph_class": self.ph_class,
            "texture": self.texture_class,
            "texture_simple": self.texture_simple,
            "moisture": round(self.moisture, 3) if self.moisture is not None else None,
            "soc": round(self.soc, 1) if self.soc is not None else None,
        }


def ph_class(ph: float) -> str:
    if ph < 6.5:
        return "acidic"
    if ph <= 7.5:
        return "neutral"
    return "alkaline"


def usda_texture(sand: float, silt: float, clay: float) -> str:
    """USDA 12-class texture from percent sand/silt/clay (standard NRCS rules)."""
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52) or (clay < 7 and silt < 50):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and sand > 20:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    if clay >= 40:
        return "clay"
    return "loam"


_SIMPLE = {
    "sand": "sandy", "loamy sand": "sandy", "sandy loam": "sandy",
    "sandy clay loam": "sandy", "sandy clay": "sandy",
    "clay": "clay", "silty clay": "clay", "clay loam": "clay",
    "silty clay loam": "clay",
    "loam": "loamy", "silt loam": "loamy", "silt": "loamy",
}


def _unscale_ph(v: float) -> float:
    return v / 10.0 if v > 14 else v


def _load_prop_csv(path: Path, resolution: int) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip().lower() for c in df.columns]
    if "zone" in df.columns:
        df["cell"] = df["zone"].astype(str)
    elif "lat" in df.columns and "lon" in df.columns:
        df["cell"] = [
            latlng_to_cell(la, lo, resolution)
            for la, lo in zip(
                pd.to_numeric(df["lat"], errors="coerce"),
                pd.to_numeric(df["lon"], errors="coerce"),
            )
        ]
    else:
        raise ValueError(f"{path}: need 'lat'+'lon' or 'zone' columns")
    return df


def load_soil(
    soilgrids_csv: str | Path | None,
    moisture_csv: str | Path | None,
    resolution: int,
) -> dict[str, SoilProfile]:
    """Merge SoilGrids + SMAP CSVs into one SoilProfile per H3 cell."""
    profiles: dict[str, SoilProfile] = {}

    if soilgrids_csv:
        path = Path(soilgrids_csv)
        if not path.exists():
            # Soil is enrichment, not a dependency: a run without it still
            # ranks and still picks species, just on pollution load alone.
            # Export it with scripts/soilgrids_export.py.
            log.warning(
                "SoilGrids CSV not found (%s) - continuing without soil. "
                "Species will be matched on pollution tolerance only; "
                "generate it with scripts/soilgrids_export.py",
                path,
            )
            path = None

    if soilgrids_csv and path is not None:
        df = _load_prop_csv(path, resolution)
        cols = {c: c for c in ("phh2o", "ph", "soc", "nitrogen", "sand", "silt", "clay") if c in df.columns}
        for cell, g in df.groupby("cell"):
            prof = profiles.setdefault(cell, SoilProfile())
            def mean(name: str) -> float | None:
                if name not in cols:
                    return None
                v = pd.to_numeric(g[cols[name]], errors="coerce").mean()
                return None if pd.isna(v) else float(v)
            ph = mean("phh2o")
            ph = mean("ph") if ph is None else ph
            if ph is not None:
                prof.ph = _unscale_ph(ph)
                prof.ph_class = ph_class(prof.ph)
            prof.soc = mean("soc")
            prof.nitrogen = mean("nitrogen")
            sand, silt, clay = mean("sand"), mean("silt"), mean("clay")
            if None not in (sand, silt, clay) and (sand + silt + clay) > 0:
                total = sand + silt + clay          # unit-agnostic: use proportions
                sand, silt, clay = (100 * v / total for v in (sand, silt, clay))
                prof.sand, prof.silt, prof.clay = sand, silt, clay
                prof.texture_class = usda_texture(sand, silt, clay)
                prof.texture_simple = _SIMPLE.get(prof.texture_class)

    if moisture_csv:
        path = Path(moisture_csv)
        if not path.exists():
            raise FileNotFoundError(f"SMAP moisture CSV not found: {path}")
        df = _load_prop_csv(path, resolution)
        mcol = next((c for c in ("moisture", "smap", "value", "mean") if c in df.columns), None)
        if mcol is None:
            raise ValueError(f"{path}: need a 'moisture'/'value'/'mean' column")
        for cell, g in df.groupby("cell"):
            v = pd.to_numeric(g[mcol], errors="coerce").mean()
            if not pd.isna(v):
                profiles.setdefault(cell, SoilProfile()).moisture = float(v)

    if profiles:
        log.info("soil profiles: %d cells", len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Species <-> soil compatibility (used by the mock model and the prompt rules)
# ---------------------------------------------------------------------------


def _parse_ph_range(spec: str) -> tuple[float, float]:
    try:
        lo, hi = spec.split("-")
        return float(lo), float(hi)
    except (ValueError, AttributeError):
        return 0.0, 14.0


def species_soil_ok(
    species: dict[str, str], profile: "SoilProfile | dict | None"
) -> bool:
    """True if the species tolerates the cell's soil pH and texture. Accepts a
    SoilProfile or its ``as_dict()`` form. Unknown soil (or an 'any'-texture
    species) never rules a species out."""
    if profile is None:
        return True
    if isinstance(profile, dict):
        ph = profile.get("ph")
        texture_simple = profile.get("texture_simple")
    else:
        ph, texture_simple = profile.ph, profile.texture_simple
    if ph is not None:
        lo, hi = _parse_ph_range(species.get("soil_ph", "0-14"))
        if not (lo - 0.3 <= ph <= hi + 0.3):
            return False
    tex = species.get("soil_texture", "any")
    if texture_simple and tex not in ("any", ""):
        if texture_simple not in {t.strip() for t in tex.split("|")}:
            return False
    return True
