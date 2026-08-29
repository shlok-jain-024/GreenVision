"""Sentinel-2 NDVI at 10 m for a city — free, NO API key.

MODIS (250 m, 42-month archive) stays the engine's TEMPORAL backbone: it is
pre-composited, keyless, and long enough to forecast from. What it cannot say
is where INSIDE a ~5 km2 priority cell the bare ground actually is. This
script adds that spatial layer: true NDVI (B08 near-infrared vs B04 red) from
the most recent cloud-free Sentinel-2 scene, 625x sharper than MODIS.

Keyless access is what makes this possible now: AWS's Earth Search STAC API
(https://earth-search.aws.element84.com/v1, Element 84) serves Sentinel-2 L2A
as cloud-optimized GeoTIFFs — no Copernicus account, no token. rasterio
reads just the bbox window over HTTPS, so a whole city costs a few tens of MB.

Two outputs:
  1. --out-cells:  zone,lat,lon,month,ndvi — current per-H3-cell mean NDVI on
     the SAME grid as the engine (one snapshot month, for comparison layers).
  2. --out-sites:  lat,lon,patch_area_m2 — the lowest-NDVI 10 m ground inside
     each cell: candidate planting sites for `sites.candidates_csv` in
     config/city.yaml, replacing the 1-NDVI proxy with measured bare ground.

Usage:
    python scripts/sentinel2_ndvi_export.py --config config/city.yaml \
        --out-cells data/ahmedabad_s2_ndvi.csv \
        --out-sites data/ahmedabad_sites.csv

Honest limits, stated: one scene is a snapshot, not a composite — pass
--max-cloud (default 10%%) and the script picks the most recent scene under
it; scene date is printed and written into the header. NDVI < the bare
threshold can also be water/rock/roofs at scene time; sites are candidates
for a human to verify, not survey results.

--classify-sites attacks that last problem. It runs the land-cover classifier
from `scripts/train_landcover.py` (ESA WorldCover-trained, ONNX, executed by
OpenVINO) over the same pixels and adds three columns: landcover_class,
plantable and confidence. Nothing is dropped — a site the classifier calls
water or roof is FLAGGED and stays in the file, because the classifier is a
screening aid and being wrong about it must be visible.

It then AUDITS itself. The held-out score in report.json is measured on a
random sample of the whole scene, but these sites are the extreme low-NDVI
tail picked by a different rule, and a model can be strong on the first and
useless on the second. So ESA WorldCover — the classifier's own training
labels — is read back at the exact candidate coordinates and the agreement is
written to site_audit.json, which greenplan/features/sites.py treats as a
second gate. On the shipped Ahmedabad data the model passes gate 1 and fails
gate 2; see the README.

    python scripts/sentinel2_ndvi_export.py --config config/city.yaml \
        --out-sites data/ahmedabad_sites.csv --classify-sites

--scene-id pins an exact STAC item, so an existing sites CSV can be
reproduced pixel-for-pixel and only gain the new columns (the scene id is in
its header).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenplan.config import load_config  # noqa: E402
from greenplan.features.h3grid import cell_center, cells_covering_bbox  # noqa: E402

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
STAC_ITEM = "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items"


def get_scene(scene_id: str) -> dict:
    """One exact STAC item by id — reproduces an existing sites CSV."""
    with urllib.request.urlopen(f"{STAC_ITEM}/{scene_id}", timeout=60) as r:
        return json.load(r)


def find_scene(bbox: list[float], max_cloud: float) -> dict:
    """Most recent Sentinel-2 L2A scene covering the bbox under the cloud cap."""
    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 12,
    }
    req = urllib.request.Request(
        STAC_SEARCH,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        feats = json.load(r)["features"]
    if not feats:
        raise SystemExit(
            f"no Sentinel-2 scene with cloud cover < {max_cloud}% over this bbox; "
            "raise --max-cloud"
        )
    # prefer the scene whose footprint contains the bbox centre (UTM tiles
    # overlap at edges); fall back to the newest
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for f in feats:
        fb = f["bbox"]
        if fb[0] <= cx <= fb[2] and fb[1] <= cy <= fb[3]:
            return f
    return feats[0]


def read_band(href: str, bbox: list[float], cache: Path | None = None) -> tuple[np.ndarray, object]:
    """Windowed read of one COG band over the lon/lat bbox.

    A city bbox is a few minutes per band on a home connection, so the raw
    window is cached under data/raw/ (gitignored) and re-runs are instant."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["data"], (Affine(*z["transform"]), CRS.from_wkt(str(z["crs"])))

    with rasterio.open(href) as src:
        l, b, r, t = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(l, b, r, t, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        crs = src.crs
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, data=data, transform=np.array(transform[:6]), crs=str(crs.to_wkt()))
    return data, (transform, crs)


class SiteClassifier:
    """The trained land-cover filter, executed by OpenVINO.

    Loads what `scripts/train_landcover.py` wrote — the ONNX graph, the
    feature normalisation, and the report whose `gate` decides whether the
    downstream engine is allowed to act on the flags at all. This script
    always writes the columns; `greenplan/features/sites.py` re-reads the same
    report and ignores them if the model lost to the NDVI baseline."""

    def __init__(self, model_dir: Path, device: str = "CPU") -> None:
        import openvino as ov

        self.dir = model_dir
        norm_path = model_dir / "norm.json"
        if not norm_path.exists():
            raise SystemExit(
                f"no land-cover classifier at {model_dir}. Train one with:\n"
                "    python scripts/train_landcover.py --config config/city.yaml"
            )
        self.norm = json.loads(norm_path.read_text(encoding="utf-8"))
        self.groups = self.norm["groups"]
        self.plantable = set(self.norm["plantable_groups"])
        self.classes = self.norm["classes"]
        self.mu = np.asarray(self.norm["mu"], dtype=np.float32)
        self.sd = np.asarray(self.norm["sd"], dtype=np.float32)

        core = ov.Core()
        self.compiled = core.compile_model(
            core.read_model(str(model_dir / self.norm["onnx"])), device
        )
        self.port = next(
            (p for p in self.compiled.outputs
             if p.get_partial_shape().rank.get_length() == 2),
            self.compiled.outputs[-1],
        )
        report_path = model_dir / "report.json"
        self.report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.exists() else {}
        )
        self.gate = self.report.get("gate", self.norm.get("report", {}).get("gate", {}))

    @staticmethod
    def features(blue, green, red, nir) -> np.ndarray:
        """Same six features the trainer used, in the same order."""
        ndvi = (nir - red) / np.maximum(nir + red, 1.0)
        ndwi = (green - nir) / np.maximum(green + nir, 1.0)
        return np.stack(
            [blue / 1e4, green / 1e4, red / 1e4, nir / 1e4, ndvi, ndwi], axis=-1
        ).astype(np.float32)

    def predict(self, feats: np.ndarray) -> tuple[list[str], np.ndarray]:
        z = ((feats - self.mu) / self.sd).astype(np.float32)
        proba = np.asarray(self.compiled(z)[self.port])
        cols = [i for i, c in enumerate(self.classes) if c in self.plantable]
        p_plant = proba[:, cols].sum(axis=1)
        names = [self.groups[self.classes[i]] for i in proba.argmax(axis=1)]
        return names, p_plant


def audit_against_worldcover(
    lats: np.ndarray, lons: np.ndarray, pred_plantable: np.ndarray,
    held_out_accuracy: float | None,
) -> dict:
    """Score the classifier on the population it is actually DEPLOYED on.

    The held-out score in report.json is measured on a random spatial sample
    of the whole scene. The candidate sites are nothing like that sample: they
    are the extreme low-NDVI tail, hand-picked by a different rule. A model can
    be excellent on the general population and useless on that tail, and no
    amount of held-out accuracy will reveal it.

    So this re-reads ESA WorldCover 2021 — the classifier's own training label
    source — at the exact candidate coordinates and asks whether the model
    still agrees with the labels it was fitted to. Disagreeing with your own
    teacher on the ground you are deployed on is a failure that needs no
    interpretation."""
    import rasterio
    from rasterio.windows import from_bounds

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_landcover import (  # noqa: PLC0415
        PLANTABLE_GROUPS, WC_TO_GROUP, WORLDCOVER, worldcover_tiles,
    )

    m = 0.01  # margin: sites can sit a hair outside the nominal bbox
    box = [lons.min() - m, lats.min() - m, lons.max() + m, lats.max() + m]
    wc = np.zeros(len(lats), dtype=np.uint8)
    for tile in worldcover_tiles(box):
        with rasterio.open(WORLDCOVER.format(tile=tile)) as src:
            win = from_bounds(*box, src.transform).round_offsets().round_lengths()
            data = src.read(1, window=win, boundless=True, fill_value=0)
            tr = src.window_transform(win)
        col = np.clip(((lons - tr.c) / tr.a).astype(int), 0, data.shape[1] - 1)
        row = np.clip(((lats - tr.f) / tr.e).astype(int), 0, data.shape[0] - 1)
        wc = np.where(wc == 0, data[row, col], wc)

    plant_codes = [c for c, g in WC_TO_GROUP.items() if g in PLANTABLE_GROUPS]
    truth = np.isin(wc, plant_codes)
    labelled = wc > 0
    agreement = float(np.mean(truth[labelled] == pred_plantable[labelled]))
    # Pre-stated rule: a model may lose some accuracy off its validation
    # distribution, but not a lot. 10 points is the tolerance.
    floor = round((held_out_accuracy or 0.0) - 0.10, 4)
    codes = {int(k): int(v) for k, v in zip(*np.unique(wc[labelled], return_counts=True))}
    return {
        "question": "does the classifier still agree with its OWN training "
                    "labels on the candidate-site population it is deployed on",
        "label_source": "ESA WorldCover 2021 v200 at the candidate coordinates",
        "n_sites": int(labelled.sum()),
        "worldcover_codes": codes,
        "worldcover_plantable": int(truth[labelled].sum()),
        "worldcover_plantable_rate": round(float(truth[labelled].mean()), 4),
        "classifier_plantable": int(pred_plantable[labelled].sum()),
        "classifier_plantable_rate": round(float(pred_plantable[labelled].mean()), 4),
        "agreement": round(agreement, 4),
        "held_out_accuracy": held_out_accuracy,
        "required_agreement": floor,
        "passed": bool(agreement >= floor),
        "criterion": "agreement on the deployment population must be within "
                     "0.10 of the held-out accuracy",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Sentinel-2 10 m NDVI export (keyless)")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--out-cells", default=None, help="per-H3-cell mean NDVI csv")
    ap.add_argument("--out-sites", default=None, help="low-NDVI candidate sites csv")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--bare-ndvi", type=float, default=0.15,
                    help="NDVI below this counts as bare, plantable ground")
    ap.add_argument("--sites-per-cell", type=int, default=15)
    ap.add_argument("--scene-id", default=None,
                    help="pin an exact STAC item id (reproduces an existing CSV)")
    ap.add_argument("--classify-sites", action="store_true",
                    help="add landcover_class/plantable/confidence columns using "
                         "the model from scripts/train_landcover.py")
    ap.add_argument("--landcover-dir", default=None,
                    help="override sites.landcover_dir from the config")
    ap.add_argument("--cache-dir", default="data/raw/landcover",
                    help="cache the raw band windows here (gitignored)")
    args = ap.parse_args()
    if not (args.out_cells or args.out_sites):
        ap.error("nothing to do: pass --out-cells and/or --out-sites")
    if args.classify_sites and not args.out_sites:
        ap.error("--classify-sites needs --out-sites")

    cfg = load_config(args.config)
    bbox = list(cfg.city.bbox)
    res = cfg.grid.h3_resolution

    clf = None
    if args.classify_sites:
        clf = SiteClassifier(
            Path(args.landcover_dir) if args.landcover_dir
            else cfg.resolve(cfg.sites.landcover_dir),
            cfg.model.device,
        )
        g = clf.gate
        print(
            f"land-cover filter from {clf.dir}: gate 1 (held-out accuracy vs "
            f"the tuned NDVI rule) {'PASSED' if g.get('passed') else 'FAILED'} "
            f"at {g.get('accuracy_gain', 0):+.4f}. Gate 2 is measured below, "
            f"on the candidate sites themselves."
        )

    scene = get_scene(args.scene_id) if args.scene_id else find_scene(bbox, args.max_cloud)
    date = scene["properties"]["datetime"][:10]
    cloud = scene["properties"].get("eo:cloud_cover")
    print(f"scene {scene['id']}  date {date}  cloud {cloud:.1f}%")

    cache_dir = Path(cfg.resolve(args.cache_dir)) / scene["id"]
    red, _ = read_band(scene["assets"]["red"]["href"], bbox, cache_dir / "red.npz")
    nir, georef = read_band(scene["assets"]["nir"]["href"], bbox, cache_dir / "nir.npz")
    transform, crs = georef
    h = min(red.shape[0], nir.shape[0])
    w = min(red.shape[1], nir.shape[1])
    red, nir = red[:h, :w], nir[:h, :w]

    blue = green = None
    if clf is not None:
        # The classifier needs the visible bands too — same window, so this
        # costs two more COG reads, not a second download of the scene.
        blue, _ = read_band(scene["assets"]["blue"]["href"], bbox, cache_dir / "blue.npz")
        green, _ = read_band(scene["assets"]["green"]["href"], bbox, cache_dir / "green.npz")
        blue, green = blue[:h, :w], green[:h, :w]

    valid = (red > 0) & (nir > 0)  # 0 = L2A nodata
    ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
    print(f"grid {w}x{h} px @10m — mean NDVI {np.nanmean(ndvi):.3f}")

    # geolocate every pixel centre once (rows/cols -> lon/lat)
    from rasterio.warp import transform as warp_transform

    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lons, lats = warp_transform(crs, "EPSG:4326", xs.ravel(), ys.ravel())
    lons = np.asarray(lons).reshape(h, w)
    lats = np.asarray(lats).reshape(h, w)

    # assign pixels to H3 cells via each cell's bounding box mask (fast, no
    # per-pixel python h3 calls); cells overlap boxes slightly at hex edges,
    # which shifts a mean by <1% — fine for a comparison layer.
    cells = cells_covering_bbox(tuple(bbox), res)
    print(f"{len(cells)} H3 res-{res} cells")

    import h3 as h3lib

    cell_rows = []
    site_rows = []
    for cell in cells:
        boundary = np.array(h3lib.cell_to_boundary(cell))  # (lat, lon) pairs
        la0, la1 = boundary[:, 0].min(), boundary[:, 0].max()
        lo0, lo1 = boundary[:, 1].min(), boundary[:, 1].max()
        mask = (lats >= la0) & (lats <= la1) & (lons >= lo0) & (lons <= lo1)
        vals = ndvi[mask]
        vals = vals[~np.isnan(vals)]
        if not len(vals):
            continue
        lat_c, lon_c = cell_center(cell)
        cell_rows.append((cell, lat_c, lon_c, float(np.mean(vals))))

        if args.out_sites:
            bare = mask & (ndvi < args.bare_ndvi)
            by, bx = np.nonzero(bare)
            if len(by):
                order = np.argsort(ndvi[by, bx])[: args.sites_per_cell]
                for i in order:
                    site_rows.append(
                        (float(lats[by[i], bx[i]]), float(lons[by[i], bx[i]]), 100.0,
                         int(by[i]), int(bx[i]))
                    )

    header = (
        f"# Sentinel-2 L2A 10 m NDVI via AWS Earth Search STAC (no key).\n"
        f"# scene {scene['id']} date {date} cloud {cloud:.1f}%. "
        f"SNAPSHOT of one scene, not a composite.\n"
    )
    if args.out_cells:
        with open(args.out_cells, "w", encoding="utf-8") as f:
            f.write(header + "zone,lat,lon,month,ndvi\n")
            for cell, lat_c, lon_c, v in cell_rows:
                f.write(f"{cell},{lat_c:.5f},{lon_c:.5f},0,{v:.4f}\n")
        print(f"wrote {args.out_cells}: {len(cell_rows)} cell means")
    if args.out_sites:
        classes: list[str] = []
        p_plant = np.ones(len(site_rows), dtype=np.float32)
        if clf is not None and site_rows:
            ys = np.array([r[3] for r in site_rows])
            xs_ = np.array([r[4] for r in site_rows])
            feats = clf.features(
                blue[ys, xs_], green[ys, xs_], red[ys, xs_], nir[ys, xs_]
            )
            classes, p_plant = clf.predict(feats)
            flagged = int((p_plant < 0.5).sum())
            counts: dict[str, int] = {}
            for c, p in zip(classes, p_plant):
                counts[c] = counts.get(c, 0) + 1

            # Score the model where it is USED, not just where it was validated.
            audit = audit_against_worldcover(
                np.array([r[0] for r in site_rows]),
                np.array([r[1] for r in site_rows]),
                p_plant >= 0.5,
                (clf.report.get("model", {}).get("openvino", {}) or {}).get("accuracy"),
            )
            audit["scene"] = scene["id"]
            audit["date"] = date
            (clf.dir / "site_audit.json").write_text(
                json.dumps(audit, indent=2), encoding="utf-8"
            )

            header += (
                f"# land-cover filter {clf.dir.name} ({clf.report.get('labels', {}).get('source', '?')}): "
                f"{flagged}/{len(site_rows)} candidates FLAGGED not-plantable. "
                f"Flagged rows are KEPT, never dropped.\n"
                f"# classifier gate 1 (held-out accuracy vs the tuned NDVI rule): "
                f"{'PASSED' if clf.gate.get('passed') else 'FAILED'} "
                f"({clf.gate.get('accuracy_gain', 0):+.4f})\n"
                f"# classifier gate 2 (agreement with ESA WorldCover on THESE sites): "
                f"{'PASSED' if audit['passed'] else 'FAILED'} "
                f"({audit['agreement']:.4f}, needs >= {audit['required_agreement']:.4f})\n"
            )
            print(f"\nland-cover classes over {len(site_rows)} candidate sites:")
            for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                print(f"  {c:14s} {n:6d}  ({100 * n / len(site_rows):5.1f}%)")
            print(f"  -> {flagged} of {len(site_rows)} flagged not-plantable "
                  f"({100 * flagged / len(site_rows):.1f}%)")
            print(
                f"\ndeployment audit vs ESA WorldCover 2021 at these same "
                f"{audit['n_sites']} coordinates:\n"
                f"  WorldCover calls plantable : {audit['worldcover_plantable']:6d} "
                f"({100 * audit['worldcover_plantable_rate']:.1f}%)\n"
                f"  classifier calls plantable : {audit['classifier_plantable']:6d} "
                f"({100 * audit['classifier_plantable_rate']:.1f}%)\n"
                f"  agreement                  : {audit['agreement']:.4f} "
                f"(held-out was {audit['held_out_accuracy']}, "
                f"needs >= {audit['required_agreement']:.4f})\n"
                f"  gate 2: {'PASSED' if audit['passed'] else 'FAILED'} — "
                + ("flags will be acted on"
                   if audit["passed"] else
                   "the classifier is BENCHED; sites.py will keep the columns "
                   "for inspection and filter nothing")
            )

        with open(args.out_sites, "w", encoding="utf-8") as f:
            if clf is None:
                f.write(header + "lat,lon,patch_area_m2\n")
                for lat, lon, area, _y, _x in site_rows:
                    f.write(f"{lat:.6f},{lon:.6f},{area:.0f}\n")
            else:
                f.write(header + "lat,lon,patch_area_m2,landcover_class,plantable,confidence\n")
                for (lat, lon, area, _y, _x), c, p in zip(site_rows, classes, p_plant):
                    f.write(f"{lat:.6f},{lon:.6f},{area:.0f},{c},"
                            f"{int(p >= 0.5)},{float(p):.4f}\n")
        print(
            f"wrote {args.out_sites}: {len(site_rows)} candidate sites — set\n"
            f"  sites.candidates_csv: {args.out_sites}\nin config to use them"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
