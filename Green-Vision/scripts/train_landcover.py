"""Land-cover classifier for the candidate planting sites — free, NO API key.

`sentinel2_ndvi_export.py --out-sites` finds the lowest-NDVI 10 m ground in
every priority cell. Its own header admits the hole: **low NDVI is not the
same thing as bare ground.** Open water, rock, bright sand, a flat concrete
roof and a fresh asphalt car park all sit below the bare threshold, and the
NDVI rule cannot tell them from the dusty vacant plot you actually want. The
README's answer today is "a human verifies all 2,190 candidates".

This script trains the thing that narrows that list: a per-pixel land-cover
classifier, trained on free labelled data, exported to ONNX and executed by
OpenVINO exactly like the forecaster.

Where the training data comes from (both keyless, both free):

  * **X** — one cloud-free 2021 Sentinel-2 L2A scene over the city bbox, from
    the same AWS Earth Search STAC API `sentinel2_ndvi_export.py` uses.
    Features are the four 10 m bands (B02 blue, B03 green, B04 red, B08 NIR)
    plus two derived indices, NDVI and NDWI — NDWI is the one that separates
    water from dry ground, which is precisely the confusion NDVI alone makes.
  * **y** — **ESA WorldCover 2021 v200**, the 10 m global land-cover map, from
    the public `esa-worldcover` S3 bucket. Same year, same 10 m grid, no
    account. Tile N21E072 (3x3 degrees, lower-left corner) covers Ahmedabad.

WorldCover's 11 classes collapse to 6 groups the planting question cares
about, and those collapse again to the binary decision that is actually used:

    tree_cover      (10)                 -> NOT plantable (trees already there)
    shrub_grass     (20, 30)             -> plantable
    cropland        (40)                 -> plantable
    built_up        (50)                 -> NOT plantable
    bare_sparse     (60)                 -> plantable
    water_wetland   (70, 80, 90, 95,100) -> NOT plantable

The split is SPATIAL, not random. Neighbouring 10 m pixels are near-copies of
each other, so a random pixel split scores a model on pixels it has all but
memorised — the classic remote-sensing leak that produces 99% accuracies that
mean nothing. Here the scene is cut into contiguous square blocks, whole
blocks go to train or test, and a buffer of pixels around every test block is
dropped from training so the two sets never touch.

Scored against the honest incumbent: the NDVI threshold rule the site finder
already uses, with its threshold tuned on the SAME training pixels. The
classifier only earns its way into the pipeline if it beats that on held-out
ground — `greenplan/features/sites.py` reads the verdict out of report.json
and benches the model if it lost, exactly as `hybrid` benches the forecaster.

Usage:
    python scripts/train_landcover.py --config config/city.yaml
    python scripts/train_landcover.py --config config/city.yaml --tile 42QZL
    python scripts/train_landcover.py --no-quantize      # skip NNCF INT8

Writes models/{city}/landcover/: landcover.onnx, norm.json, report.json and
(unless --no-quantize) the NNCF INT8 OpenVINO IR.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from greenplan.config import load_config  # noqa: E402

log = logging.getLogger("train_landcover")

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
WORLDCOVER = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
)

# The six groups the planting question cares about, in label order.
GROUPS = [
    "tree_cover",
    "shrub_grass",
    "cropland",
    "built_up",
    "bare_sparse",
    "water_wetland",
]
# ESA WorldCover v200 class code -> group index
WC_TO_GROUP = {
    10: 0,                    # tree cover
    20: 1, 30: 1,             # shrubland, grassland
    40: 2,                    # cropland
    50: 3,                    # built-up
    60: 4,                    # bare / sparse vegetation
    70: 5, 80: 5, 90: 5, 95: 5, 100: 5,  # snow+ice, water, wetland, mangrove, moss
}
# Groups where a tree could actually be put in the ground.
PLANTABLE_GROUPS = {1, 2, 4}

FEATURE_NAMES = ["blue", "green", "red", "nir", "ndvi", "ndwi"]
BANDS = ["blue", "green", "red", "nir"]


# --------------------------------------------------------------------------
# data


def find_scene(bbox: list[float], max_cloud: float, start: str, end: str,
               tile: str | None) -> dict:
    """Least-cloudy Sentinel-2 L2A scene covering the bbox in a date window."""
    body: dict[str, Any] = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": 50,
    }
    req = urllib.request.Request(
        STAC_SEARCH, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        feats = json.load(r)["features"]
    if not feats:
        raise SystemExit(
            f"no Sentinel-2 scene under {max_cloud}% cloud over this bbox between "
            f"{start} and {end}; widen --start/--end or raise --max-cloud"
        )
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    covering = [
        f for f in feats
        if f["bbox"][0] <= cx <= f["bbox"][2] and f["bbox"][1] <= cy <= f["bbox"][3]
    ] or feats
    if tile:
        # Same MGRS tile as the scene the site finder runs on = same viewing
        # geometry, so train and deploy spectra are as comparable as free data
        # allows. Optional: any covering tile is scientifically valid.
        same = [f for f in covering if f"_{tile}_" in f["id"]]
        if same:
            return same[0]
        log.warning("no scene on tile %s in the window; using %s",
                    tile, covering[0]["id"])
    return covering[0]


def read_band(href: str, bbox: list[float], cache: Path | None = None) -> tuple[np.ndarray, tuple]:
    """Windowed read of one COG band over the lon/lat bbox (same as the NDVI
    exporter: only the bytes inside the window cross the network).

    A full four-band read of a city bbox is ~11 minutes on a home connection,
    so the raw window is cached under data/raw/ (gitignored) and re-runs of
    the trainer are instant."""
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
        tr, crs = src.window_transform(win), src.crs
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, data=data, transform=np.array(tr[:6]), crs=str(crs.to_wkt()))
    return data, (tr, crs)


def worldcover_tiles(bbox: list[float]) -> list[str]:
    """WorldCover tile ids (3-degree grid, named by lower-left corner)."""
    out = []
    lon0 = int(math.floor(bbox[0] / 3.0) * 3)
    lon1 = int(math.floor(bbox[2] / 3.0) * 3)
    lat0 = int(math.floor(bbox[1] / 3.0) * 3)
    lat1 = int(math.floor(bbox[3] / 3.0) * 3)
    for la in range(lat0, lat1 + 1, 3):
        for lo in range(lon0, lon1 + 1, 3):
            ns, ew = ("N", "S")[la < 0], ("E", "W")[lo < 0]
            out.append(f"{ns}{abs(la):02d}{ew}{abs(lo):03d}")
    return out


def sample_worldcover(lons: np.ndarray, lats: np.ndarray, bbox: list[float],
                      cache: Path | None = None) -> np.ndarray:
    """Nearest-neighbour WorldCover label for every Sentinel-2 pixel centre.

    Both grids are 10 m, but Sentinel-2 is UTM and WorldCover is EPSG:4326, so
    they are offset by up to half a pixel (~5 m). At 10 m that is the accuracy
    floor of the whole exercise and is stated in the report."""
    import rasterio
    from rasterio.windows import from_bounds

    if cache is not None and cache.exists():
        return np.load(cache, allow_pickle=False)["wc"]

    out = np.zeros(lons.shape, dtype=np.uint8)
    for tile in worldcover_tiles(bbox):
        url = WORLDCOVER.format(tile=tile)
        log.info("WorldCover 2021 v200 tile %s", tile)
        with rasterio.open(url) as src:
            win = (
                from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
                .round_offsets().round_lengths()
            )
            data = src.read(1, window=win, boundless=True, fill_value=0)
            tr = src.window_transform(win)
        col = ((lons - tr.c) / tr.a).astype(np.int64)
        row = ((lats - tr.f) / tr.e).astype(np.int64)
        ok = (row >= 0) & (row < data.shape[0]) & (col >= 0) & (col < data.shape[1])
        vals = np.zeros(lons.shape, dtype=np.uint8)
        vals[ok] = data[row[ok], col[ok]]
        out = np.where(out == 0, vals, out)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, wc=out)
    return out


# --------------------------------------------------------------------------
# metrics


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[dict]:
    rows = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "support": int(np.sum(y_true == c)),
        })
    return rows


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """y_* are 1 = plantable, 0 = not plantable."""
    rows = per_class_f1(y_true, y_pred, 2)
    return {
        "accuracy": round(float(np.mean(y_true == y_pred)), 4),
        "macro_f1": round(float(np.mean([r["f1"] for r in rows])), 4),
        "not_plantable": rows[0] | {"class": "not_plantable"},
        "plantable": rows[1] | {"class": "plantable"},
    }


# --------------------------------------------------------------------------


def main() -> int:  # noqa: PLR0915 - one linear, readable pipeline
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Train the OpenVINO land-cover site filter")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--start", default="2021-04-15", help="scene search window start")
    ap.add_argument("--end", default="2021-07-15", help="scene search window end")
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument("--tile", default=None,
                    help="prefer this MGRS tile (match the site finder's scene)")
    ap.add_argument("--block-px", type=int, default=256,
                    help="side of a spatial train/test block, in 10 m pixels")
    ap.add_argument("--buffer-px", type=int, default=32,
                    help="training pixels this close to a test block are dropped")
    ap.add_argument("--test-frac", type=float, default=0.30)
    ap.add_argument("--max-train", type=int, default=200_000)
    ap.add_argument("--max-test", type=int, default=300_000)
    ap.add_argument("--hidden", type=int, nargs="*", default=[32, 16])
    ap.add_argument("--balance", action="store_true",
                    help="draw equal numbers of TRAINING pixels per land-cover "
                         "class (the test set keeps the true, skewed mix)")
    ap.add_argument("--no-quantize", action="store_true",
                    help="skip NNCF INT8 post-training quantization")
    ap.add_argument("--cache-dir", default="data/raw/landcover",
                    help="cache the raw band + label windows here (gitignored)")
    args = ap.parse_args()

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
    except ImportError:
        print("error: needs scikit-learn and skl2onnx:\n"
              "    pip install scikit-learn skl2onnx", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    bbox = list(cfg.city.bbox)
    rng = np.random.default_rng(cfg.run.seed)

    # --- 1. imagery -------------------------------------------------------
    scene = find_scene(bbox, args.max_cloud, args.start, args.end, args.tile)
    date = scene["properties"]["datetime"][:10]
    cloud = float(scene["properties"].get("eo:cloud_cover") or 0.0)
    log.info("training scene %s  date %s  cloud %.2f%%", scene["id"], date, cloud)

    cache_dir = Path(cfg.resolve(args.cache_dir)) / scene["id"]
    t0 = time.time()
    stack, georef = [], None
    for b in BANDS:
        arr, georef = read_band(
            scene["assets"][b]["href"], bbox, cache_dir / f"{b}.npz"
        )
        stack.append(arr)
    h = min(a.shape[0] for a in stack)
    w = min(a.shape[1] for a in stack)
    blue, green, red, nir = (a[:h, :w] for a in stack)
    transform, crs = georef
    log.info("read 4 bands, %dx%d px @10 m in %.0fs", w, h, time.time() - t0)

    # --- 2. labels --------------------------------------------------------
    from rasterio.warp import transform as warp_transform

    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    lons, lats = warp_transform(crs, "EPSG:4326", xs.ravel(), ys.ravel())
    lons = np.asarray(lons).reshape(h, w)
    lats = np.asarray(lats).reshape(h, w)
    del cols, rows, xs, ys

    wc = sample_worldcover(lons, lats, bbox, cache_dir / "worldcover.npz")
    del lons, lats

    group = np.full((h, w), 255, dtype=np.uint8)
    for code, g in WC_TO_GROUP.items():
        group[wc == code] = g

    # --- 3. features ------------------------------------------------------
    # L2A digital numbers are reflectance x 10000; 0 is nodata.
    valid = (blue > 0) & (green > 0) & (red > 0) & (nir > 0) & (group != 255)
    denom_v = np.maximum(nir + red, 1.0)
    ndvi = (nir - red) / denom_v
    denom_w = np.maximum(green + nir, 1.0)
    ndwi = (green - nir) / denom_w
    feats = np.stack(
        [blue / 10000.0, green / 10000.0, red / 10000.0, nir / 10000.0, ndvi, ndwi],
        axis=-1,
    ).astype(np.float32)
    del blue, green, red, nir, denom_v, denom_w

    wc_hist = {int(k): int(v) for k, v in zip(*np.unique(wc[valid], return_counts=True))}
    log.info("%d labelled valid pixels; WorldCover histogram %s", int(valid.sum()), wc_hist)

    # --- 4. SPATIAL split -------------------------------------------------
    bp = args.block_px
    nby, nbx = math.ceil(h / bp), math.ceil(w / bp)
    block_id = (np.arange(h)[:, None] // bp) * nbx + (np.arange(w)[None, :] // bp)
    n_blocks = nby * nbx
    order = rng.permutation(n_blocks)
    n_test_blocks = max(1, int(round(args.test_frac * n_blocks)))
    test_blocks = set(int(b) for b in order[:n_test_blocks])

    is_test_block = np.isin(block_id, list(test_blocks))
    from scipy.ndimage import binary_dilation

    k = 2 * args.buffer_px + 1
    near_test = binary_dilation(is_test_block, structure=np.ones((k, k), bool))
    test_mask = valid & is_test_block
    train_mask = valid & ~near_test  # buffer keeps train and test from touching
    log.info(
        "spatial split: %d blocks of %dx%d px (%.1f km), %d test; "
        "%d train px, %d test px (%d px dropped into the %d px buffer)",
        n_blocks, bp, bp, bp * 10 / 1000, n_test_blocks,
        int(train_mask.sum()), int(test_mask.sum()),
        int(valid.sum() - train_mask.sum() - test_mask.sum()), args.buffer_px,
    )

    flat_feats = feats.reshape(-1, len(FEATURE_NAMES))
    flat_group = group.ravel()

    def take(mask: np.ndarray, cap: int, balance: bool = False) -> tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(mask.ravel())
        if balance:
            # Ahmedabad is 36% cropland and 40% built-up, while bare/sparse —
            # the class the site finder most needs identified — is 2%. Trained
            # on the raw mix the network simply never predicts the rare
            # classes. Equal-size draws per class fix that; the TEST set is
            # left at the true mix so the reported numbers stay honest.
            g = flat_group[idx]
            present = [c for c in range(len(GROUPS)) if np.any(g == c)]
            per = max(1, cap // len(present))
            picks = []
            for c in present:
                ci = idx[g == c]
                picks.append(
                    rng.choice(ci, per, replace=len(ci) < per) if len(ci) != per else ci
                )
            idx = np.concatenate(picks)
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        return flat_feats[idx], flat_group[idx].astype(np.int64)

    Xtr, ytr = take(train_mask, args.max_train, balance=args.balance)
    Xte, yte = take(test_mask, args.max_test)
    btr = np.isin(ytr, list(PLANTABLE_GROUPS)).astype(np.int64)
    bte = np.isin(yte, list(PLANTABLE_GROUPS)).astype(np.int64)
    log.info("sampled %d train / %d test pixels (plantable share %.3f / %.3f)",
             len(ytr), len(yte), btr.mean(), bte.mean())

    # --- 5. the incumbent: the NDVI threshold the site finder already uses --
    # Tuned on TRAIN pixels only, then applied unchanged to the held-out ones.
    ndvi_tr, ndvi_te = Xtr[:, 4], Xte[:, 4]
    grid = np.arange(0.02, 0.90, 0.01)
    # "plantable" under the incumbent = low NDVI (bare) OR mid NDVI (grass/crop
    # but not closed canopy): the rule is a band, and both edges are swept so
    # the baseline gets its best possible shot.
    best = None
    for lo in np.arange(-0.20, 0.30, 0.02):
        for hi in grid:
            if hi <= lo:
                continue
            pred = ((ndvi_tr >= lo) & (ndvi_tr < hi)).astype(np.int64)
            acc = float(np.mean(pred == btr))
            if best is None or acc > best[0]:
                best = (acc, float(lo), float(hi))
    _, lo, hi = best
    base_te = ((ndvi_te >= lo) & (ndvi_te < hi)).astype(np.int64)
    base_scores = binary_scores(bte, base_te)
    base_scores["ndvi_low"] = round(lo, 3)
    base_scores["ndvi_high"] = round(hi, 3)
    log.info("NDVI baseline band [%.2f, %.2f): held-out acc %.4f macro-F1 %.4f",
             lo, hi, base_scores["accuracy"], base_scores["macro_f1"])

    # --- 6. the challenger -------------------------------------------------
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-9
    Ztr = ((Xtr - mu) / sd).astype(np.float32)
    Zte = ((Xte - mu) / sd).astype(np.float32)

    net = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden), alpha=1e-3, max_iter=300,
        early_stopping=True, n_iter_no_change=10, random_state=cfg.run.seed,
    )
    t0 = time.time()
    net.fit(Ztr, ytr)
    fit_s = time.time() - t0
    log.info("MLP%s trained on %d px in %.0fs", tuple(args.hidden), len(ytr), fit_s)

    proba = net.predict_proba(Zte)
    classes = list(net.classes_)
    p_plant = proba[:, [i for i, c in enumerate(classes) if c in PLANTABLE_GROUPS]].sum(1)
    mlp_bin = (p_plant >= 0.5).astype(np.int64)
    mlp_multi = np.asarray(classes)[proba.argmax(1)]

    mlp_scores = binary_scores(bte, mlp_bin)
    multi_rows = per_class_f1(yte, mlp_multi, len(GROUPS))
    log.info("MLP held-out acc %.4f macro-F1 %.4f",
             mlp_scores["accuracy"], mlp_scores["macro_f1"])

    # Reference only — a tree ensemble is not weight-quantizable, so it is
    # scored for honesty about what was left on the table, never deployed.
    rf = RandomForestClassifier(
        n_estimators=60, max_depth=14, min_samples_leaf=20,
        n_jobs=-1, random_state=cfg.run.seed,
    )
    t0 = time.time()
    rf.fit(Xtr, ytr)
    rf_p = rf.predict_proba(Xte)
    rf_classes = list(rf.classes_)
    rf_pl = rf_p[:, [i for i, c in enumerate(rf_classes) if c in PLANTABLE_GROUPS]].sum(1)
    rf_scores = binary_scores(bte, (rf_pl >= 0.5).astype(np.int64))
    log.info("RandomForest reference held-out acc %.4f macro-F1 %.4f (%.0fs)",
             rf_scores["accuracy"], rf_scores["macro_f1"], time.time() - t0)

    # --- 7. ONNX + OpenVINO -------------------------------------------------
    out_dir = Path(cfg.resolve(cfg.sites.landcover_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "landcover.onnx"

    onx = convert_sklearn(
        net,
        initial_types=[("X", FloatTensorType([None, len(FEATURE_NAMES)]))],
        options={id(net): {"zipmap": False}},
        target_opset=13,
    )
    # skl2onnx gives a classifier graph two outputs: the argmax label (looked
    # up through ai.onnx.ml.ArrayFeatureExtractor) and the probabilities.
    # OpenVINO's ONNX front end has no conversion rule for that ai.onnx.ml op
    # and refuses the whole model — measured, not assumed:
    #   "No conversion rule found for operations: ai.onnx.ml.ArrayFeatureExtractor"
    # The probability branch is plain Gemm/Relu/Softmax, so the deployed graph
    # is pruned down to just that: X -> probabilities. The class label is the
    # argmax of those probabilities, taken in numpy on the other side, and the
    # ordering lives in norm.json["classes"].
    import onnx

    full_path = out_dir / "landcover_full.onnx"
    full_path.write_bytes(onx.SerializeToString())
    prob_name = next(
        o.name for o in onx.graph.output
        if o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    )
    onnx.utils.extract_model(str(full_path), str(onnx_path), ["X"], [prob_name])
    full_path.unlink()
    log.info("wrote %s (%.1f kB, output '%s' only — label branch pruned)",
             onnx_path, onnx_path.stat().st_size / 1e3, prob_name)

    import openvino as ov

    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    compiled = core.compile_model(ov_model, cfg.model.device)
    prob_port = _prob_port(compiled)

    t0 = time.time()
    ov_proba = compiled(Zte)[prob_port]
    ov_ms = (time.time() - t0) * 1000.0
    ov_pl = ov_proba[:, [i for i, c in enumerate(classes) if c in PLANTABLE_GROUPS]].sum(1)
    ov_scores = binary_scores(bte, (ov_pl >= 0.5).astype(np.int64))
    log.info("OpenVINO(%s) held-out acc %.4f — %d px in %.0f ms (%.2f us/px)",
             cfg.model.device, ov_scores["accuracy"], len(Zte), ov_ms,
             ov_ms * 1000 / len(Zte))

    quant: dict[str, Any] = {"attempted": not args.no_quantize}
    if not args.no_quantize:
        try:
            import nncf

            calib = Ztr[rng.choice(len(Ztr), min(2048, len(Ztr)), replace=False)]
            batches = [calib[i: i + 64] for i in range(0, len(calib), 64)]
            ds = nncf.Dataset(batches)
            t0 = time.time()
            q_model = nncf.quantize(ov_model, ds, subset_size=len(batches))
            q_dir = out_dir / "landcover_int8"
            q_dir.mkdir(exist_ok=True)
            ov.save_model(q_model, str(q_dir / "landcover_int8.xml"))
            # ONNX vs OpenVINO IR is not a like-for-like size comparison (the
            # IR .xml is verbose text), so the FP32 model is also written as
            # IR and the weights-only .bin files are what get compared.
            f_dir = out_dir / "landcover_fp32_ir"
            f_dir.mkdir(exist_ok=True)
            ov.save_model(ov_model, str(f_dir / "landcover.xml"), compress_to_fp16=False)
            q_compiled = core.compile_model(q_model, cfg.model.device)
            q_proba = q_compiled(Zte)[_prob_port(q_compiled)]
            q_pl = q_proba[:, [i for i, c in enumerate(classes)
                               if c in PLANTABLE_GROUPS]].sum(1)
            q_scores = binary_scores(bte, (q_pl >= 0.5).astype(np.int64))
            fp32_kb = (f_dir / "landcover.bin").stat().st_size / 1e3
            int8_kb = (q_dir / "landcover_int8.bin").stat().st_size / 1e3
            quant |= {
                "ok": True,
                "seconds": round(time.time() - t0, 1),
                "calibration_pixels": int(len(calib)),
                "fp32_onnx_kb": round(onnx_path.stat().st_size / 1e3, 1),
                "fp32_ir_weights_kb": round(fp32_kb, 1),
                "int8_ir_weights_kb": round(int8_kb, 1),
                "int8": q_scores,
                "accuracy_delta": round(q_scores["accuracy"] - ov_scores["accuracy"], 4),
                "macro_f1_delta": round(q_scores["macro_f1"] - ov_scores["macro_f1"], 4),
            }
            log.info("NNCF INT8: acc %.4f (delta %+.4f), weights %.1f kB vs %.1f kB FP32",
                     q_scores["accuracy"], quant["accuracy_delta"], int8_kb, fp32_kb)
        except Exception as exc:  # noqa: BLE001 - report the failure, don't hide it
            quant |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            log.warning("NNCF INT8 quantization failed: %s", quant["error"])

    # --- 8. the evidence gate ----------------------------------------------
    passed = bool(
        ov_scores["accuracy"] > base_scores["accuracy"]
        and ov_scores["macro_f1"] > base_scores["macro_f1"]
    )

    report = {
        "task": "per-pixel land cover -> is this candidate site plantable ground",
        "city": cfg.city.name,
        "train_scene": {"id": scene["id"], "date": date, "cloud_pct": round(cloud, 2)},
        "labels": {
            "source": "ESA WorldCover 2021 v200 (10 m, AWS esa-worldcover, keyless)",
            "tiles": worldcover_tiles(bbox),
            "groups": GROUPS,
            "plantable_groups": [GROUPS[i] for i in sorted(PLANTABLE_GROUPS)],
            "worldcover_histogram": wc_hist,
        },
        "features": FEATURE_NAMES,
        "split": {
            "kind": "spatial block hold-out",
            "block_px": bp,
            "block_km": round(bp * 10 / 1000, 2),
            "buffer_px": args.buffer_px,
            "n_blocks": n_blocks,
            "n_test_blocks": n_test_blocks,
            "train_px_available": int(train_mask.sum()),
            "test_px_available": int(test_mask.sum()),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "class_balanced_training_draw": bool(args.balance),
            "plantable_share_train": round(float(btr.mean()), 4),
            "plantable_share_test": round(float(bte.mean()), 4),
        },
        "baseline_ndvi_threshold": base_scores,
        "model": {
            "kind": f"MLPClassifier{tuple(args.hidden)}",
            "fit_seconds": round(fit_s, 1),
            "sklearn": mlp_scores,
            "openvino": ov_scores | {"device": cfg.model.device,
                                     "us_per_pixel": round(ov_ms * 1000 / len(Zte), 3)},
            "per_group_f1": {
                g: multi_rows[i] | {"plantable": i in PLANTABLE_GROUPS}
                for i, g in enumerate(GROUPS)
            },
            "multiclass_accuracy": round(float(np.mean(mlp_multi == yte)), 4),
        },
        "reference_random_forest": rf_scores | {
            "note": "scored for honesty, never deployed: a tree ensemble is not "
                    "weight-quantizable and this repo deploys the OpenVINO path"
        },
        "quantization_nncf_int8": quant,
        "gate": {
            "criterion": "held-out accuracy AND macro-F1 both above the tuned "
                         "NDVI-threshold baseline, on spatially disjoint blocks",
            "passed": passed,
            "accuracy_gain": round(ov_scores["accuracy"] - base_scores["accuracy"], 4),
            "macro_f1_gain": round(ov_scores["macro_f1"] - base_scores["macro_f1"], 4),
        },
        "limits": [
            "labels are ESA WorldCover 2021; the site finder runs on 2026 imagery, "
            "so anything built, cleared or flooded since 2021 is mislabelled here "
            "and the held-out score does NOT measure that drift",
            "one scene, one date: seasonal spectra differ, and a monsoon-green "
            "fallow field looks nothing like the same field in May",
            "Sentinel-2 (UTM) and WorldCover (EPSG:4326) are both 10 m but offset "
            "by up to half a pixel (~5 m); label noise at parcel edges is real",
            "cropland counts as plantable because it is unsealed ground — it is "
            "not vacant, and ownership/tenure is not modelled anywhere",
            "the classifier says what the ground IS, never who owns it or whether "
            "planting is permitted; every site still needs a human",
        ],
    }
    norm = {
        "feature_names": FEATURE_NAMES,
        "groups": GROUPS,
        "plantable_groups": sorted(PLANTABLE_GROUPS),
        "classes": [int(c) for c in classes],
        "mu": [float(v) for v in mu],
        "sd": [float(v) for v in sd],
        "onnx": onnx_path.name,
        "report": {"gate": report["gate"], "openvino": report["model"]["openvino"],
                   "baseline": base_scores},
    }
    (out_dir / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- 9. say it plainly --------------------------------------------------
    print(f"\ntrain scene {scene['id']} ({date}, {cloud:.2f}% cloud) — "
          f"{len(ytr):,} train px / {len(yte):,} held-out px, spatially disjoint\n")
    print(f"{'':22s} {'accuracy':>9s} {'macro-F1':>9s}")
    print(f"{'NDVI threshold rule':22s} {base_scores['accuracy']:9.4f} "
          f"{base_scores['macro_f1']:9.4f}   <- the incumbent")
    print(f"{'MLP via OpenVINO':22s} {ov_scores['accuracy']:9.4f} "
          f"{ov_scores['macro_f1']:9.4f}")
    print(f"{'RandomForest (ref)':22s} {rf_scores['accuracy']:9.4f} "
          f"{rf_scores['macro_f1']:9.4f}   (not deployed)")
    if quant.get("ok"):
        q = quant["int8"]
        print(f"{'MLP INT8 (NNCF)':22s} {q['accuracy']:9.4f} {q['macro_f1']:9.4f}   "
              f"({quant['accuracy_delta']:+.4f} acc, weights "
              f"{quant['int8_ir_weights_kb']:.1f} kB vs "
              f"{quant['fp32_ir_weights_kb']:.1f} kB)")
    print("\nper-group F1 (held out):")
    for i, g in enumerate(GROUPS):
        r = multi_rows[i]
        tag = "plantable" if i in PLANTABLE_GROUPS else "NOT plantable"
        print(f"  {g:14s} F1 {r['f1']:.3f}  support {r['support']:>7,}  {tag}")
    verdict = (
        "PASSED — sites.py will use the classifier to FLAG (never drop) "
        "not-plantable candidates"
        if passed else
        "FAILED — the NDVI rule wins; the classifier stays in the repo with its "
        "score and is NOT used, like the benched forecast challenger"
    )
    print(f"\nevidence gate: {verdict}\nwrote {out_dir}/landcover.onnx + norm.json + report.json")
    return 0


def _prob_port(compiled: Any) -> Any:
    """The probability output of an skl2onnx classifier graph (the other one is
    the argmax label)."""
    for port in compiled.outputs:
        if port.get_partial_shape().rank.get_length() == 2:
            return port
    return compiled.outputs[-1]


if __name__ == "__main__":
    sys.exit(main())
