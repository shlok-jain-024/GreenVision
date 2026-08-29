"""Train the numeric forecaster on the real city panel and export it for
OpenVINO inference.

    python -m greenplan.forecast.train --config config/city.yaml            # MLP
    python -m greenplan.forecast.train --config config/city.yaml --model rf # forest

What it does, honestly:
  1. loads the same panel the engine uses (146 cells x 42 months for the
     shipped Ahmedabad data);
  2. builds one sample per (zone, cutoff) exactly like the backtest does —
     the model only ever sees history strictly BEFORE each cutoff;
  3. splits by TIME, not randomly: early cutoffs train, the latest cutoffs
     test, so the score is a genuine out-of-sample forecast score;
  4. trains one small model per metric — `--model mlp` (default) or
     `--model rf` (RandomForestRegressor) — on the standardized RESIDUAL
     from the same Theil-Sen + seasonality baseline;
  5. scores model vs that baseline on the SAME test tasks — the identical
     skill definition the LLM was scored with, so all contenders are
     directly comparable;
  6. writes {metric}.onnx + norm.json (+ report.json) to
     models/{city}/forecaster/, then VERIFIES the exported graphs through
     the OpenVINO runtime against sklearn's own predictions and records the
     measured max deviation in the report. The MLP converts via skl2onnx
     (plain MatMul graph); the forest is lowered to standard ONNX ops by
     `onnx_trees.py`, because skl2onnx emits ai.onnx.ml.TreeEnsembleRegressor
     which OpenVINO's ONNX frontend cannot convert (measured, 2026.3).

`--intel` calls sklearnex.patch_sklearn() before any sklearn import so
accelerated estimators dispatch to Intel oneDAL, and the report records
whether the patch was active next to the measured train time — re-running
with and without the flag IS the acceleration measurement. Honesty notes:
MLPRegressor is not on sklearnex's accelerated-estimator list (no neural
nets are), so for `--model mlp` the patch is a documented no-op;
RandomForestRegressor IS on the list. scikit-learn-intelex ships wheels for
Windows/Linux x86_64 only (Python <= 3.13) — on Apple-Silicon Macs pip finds
no distribution, the flag logs that, and training proceeds on stock sklearn.

Traffic note: the traffic stream is a disclosed inert placeholder (flat
value, MCDA weight 0). Its model trains and exports for interface
completeness, but its score is meaningless and is excluded from the combined
skill figure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..engine import load_panel
from ..features.trends import (
    METRIC_BOUNDS,
    METRICS,
    panel_stats,
    trend_seasonal_estimate,
)
from ..training.backtest import build_predict_context
from .features import FEATURE_NAMES, feature_vector

log = logging.getLogger(__name__)

REAL_METRICS = [m for m in METRICS if m != "traffic"]  # traffic = inert placeholder


class _NoMemory:
    """The forecaster's features use no memory records; satisfy the context
    builder's interface with an empty retrieval."""

    @staticmethod
    def retrieve(zone: str, feat: Any, k: int) -> list[dict[str, Any]]:
        return []


def build_dataset(cfg, panel) -> dict[str, Any]:
    t = cfg.training
    stats = panel_stats(panel)
    last_month = int(panel["month"].max())
    horizon = t.horizon_months
    memory = _NoMemory()

    X: dict[str, list[np.ndarray]] = {m: [] for m in METRICS}
    y: dict[str, list[float]] = {m: [] for m in METRICS}
    base: dict[str, list[float]] = {m: [] for m in METRICS}
    cutoffs: list[int] = []

    zones = sorted(panel["zone"].unique())
    for zone in zones:
        g = panel[panel["zone"] == zone]
        for cutoff in range(t.min_history_months, last_month - horizon + 1):
            target = g[g["month"] == cutoff + horizon]
            if target.empty or target[METRICS].isna().any(axis=None):
                continue
            if g[g["month"] < cutoff]["month"].nunique() < t.min_history_months:
                continue
            ctx = build_predict_context(
                panel, zone, cutoff, horizon, t.context_window,
                cfg.data.start_month, stats, memory, 0,
            )
            for m in METRICS:
                X[m].append(feature_vector(ctx, m))
                y[m].append(float(target.iloc[0][m]))
                pts = [
                    (h["month"], h["cal_month"], h[m])
                    for h in ctx["history"] if h[m] is not None
                ]
                est = trend_seasonal_estimate(pts, ctx["target_month"], ctx["season"])
                if est is None:
                    est = ctx["inputs"][f"{m}_latest"]
                base[m].append(float(np.clip(est, *METRIC_BOUNDS[m])))
            cutoffs.append(cutoff)

    return {
        "X": {m: np.vstack(X[m]) for m in METRICS},
        "y": {m: np.asarray(y[m], dtype=np.float32) for m in METRICS},
        "base": {m: np.asarray(base[m], dtype=np.float32) for m in METRICS},
        "cutoffs": np.asarray(cutoffs),
        "stats": stats,
    }


def verify_openvino_parity(
    out_dir: Path, fitted: dict[str, Any], z_test: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Compile each exported ONNX with the OpenVINO runtime — f32 pinned,
    exactly as the deploy path (`OVForecaster`) compiles it — run it on the
    held-out feature matrix, and measure the worst deviation from sklearn's
    own predictions. Units are standardized residuals, the networks' output
    scale. Measured, not assumed: the result lands in report.json."""
    try:
        import openvino as ov  # noqa: PLC0415
    except ImportError:
        return {"measured": False, "reason": "openvino not installed"}

    core = ov.Core()
    diffs: dict[str, float] = {}
    for m, net in fitted.items():
        compiled = core.compile_model(
            core.read_model(str(out_dir / f"{m}.onnx")), "CPU",
            {"INFERENCE_PRECISION_HINT": "f32"},
        )
        got = next(iter(compiled(z_test[m]).values())).ravel()
        want = np.asarray(net.predict(z_test[m]), dtype=np.float64).ravel()
        diffs[m] = float(np.max(np.abs(got - want))) if len(want) else 0.0
    worst = max(diffs.values())
    return {
        "measured": True,
        "device": "CPU",
        "max_abs_diff": {m: round(d, 8) for m, d in diffs.items()},
        "worst_abs_diff": round(worst, 8),
        "tolerance": 1e-3,
        "ok": worst < 1e-3,
    }


def _try_patch_sklearnex(requested: bool, model: str) -> dict[str, Any]:
    """Activate Intel Extension for Scikit-learn BEFORE sklearn imports, and
    report what actually happened — never assume the patch took effect."""
    info: dict[str, Any] = {"requested": requested, "active": False}
    if not requested:
        return info
    try:
        import sklearnex  # noqa: PLC0415

        sklearnex.patch_sklearn()
        info["active"] = True
        info["version"] = getattr(sklearnex, "__version__", "?")
        log.info(
            "scikit-learn-intelex %s active — accelerated estimators dispatch "
            "to oneDAL (set SKLEARNEX_VERBOSE=INFO to see per-call dispatch)",
            info["version"],
        )
    except ImportError:
        info["error"] = "scikit-learn-intelex not installed"
        log.warning(
            "--intel requested but scikit-learn-intelex is not installed — "
            "it ships wheels for Windows/Linux x86_64 only (Python <= 3.13); "
            "on this platform pip may find no distribution at all. "
            "Training on stock scikit-learn."
        )
    if model == "mlp":
        log.info(
            "note: MLPRegressor is NOT on sklearnex's accelerated-estimator "
            "list, so --intel is a documented no-op for --model mlp; use "
            "--model rf to exercise an accelerated estimator"
        )
    return info


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Train the OpenVINO numeric forecaster")
    ap.add_argument("--config", default="config/city.yaml")
    ap.add_argument("--model", choices=["mlp", "rf"], default="mlp",
                    help="challenger family: mlp (skl2onnx export) or rf "
                    "(RandomForestRegressor, sklearnex-accelerated on Intel "
                    "hardware, custom standard-op ONNX export)")
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 32],
                    help="MLP hidden layer sizes (--model mlp only)")
    ap.add_argument("--trees", type=int, default=200,
                    help="forest size (--model rf only)")
    ap.add_argument("--max-depth", type=int, default=None,
                    help="forest depth cap (--model rf only; default unlimited)")
    ap.add_argument("--intel", action="store_true",
                    help="patch sklearn with scikit-learn-intelex (oneDAL) "
                    "before training; the report records whether it was "
                    "actually active next to the measured train time")
    ap.add_argument("--test-cutoffs", type=int, default=3,
                    help="how many of the LATEST cutoff months form the test set")
    args = ap.parse_args()

    intel_info = _try_patch_sklearnex(args.intel, args.model)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor

        from .onnx_trees import forest_to_onnx
    except ImportError:
        print(
            "error: training needs scikit-learn and skl2onnx:\n"
            "    pip install scikit-learn skl2onnx",
            file=sys.stderr,
        )
        return 2

    cfg = load_config(args.config)
    panel = load_panel(cfg, mock=False)
    ds = build_dataset(cfg, panel)
    n = len(ds["cutoffs"])

    split = sorted(set(ds["cutoffs"]))[-args.test_cutoffs]
    train_idx = ds["cutoffs"] < split
    test_idx = ~train_idx
    log.info(
        "%d samples: %d train (cutoff < %d), %d test — split by time, not randomly",
        n, int(train_idx.sum()), split, int(test_idx.sum()),
    )

    out_dir = Path(cfg.model.forecaster_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm: dict[str, Any] = {"feature_names": FEATURE_NAMES, "mu": {}, "sd": {}}
    report: dict[str, Any] = {
        "model": args.model,
        "model_params": (
            {"hidden": args.hidden} if args.model == "mlp"
            else {"trees": args.trees, "max_depth": args.max_depth}
        ),
        "sklearnex": intel_info,
        "n_train": int(train_idx.sum()), "n_test": int(test_idx.sum()),
        "horizon_months": cfg.training.horizon_months, "per_metric": {},
    }
    skills = []
    train_seconds = 0.0
    fitted: dict[str, Any] = {}
    z_test: dict[str, np.ndarray] = {}

    for m in METRICS:
        Xm, ym, bm = ds["X"][m], ds["y"][m], ds["base"][m]
        mu = Xm[train_idx].mean(axis=0)
        sd = Xm[train_idx].std(axis=0) + 1e-9
        Z = (Xm - mu) / sd

        # The network learns the RESIDUAL from the Theil-Sen + seasonal
        # baseline, standardized. At worst it learns nothing and collapses to
        # the baseline (skill ~ 0); it can only add value, never quietly
        # replace an honest estimate with a worse one.
        resid = ym - bm
        r_mu = float(resid[train_idx].mean())
        r_sd = float(resid[train_idx].std() + 1e-9)

        if args.model == "rf":
            net = RandomForestRegressor(
                n_estimators=args.trees, max_depth=args.max_depth,
                random_state=cfg.run.seed, n_jobs=-1,
            )
        else:
            net = MLPRegressor(
                hidden_layer_sizes=tuple(args.hidden),
                alpha=1e-2, max_iter=4000, random_state=cfg.run.seed,
            )
        t0 = time.perf_counter()
        net.fit(Z[train_idx], (resid[train_idx] - r_mu) / r_sd)
        fit_s = time.perf_counter() - t0
        train_seconds += fit_s

        pred_resid = net.predict(Z[test_idx]) * r_sd + r_mu
        pred = np.clip(bm[test_idx] + pred_resid, *METRIC_BOUNDS[m])
        mae = float(np.mean(np.abs(pred - ym[test_idx])))
        base_mae = float(np.mean(np.abs(bm[test_idx] - ym[test_idx])))
        std = ds["stats"][m]["std"]
        skill = float((base_mae - mae) / std)  # + = beats the baseline
        report["per_metric"][m] = {
            "test_mae": round(mae, 4),
            "baseline_mae": round(base_mae, 4),
            "skill_vs_baseline": round(skill, 4),
            "train_seconds": round(fit_s, 3),
            "inert_placeholder": m == "traffic",
        }
        if m in REAL_METRICS:
            skills.append(skill)

        if args.model == "rf":
            # skl2onnx emits ai.onnx.ml.TreeEnsembleRegressor for forests,
            # and OpenVINO's ONNX frontend has no conversion rule for it
            # (measured on 2026.3) — so lower the trees to standard ops.
            onx = forest_to_onnx(net, len(FEATURE_NAMES), name=f"{m}_forest")
        else:
            onx = convert_sklearn(
                net, initial_types=[("X", FloatTensorType([None, len(FEATURE_NAMES)]))]
            )
        (out_dir / f"{m}.onnx").write_bytes(onx.SerializeToString())
        fitted[m] = net
        z_test[m] = Z[test_idx].astype(np.float32)
        norm["mu"][m] = [float(v) for v in mu]
        norm["sd"][m] = [float(v) for v in sd]
        norm.setdefault("resid_mu", {})[m] = r_mu
        norm.setdefault("resid_sd", {})[m] = r_sd

    report["skill_combined"] = round(float(np.mean(skills)), 4)
    report["train_seconds_total"] = round(train_seconds, 3)
    report["openvino_parity"] = verify_openvino_parity(out_dir, fitted, z_test)
    norm["report"] = report
    (out_dir / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'metric':9s} {'test MAE':>9s} {'baseline':>9s} {'skill':>7s}")
    for m in METRICS:
        r = report["per_metric"][m]
        tag = "  (inert placeholder, excluded)" if r["inert_placeholder"] else ""
        print(f"{m:9s} {r['test_mae']:9.4f} {r['baseline_mae']:9.4f} "
              f"{r['skill_vs_baseline']:+7.3f}{tag}")
    verdict = (
        "POSITIVE — provider `hybrid` will deploy this network"
        if report["skill_combined"] > 0
        else "NEGATIVE — provider `hybrid` keeps the statistical forecaster "
        "(the honest champion) and benches this challenger"
    )
    parity = report["openvino_parity"]
    parity_line = (
        f"OpenVINO parity vs sklearn (held-out set, f32): worst "
        f"|diff| {parity['worst_abs_diff']:.2e} "
        f"{'OK' if parity['ok'] else 'FAILED'} (tolerance 1e-3)"
        if parity["measured"]
        else f"OpenVINO parity NOT measured: {parity['reason']}"
    )
    intel_line = (
        "sklearnex (Intel oneDAL): active"
        if intel_info["active"]
        else "sklearnex (Intel oneDAL): "
        + ("requested but unavailable — stock sklearn" if intel_info["requested"] else "off")
    )
    print(
        f"\nmodel: {args.model} {report['model_params']} — "
        f"train time {train_seconds:.1f}s ({intel_line})\n"
        f"{parity_line}\n"
        f"combined skill vs Theil-Sen+seasonal baseline (real metrics): "
        f"{report['skill_combined']:+.3f}\n"
        f"(same definition scored the LLM at -0.333 — see README table)\n"
        f"verdict: {verdict}\n"
        f"wrote {out_dir}/[metric].onnx + norm.json + report.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
