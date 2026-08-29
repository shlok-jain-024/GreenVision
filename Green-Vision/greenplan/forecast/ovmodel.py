"""Inference for the trained forecaster — through the OpenVINO runtime — and
the hybrid model that pairs it with the language model.

`OVForecaster` loads the ONNX networks that `greenplan.forecast.train` wrote
and compiles them with OpenVINO for the device named in config (`CPU`, `GPU`,
`NPU`, `AUTO` — same knob as the LLM). Inference is a single matrix pass, so
a full 146-cell city forecast is effectively instant.

`HybridModel` implements the same predict/lesson/recommend interface as
`ReasoningModel` and `MockModel`, split by what each side is measurably good
at (`build_model` picks the numeric side from the bake-off evidence):

  * predict      -> the best MEASURED numeric model (statistical forecaster
                    today; a trained network the day one reports positive
                    held-out skill)
  * lesson       -> the numeric side (deterministic, fast, and the recommend
                    prompt still gets genuine hit/miss context)
  * recommend    -> the language model (words and species are its job)
  * project_note -> the language model
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..features.trends import METRIC_BOUNDS as BOUNDS
from ..features.trends import METRICS
from .features import FEATURE_NAMES, feature_vector

log = logging.getLogger(__name__)


class OVForecaster:
    """The trained per-metric networks, compiled by OpenVINO."""

    def __init__(self, forecaster_dir: str | Path, device: str = "CPU") -> None:
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install
            raise RuntimeError(
                "the trained forecaster needs the OpenVINO runtime:\n"
                "    pip install openvino-genai"
            ) from exc

        d = Path(forecaster_dir)
        norm_path = d / "norm.json"
        if not norm_path.exists():
            raise RuntimeError(
                f"no trained forecaster at {d}. Train one with:\n"
                "    python -m greenplan.forecast.train --config config/city.yaml"
            )
        self.norm = json.loads(norm_path.read_text(encoding="utf-8"))
        if self.norm.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError(
                f"forecaster at {d} was trained with different features — retrain it:\n"
                "    python -m greenplan.forecast.train --config config/city.yaml"
            )

        core = ov.Core()
        self.compiled: dict[str, Any] = {}
        for m in METRICS:
            # f32 pinned: ARM CPU plugins default to f16 inference, which
            # flips tree-split comparisons on values near a threshold
            # (measured: 0.05 max error on a random forest). The nets are
            # tiny; precision costs nothing here.
            self.compiled[m] = core.compile_model(
                core.read_model(str(d / f"{m}.onnx")), device,
                {"INFERENCE_PRECISION_HINT": "f32"},
            )
        log.info(
            "trained forecaster loaded from %s on %s (test skill vs baseline: %s)",
            d, device, self.norm.get("report", {}).get("skill_combined", "?"),
        )

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        for m in METRICS:
            feat = feature_vector(ctx, m)
            mu = np.asarray(self.norm["mu"][m], dtype=np.float32)
            sd = np.asarray(self.norm["sd"][m], dtype=np.float32)
            z = ((feat - mu) / sd)[None, :]
            # The network predicts the standardized RESIDUAL from the
            # Theil-Sen + seasonal baseline (feature 0 of the vector);
            # de-standardize and add it back.
            r = float(next(iter(self.compiled[m](z).values())).ravel()[0])
            y = feat[0] + r * self.norm["resid_sd"][m] + self.norm["resid_mu"][m]
            out[m] = float(np.clip(y, *BOUNDS[m]))
        return out

    @staticmethod
    def lesson(ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        """Deterministic error summary. Keeps the backtest loop fast while the
        recommend prompt still receives genuine hit/miss context."""
        parts = []
        for m in METRICS:
            err = float(pred[m]) - float(actual[m])
            fmt = ".3f" if m == "ndvi" else ".1f"
            parts.append(f"{m} {'over' if err >= 0 else 'under'} by {abs(err):{fmt}}")
        return (
            f"Zone {ctx['zone']} season {ctx['season']} h{ctx['horizon']}: "
            + ", ".join(parts) + "."
        )


class HybridModel:
    """Numbers from the trained network, words from the language model."""

    def __init__(self, forecaster: OVForecaster, llm: Any) -> None:
        self.forecaster = forecaster
        self.llm = llm

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        return self.forecaster.predict(ctx)

    def lesson(self, ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        return self.forecaster.lesson(ctx, pred, actual)

    def recommend(self, ranked_rows: list[dict[str, Any]], lessons: list[str]) -> list[dict[str, Any]]:
        return self.llm.recommend(ranked_rows, lessons)

    def project_note(self, city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
        return self.llm.project_note(city, n_months, sample_rows)
