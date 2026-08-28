"""Typed configuration loaded from config/city.yaml, with CLI overrides."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


@dataclass
class CityCfg:
    name: str = "Bengaluru"
    bbox: tuple[float, float, float, float] = (77.46, 12.85, 77.75, 13.10)


@dataclass
class GridCfg:
    h3_resolution: int = 9


@dataclass
class DataCfg:
    start_month: int = 1
    months_history: int = 60
    n_mock_zones: int = 40


@dataclass
class AdaptersCfg:
    green_cover: str = "mock"
    traffic: str = "mock"
    aqi: str = "mock"


@dataclass
class SitesCfg:
    """Bare-land site finder (Esri 10 m Land Cover 'Bare ground' patches)."""
    enabled: bool = True
    # CSV of bare-patch centroids (lat, lon, patch_area_m2). Drives both the
    # per-zone site coordinates and the real plantable-space fraction.
    candidates_csv: str | None = None
    min_patch_m2: float = 100.0
    max_sites_per_zone: int = 15
    # "temporal plantability" thresholds on the NDVI stack — CALIBRATE to ground
    ndvi_ever_green: float = 0.25
    ndvi_never_green: float = 0.15
    # Legacy: a pre-computed 0..1 plantable-fraction CSV (zone/lat-lon + value).
    # candidates_csv is preferred; this stays for hand-supplied fractions.
    plantable_csv: str | None = None


@dataclass
class SoilCfg:
    """Soil intelligence: SoilGrids chemistry/texture + NASA SMAP moisture."""
    soilgrids_csv: str | None = None
    moisture_csv: str | None = None


@dataclass
class RetrainCfg:
    policy: str = "if_new_months"  # always | if_new_months | never
    new_months_threshold: int = 3


@dataclass
class TrainingCfg:
    iterations: int = 60
    horizon_months: int = 6
    min_history_months: int = 18
    context_window: int = 24
    memory_k: int = 8
    memory_path: str = "models/memory.jsonl"
    retrain: RetrainCfg = field(default_factory=RetrainCfg)


@dataclass
class MCDACfg:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "aqi_worsening": 0.30,
            "traffic_worsening": 0.20,
            "ndvi_decline": 0.25,
            "low_green_cover": 0.15,
            "plantable_space": 0.10,
        }
    )
    top_n: int = 10
    # How many ranked cells reach recommendations.geojson. 0 = all of them.
    # Separate from top_n because top_n bounds how many zones the language
    # model is asked to justify (expensive), while this bounds what the map
    # can draw (free). Tying them together left the Priority view showing 10
    # cells out of 146 and looking like the engine had barely run.
    geojson_n: int = 0


@dataclass
class ModelCfg:
    provider: str = "openrouter"  # openrouter | nvidia | openvino | mock
    name: str = "deepseek/deepseek-chat"
    # Empty -> use the provider's default endpoint (see PROVIDERS in client.py).
    base_url: str = ""
    temperature: float = 0.2
    timeout_s: float = 60.0
    max_retries: int = 4
    json_repair_attempts: int = 3
    # Stream tokens (SSE). Recommended for slow reasoning models (e.g. NVIDIA
    # NIM glm/deepseek) so the per-chunk read timeout replaces one long wait.
    stream: bool = False

    # --- provider: openvino -------------------------------------------------
    # Local inference through Intel's OpenVINO runtime. No API key, no network,
    # no per-token cost: the weights sit on disk in OpenVINO IR, compressed to
    # INT4 so a 1.5B model runs on an ordinary CPU.
    model_dir: str = "models/openvino/qwen2.5-1.5b-instruct-int4-ov"
    device: str = "CPU"           # CPU | GPU | NPU | AUTO — any OpenVINO device
    max_new_tokens: int = 2048
    # Chat template. "chatml" suits Qwen/Yi/most OpenVINO-converted instruct
    # models; "llama3" suits Llama-3.x; "tokenizer" defers to the model's own
    # template when the bundled tokenizer ships one.
    chat_template: str = "chatml"


@dataclass
class RunCfg:
    seed: int = 42
    outputs_dir: str = "outputs"


@dataclass
class Config:
    city: CityCfg = field(default_factory=CityCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    data: DataCfg = field(default_factory=DataCfg)
    adapters: AdaptersCfg = field(default_factory=AdaptersCfg)
    sites: SitesCfg = field(default_factory=SitesCfg)
    soil: SoilCfg = field(default_factory=SoilCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    mcda: MCDACfg = field(default_factory=MCDACfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    run: RunCfg = field(default_factory=RunCfg)
    # Directory the config file lives under; relative paths resolve against its parent.
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.base_dir / p


def _build(dc_type: type, raw: dict[str, Any] | None) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown keys and
    recursing into nested dataclass fields. A null section (e.g. a YAML block
    with only comments) is treated as empty, so all defaults apply."""
    raw = raw or {}
    kwargs: dict[str, Any] = {}
    for f in dc_type.__dataclass_fields__.values():  # type: ignore[attr-defined]
        if f.name not in raw:
            continue
        value = raw[f.name]
        if hasattr(f.type, "__dataclass_fields__") or f.name == "retrain":
            nested_type = {"retrain": RetrainCfg}.get(f.name, f.type)
            value = _build(nested_type, value or {})
        elif f.name == "bbox":
            value = tuple(float(v) for v in value)
        kwargs[f.name] = value
    return dc_type(**kwargs)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = Config(
        city=_build(CityCfg, raw.get("city", {})),
        grid=_build(GridCfg, raw.get("grid", {})),
        data=_build(DataCfg, raw.get("data", {})),
        adapters=_build(AdaptersCfg, raw.get("adapters", {})),
        sites=_build(SitesCfg, raw.get("sites", {})),
        soil=_build(SoilCfg, raw.get("soil", {})),
        training=_build(TrainingCfg, raw.get("training", {})),
        mcda=_build(MCDACfg, raw.get("mcda", {})),
        model=_build(ModelCfg, raw.get("model", {})),
        run=_build(RunCfg, raw.get("run", {})),
        base_dir=path.resolve().parent.parent,
    )
    # "{city}" in paths expands to a slug of the city name, so each city keeps
    # its own memory and outputs (add a city = add a config file, nothing else)
    slug = re.sub(r"[^a-z0-9]+", "-", cfg.city.name.lower()).strip("-") or "city"
    cfg.training.memory_path = cfg.training.memory_path.replace("{city}", slug)
    cfg.run.outputs_dir = cfg.run.outputs_dir.replace("{city}", slug)
    return cfg


def apply_env_overrides(cfg: Config) -> Config:
    """Apply the environment variables that are allowed to override the YAML.

    MODEL_PROVIDER is the only one so far. The Dockerfile has shipped
    `ENV MODEL_PROVIDER=mock` and documented it in six lines of comments while
    nothing in the codebase read it, so setting it was a silent no-op and the
    container ran whatever config/city.yaml said.

    It also closes a real seam on the CLI side. `--mock` is NOT the offline
    switch it looks like: it swaps the DATA adapters to synthetic as well, so a
    user who wanted the real Ahmedabad CSVs scored by the offline writer had no
    option but to hand-edit `model.provider` into a tracked config file.

    Nothing is validated beyond non-emptiness, deliberately: an unknown
    provider name already fails gracefully at build_model — degrading to the
    offline engine under strict=False (the server), and telling the user
    plainly under strict=True (the CLI).

    Every entry point calls this immediately after load_config, so `greenplan
    run`, `greenplan horizon` and `greenplan.server` all behave the same way.
    `cfg` is mutated in place and returned for convenience.
    """
    provider = (os.environ.get("MODEL_PROVIDER") or "").strip()
    if provider:
        log.info(
            "MODEL_PROVIDER=%s overrides config model.provider=%s",
            provider, cfg.model.provider,
        )
        cfg.model.provider = provider
    return cfg
