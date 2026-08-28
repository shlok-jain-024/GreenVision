"""Reasoning models.

One interface — `predict` / `lesson` / `recommend` — over three very different
back ends. Inference ONLY, in every case: nothing here trains or fine-tunes
anything. All "learning" happens in the prompt, via the memory store.

* OpenVINOClient — LOCAL inference on Intel's OpenVINO runtime. The weights
  sit on disk in OpenVINO IR compressed to INT4, so a 1.5B instruct model
  loads in seconds and answers on an ordinary CPU with no GPU, no API key and
  no network. This is the default: it is what lets the whole pipeline run on
  free public data with no paid service anywhere in it, and it keeps every
  figure about a city on the machine that is planning that city.
* OpenRouterClient — a hosted chat-completions endpoint (OpenRouter or NVIDIA
  NIM), for when a larger model is worth the round trip. Key comes from an
  env var, never hard-coded.
* MockModel — a fully offline stand-in (used by --mock) that fills the same
  role numerically: trend + seasonal extrapolation, bias-corrected using the
  retrieved memory records. It exercises the exact same pipeline, including
  genuine use of memory, at zero cost and with no model at all.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import requests

from ..config import ModelCfg
from ..features.soil import species_soil_ok
from ..features.trends import METRIC_BOUNDS as BOUNDS
from ..features.trends import METRICS, trend_seasonal_estimate
from . import prompts
from .species import SPECIES_KB, kb_markdown_table, validate_selection

log = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    pass


# Provider registry: env var holding the API key + default base URL. All of
# these speak the OpenAI chat-completions protocol, so one client serves them.
PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "nvidia": {  # NVIDIA NIM — https://integrate.api.nvidia.com/v1
        "key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
    },
}


def extract_json(text: str, allow_list: bool = False) -> Any:
    """Strict-JSON parsing that tolerates the usual LLM wrapping: reasoning
    tags, code fences, and prose around the object.

    With allow_list, a top-level ARRAY is also accepted. Small local models
    routinely emit the inner list and drop the wrapper object; the caller
    re-wraps it under the key the schema expects rather than throwing away
    an otherwise-correct answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?", "", text)
    if allow_list:
        obj_at, arr_at = text.find("{"), text.find("[")
        if arr_at != -1 and (obj_at == -1 or arr_at < obj_at):
            return _balanced(text, arr_at, "[", "]")
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model reply")
    return _balanced(text, start, "{", "}")


def _balanced(text: str, start: int, open_c: str, close_c: str) -> Any:
    """Return the first balanced JSON value starting at `start`."""
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON value in model reply")


def _require_number(obj: dict[str, Any], key: str) -> float:
    v = obj.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"key '{key}' must be a number, got {v!r}")
    return float(v)


class OpenRouterClient:
    """Thin chat-completions client: retries with backoff, timeout, and
    strict-JSON extraction with repair re-asks."""

    def __init__(self, cfg: ModelCfg) -> None:
        provider = PROVIDERS.get(cfg.provider)
        if provider is None:
            raise OpenRouterError(
                f"unknown model.provider {cfg.provider!r}; "
                f"expected one of {sorted(PROVIDERS)} or 'mock'"
            )
        key_env = provider["key_env"]
        key = os.environ.get(key_env)
        if not key:
            raise OpenRouterError(
                f"{key_env} is not set. Export your {cfg.provider} API key, "
                "or run with --mock (no key, no cost)."
            )
        self.cfg = cfg
        self._provider = cfg.provider
        self._key = key
        self._session = requests.Session()
        # base_url from config wins if given; otherwise the provider default.
        base = cfg.base_url or provider["base_url"]
        self._url = base.rstrip("/") + "/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        if self._provider == "openrouter":  # OpenRouter-only ranking headers
            headers["HTTP-Referer"] = "https://github.com/local/greengrid"
            headers["X-Title"] = "GreenGrid"
        return headers

    @staticmethod
    def _read_stream(resp: requests.Response) -> str:
        """Assemble assistant text from an OpenAI-style SSE stream. The read
        timeout applies *per chunk*, so a slow model that keeps emitting tokens
        won't trip a single-shot timeout the way a non-streamed call does."""
        parts: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw[6:] if raw.startswith("data: ") else raw
            if line.strip() == "[DONE]":
                break
            try:
                delta = json.loads(line)["choices"][0]["delta"]
            except (KeyError, IndexError, ValueError, TypeError):
                continue  # keep-alive / non-delta line
            piece = delta.get("content")
            if piece:
                parts.append(piece)
        return "".join(parts)

    def _post(self, payload: dict[str, Any]) -> str:
        """POST with retry/backoff and return the assistant message text.
        Streams when cfg.stream is set (needed for slow reasoning models)."""
        stream = getattr(self.cfg, "stream", False)
        if stream:
            payload = {**payload, "stream": True}
        delay = 1.5
        last = "no attempt made"
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                resp = self._session.post(
                    self._url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.cfg.timeout_s,
                    stream=stream,
                )
            except requests.RequestException as exc:
                last = f"network error: {exc}"
                log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                time.sleep(delay)
                delay *= 2
                continue
            if resp.status_code == 200:
                if stream:
                    try:
                        return self._read_stream(resp)
                    except requests.RequestException as exc:
                        last = f"stream interrupted: {exc}"
                        log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                        time.sleep(delay)
                        delay *= 2
                        continue
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError) as exc:
                    raise OpenRouterError(f"unexpected {self._provider} response shape: {exc}")
            if resp.status_code == 400 and "response_format" in resp.text and "response_format" in payload:
                log.warning("model rejected response_format; retrying without it")
                payload = {k: v for k, v in payload.items() if k != "response_format"}
                continue
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            if resp.status_code in (408, 429, 500, 502, 503, 524):
                log.warning("%s attempt %d failed (%s); retrying", self._provider, attempt, last)
                time.sleep(delay)
                delay *= 2
                continue
            raise OpenRouterError(f"{self._provider} request failed: {last}")
        raise OpenRouterError(f"{self._provider} retries exhausted; last error: {last}")

    def chat_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None] | None = None,
        list_key: str | None = None,
    ) -> dict[str, Any]:
        # list_key is accepted for interface parity with OpenVINOClient; hosted
        # models honour response_format and return the wrapper object already.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = "no attempt made"
        for _ in range(self.cfg.json_repair_attempts):
            text = self._post(
                {
                    "model": self.cfg.name,
                    "messages": messages,
                    "temperature": self.cfg.temperature,
                    "response_format": {"type": "json_object"},
                }
            )
            try:
                obj = extract_json(text)
                if validate:
                    validate(obj)
                return obj
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                log.warning("invalid JSON from model (%s); re-asking", last_err)
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {last_err}. "
                            "Reply again with STRICT JSON only, exactly matching the "
                            "schema in the system message. No prose, no code fences."
                        ),
                    }
                )
        raise OpenRouterError(f"model never produced valid JSON: {last_err}")


class OpenVINOClient:
    """Local inference through Intel's OpenVINO runtime — same `chat_json`
    contract as OpenRouterClient, so every model class above works unchanged.

    The weights live on disk in OpenVINO IR, compressed to INT4. That is what
    makes a 1.5B instruct model practical on an ordinary CPU: ~1 GB on disk
    instead of ~3 GB at FP16, a couple of seconds to load, and a few seconds
    per reply with no GPU. Nothing leaves the machine and there is no API key,
    which is what lets the whole pipeline run on free public data alone.

    `device` is passed straight to OpenVINO, so the same build targets CPU,
    an integrated GPU, or an NPU without any code change.
    """

    # Prompt formats for the instruct models OpenVINO ships pre-converted.
    _TEMPLATES = {
        "chatml": (
            "<|im_start|>system\n{system}<|im_end|>\n"
            "<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "llama3": (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
    }

    def __init__(self, cfg: ModelCfg) -> None:
        try:
            import openvino_genai  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - depends on install
            raise OpenRouterError(
                "provider 'openvino' needs the OpenVINO runtime. Install it with:\n"
                "    pip install openvino-genai huggingface_hub\n"
                "then fetch a pre-compressed model with:\n"
                "    python scripts/fetch_openvino_model.py"
            ) from exc

        self._genai = openvino_genai
        self.cfg = cfg

        model_dir = Path(cfg.model_dir)
        if not model_dir.is_absolute():
            model_dir = Path.cwd() / model_dir
        if not (model_dir / "openvino_model.xml").exists():
            raise OpenRouterError(
                f"no OpenVINO model at {model_dir}. Fetch one with:\n"
                "    python scripts/fetch_openvino_model.py"
            )

        log.info("loading OpenVINO model %s on %s", model_dir.name, cfg.device)
        t0 = time.time()
        self.pipe = openvino_genai.LLMPipeline(str(model_dir), cfg.device)
        log.info("OpenVINO pipeline ready in %.1fs", time.time() - t0)

    def _render(self, system: str, user: str) -> str:
        name = self.cfg.chat_template
        if name == "tokenizer":
            try:
                return self.pipe.get_tokenizer().apply_chat_template(
                    [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
                    add_generation_prompt=True,
                )
            except Exception:  # tokenizer ships no template — fall back
                name = "chatml"
        return self._TEMPLATES.get(name, self._TEMPLATES["chatml"]).format(
            system=system, user=user
        )

    def _generate(self, system: str, user: str) -> str:
        gen = self._genai.GenerationConfig()
        gen.max_new_tokens = self.cfg.max_new_tokens
        # Greedy decoding at temperature 0 — strict-JSON output is a parsing
        # task, not a creative one, and sampling only adds failure modes.
        if self.cfg.temperature > 0:
            gen.do_sample = True
            gen.temperature = self.cfg.temperature
        else:
            gen.do_sample = False
        return str(self.pipe.generate(self._render(system, user), gen))

    def chat_json(
        self,
        system: str,
        user: str,
        validate: Callable[[dict[str, Any]], None] | None = None,
        list_key: str | None = None,
    ) -> dict[str, Any]:
        """Mirrors OpenRouterClient.chat_json, including the re-ask loop. A
        small local model needs that loop more often than a hosted one, which
        is exactly why it lives in both clients.

        `list_key` names the single array the schema expects. A small model
        very often returns that array bare, dropping the wrapper object; when
        it does, we re-wrap rather than discard a correct answer."""
        last_err = "no attempt made"
        sys_prompt, usr_prompt = system, user
        for _ in range(self.cfg.json_repair_attempts):
            text = self._generate(sys_prompt, usr_prompt)
            try:
                obj = extract_json(text, allow_list=list_key is not None)
                if isinstance(obj, list):
                    obj = {list_key: obj}
                if validate:
                    validate(obj)
                return obj
            except (ValueError, json.JSONDecodeError) as exc:
                last_err = str(exc)
                log.warning("invalid JSON from local model (%s); re-asking", last_err)
                usr_prompt = (
                    f"{user}\n\n---\nYour previous reply was invalid: {last_err}\n"
                    "Reply again with STRICT JSON only, exactly matching the schema "
                    "in the system message. No prose, no code fences, no commentary."
                )
        raise OpenRouterError(f"local model never produced valid JSON: {last_err}")


class ReasoningModel:
    """The reasoning interface, over any client exposing `chat_json`.

    Identical behaviour whether the tokens come from a hosted endpoint or from
    OpenVINO on this machine — swapping `model.provider` in the config changes
    where inference happens and nothing else.
    """

    def __init__(self, cfg: ModelCfg, client: Any | None = None) -> None:
        self.client = client if client is not None else OpenRouterClient(cfg)

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        def _validate(obj: dict[str, Any]) -> None:
            for m in METRICS:
                _require_number(obj, m)

        obj = self.client.chat_json(prompts.PREDICT_SYSTEM, prompts.predict_user(ctx), _validate)
        return {
            m: float(np.clip(_require_number(obj, m), *BOUNDS[m])) for m in METRICS
        }

    def lesson(self, ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        def _validate(obj: dict[str, Any]) -> None:
            if not isinstance(obj.get("lesson"), str) or not obj["lesson"].strip():
                raise ValueError("key 'lesson' must be a non-empty string")

        obj = self.client.chat_json(
            prompts.LESSON_SYSTEM, prompts.lesson_user(ctx, pred, actual), _validate
        )
        return obj["lesson"].strip()

    def recommend(
        self, ranked_rows: list[dict[str, Any]], lessons: list[str]
    ) -> list[dict[str, Any]]:
        given_order = [r["zone"] for r in ranked_rows]

        def _validate(obj: dict[str, Any]) -> None:
            zones = obj.get("zones")
            if not isinstance(zones, list) or not zones:
                raise ValueError("key 'zones' must be a non-empty list")
            for z in zones:
                if z.get("zone") not in given_order:
                    raise ValueError(f"unknown zone id {z.get('zone')!r}")
                picked = validate_selection(list(z.get("species") or []))
                if len(picked) < 3:
                    raise ValueError(
                        f"zone {z.get('zone')}: need 3-5 species chosen EXACTLY from the "
                        f"table's `common` column; valid picks were {picked}"
                    )

        obj = self.client.chat_json(
            prompts.RECOMMEND_SYSTEM,
            prompts.recommend_user(ranked_rows, lessons, kb_markdown_table()),
            _validate,
            list_key="zones",
        )
        by_zone = {z["zone"]: z for z in obj["zones"]}
        out = []
        for row in ranked_rows:  # numeric order is authoritative, never the model's
            z = by_zone.get(row["zone"], {})
            out.append(
                {
                    "zone": row["zone"],
                    "justification": str(z.get("justification", "")).strip()
                    or "(model gave no justification)",
                    "species": validate_selection(list(z.get("species") or [])),
                }
            )
        return out

    def project_note(self, city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
        def _validate(obj: dict[str, Any]) -> None:
            if not isinstance(obj.get("note"), str) or not obj["note"].strip():
                raise ValueError("key 'note' must be a non-empty string")

        obj = self.client.chat_json(
            prompts.PROJECT_SYSTEM, prompts.project_user(city, n_months, sample_rows), _validate
        )
        return obj["note"].strip()


# ---------------------------------------------------------------------------
# Offline mock model
# ---------------------------------------------------------------------------


class MockModel:
    """Deterministic offline stand-in for the reasoning model.

    Prediction = Theil-Sen trend + detrended seasonal profile, then a genuine
    memory correction: the mean signed error of the retrieved past cases is
    subtracted, so as the memory store grows the mock really does get better —
    a numeric analog of the in-context learning the real model does in text.
    """

    def predict(self, ctx: dict[str, Any]) -> dict[str, float]:
        out: dict[str, float] = {}
        target = ctx["target_month"]
        target_cal = ctx["season"]
        for m in METRICS:
            pts = [(h["month"], h["cal_month"], h[m]) for h in ctx["history"] if h[m] is not None]
            est = trend_seasonal_estimate(pts, target, target_cal)
            if est is None:
                out[m] = float(np.clip(ctx["inputs"].get(f"{m}_latest", 0.0), *BOUNDS[m]))
                continue
            pred = est
            # memory correction: learn from own past signed errors, weighting
            # records by season proximity (errors are strongly seasonal) and
            # doubling down on same-zone history
            errs, wts = [], []
            for r in ctx.get("memories", []):
                e = r.get("signed_error", {}).get(m)
                if e is None:
                    continue
                r_season = r.get("inputs", {}).get("season", target_cal)
                w = 1.0 + np.cos(2 * np.pi * (r_season - target_cal) / 12.0)
                if r.get("zone") == ctx["zone"]:
                    w *= 1.5
                errs.append(e)
                wts.append(w)
            if errs and sum(wts) > 1e-9:
                pred -= 0.6 * float(np.average(errs, weights=wts))
            out[m] = float(np.clip(pred, *BOUNDS[m]))
        return out

    def lesson(self, ctx: dict[str, Any], pred: dict[str, float], actual: dict[str, float]) -> str:
        stats = ctx.get("stats", {})
        worst, worst_z = "aqi", -1.0
        for m in METRICS:
            std = stats.get(m, {}).get("std", 1.0) or 1.0
            z = abs(pred[m] - actual[m]) / std
            if z > worst_z:
                worst, worst_z = m, z
        direction = "Overestimated" if pred[worst] > actual[worst] else "Underestimated"
        return (
            f"{direction} {worst} in zone {ctx['zone']} for calendar month "
            f"{ctx['season']}; weight the seasonal cycle more against the raw trend."
        )

    def recommend(
        self, ranked_rows: list[dict[str, Any]], lessons: list[str]
    ) -> list[dict[str, Any]]:
        out = []
        for row in ranked_rows:
            need_tolerance = row.get("aqi_latest", 0) >= 120 or row.get("aqi_pred_delta", 0) > 0
            tight_space = row.get("plantable_space", 0.5) < 0.35
            dry = row.get("ndvi_slope", 0) < 0
            soil = row.get("soil")

            def score(sp: dict[str, str]) -> float:
                s = 0.0
                tol = {"high": 2.0, "medium": 1.0, "low": 0.0}[sp["pollution_tolerance"]]
                s += tol * (2.0 if need_tolerance else 1.0)
                if tight_space and sp["canopy"] in ("small", "medium"):
                    s += 1.5
                if not tight_space and sp["canopy"] == "large":
                    s += 1.0
                if dry and sp["water_need"] == "low":
                    s += 1.0
                if sp["native_status"] == "native":
                    s += 0.5
                return s

            # drop species whose soil pH/texture clashes with the zone; only
            # fall back to the full KB if the soil filter leaves too few
            compatible = [sp for sp in SPECIES_KB if species_soil_ok(sp, soil)]
            pool = compatible if len(compatible) >= 5 else SPECIES_KB
            ranked_sp = sorted(pool, key=score, reverse=True)
            species = [sp["common"] for sp in ranked_sp[:4]]
            soil_note = ""
            if isinstance(soil, dict) and soil.get("ph") is not None:
                soil_note = (
                    f" Soil pH {soil['ph']:.1f} ({soil.get('ph_class', '?')}), "
                    f"{soil.get('texture', 'unknown')} — species filtered for soil fit."
                )
            out.append(
                {
                    "zone": row["zone"],
                    # Traffic is deliberately absent from this sentence. The
                    # stream is an inert placeholder with MCDA weight 0 — it
                    # never moves the ranking — so quoting "traffic change
                    # +0.0" alongside real figures implied it was a measured
                    # input that happened to be flat. It is neither.
                    "justification": (
                        f"Priority score {row['score']:.3f}: predicted AQI change "
                        f"{row['aqi_pred_delta']:+.1f}, "
                        f"NDVI trend {row['ndvi_slope'] * 12:+.4f}/yr with canopy at "
                        f"{row['ndvi_latest']:.2f} and plantable-space {row['plantable_space']:.2f}. "
                        "Species chosen for pollution tolerance and fit to available space."
                        + soil_note
                    ),
                    "species": validate_selection(species),
                }
            )
        return out

    def project_note(self, city: str, n_months: int, sample_rows: list[dict[str, Any]]) -> str:
        return (
            f"UNVALIDATED projection: {n_months} months of damped trend extrapolation for "
            f"{city}. No observations exist that far out, so error compounds and is "
            "uncheckable; policy shifts, climate variation and land-use change are not "
            "modeled. Treat as a scenario sketch, not a forecast."
        )


# Back-compat: this class was OpenRouter-only before local inference existed.
OpenRouterModel = ReasoningModel


def build_model(cfg: ModelCfg, mock: bool, strict: bool = True) -> Any:
    """Construct the reasoning back end named by `cfg`.

    With strict=False a provider that cannot be brought up (OpenVINO runtime
    not installed, weights not fetched, hosted key missing) degrades to
    MockModel with a loud warning instead of raising. Long-running callers —
    the HTTP server above all — must never die at startup because an optional
    local model is absent; the offline engine answers every request correctly,
    just less fluently. `greenplan run` keeps strict=True so a CLI user who
    asked for a specific provider is told when they did not get it."""
    if mock or cfg.provider == "mock":
        log.info("using MockModel (offline, no API key, no cost)")
        return MockModel()
    try:
        if cfg.provider == "openvino":
            log.info(
                "using OpenVINO local model %s on %s (inference only, no network)",
                cfg.model_dir, cfg.device,
            )
            return ReasoningModel(cfg, OpenVINOClient(cfg))
        log.info("using %s model %s (inference only)", cfg.provider, cfg.name)
        return ReasoningModel(cfg)
    except Exception as exc:
        if strict:
            raise
        log.warning(
            "provider '%s' unavailable (%s) - falling back to the offline engine. "
            "Everything still works; reasoning is deterministic rather than "
            "generated.", cfg.provider, exc,
        )
        return MockModel()
