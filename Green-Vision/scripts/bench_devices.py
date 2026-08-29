"""Benchmark the local model on every OpenVINO device this machine has.

OpenVINO's whole pitch is that one IR runs unchanged on `CPU`, an integrated
`GPU`, or an `NPU` — `model.device` in config/city.yaml passes straight
through. This script proves it on the machine you are sitting at: it
enumerates the devices the runtime can actually see, loads the same model on
each, times a fixed engine-style prompt, and prints one honest table.

On a plain laptop you will see CPU only. On an Intel Core Ultra ("AI PC") the
same command also benchmarks GPU and NPU — no code changes, which is exactly
the point worth demonstrating.

Usage:
    python scripts/bench_devices.py                       # all visible devices
    python scripts/bench_devices.py --devices CPU NPU     # just these
    python scripts/bench_devices.py --model-dir models/openvino/qwen2.5-1.5b-instruct-int4-ov

The prompt asks for a strict-JSON zone forecast, like the engine does, so the
numbers are representative of a real `--recommend` run rather than a synthetic
"hello".
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

# A realistic engine-style task: strict JSON out, a few hundred tokens in.
BENCH_SYSTEM = (
    "You are the forecasting engine of GreenGrid, an urban-planning AI. "
    'Reply with STRICT JSON only: {"aqi": <number>, "ndvi": <number>, '
    '"rationale": "<one line>"}'
)
BENCH_USER = (
    "ZONE: 8742cc692ffffff\n"
    "TASK: predict month 45 (calendar month 10), 12 months after the last row.\n"
    "HISTORY (month,aqi,ndvi): 30,92.1,0.19 31,95.4,0.18 32,101.2,0.17 "
    "33,98.7,0.18 34,103.5,0.16 35,110.2,0.15\n"
    "ROBUST TRENDS (per month): aqi +1.512, ndvi -0.00310\n"
    "Return the strict JSON now."
)
CHATML = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def bench_device(genai, model_dir: str, device: str, max_new_tokens: int) -> dict:
    """Load the model on one device, run the bench prompt, return timings."""
    t0 = time.time()
    pipe = genai.LLMPipeline(model_dir, device)
    load_s = time.time() - t0

    gen = genai.GenerationConfig()
    gen.max_new_tokens = max_new_tokens
    # Mirror the engine's real settings (config/city.yaml): light sampling.
    # Pure greedy can lock this INT4 model into token repetition on terse
    # numeric prompts, which would benchmark a degenerate generation.
    gen.do_sample = True
    gen.temperature = 0.2
    gen.repetition_penalty = 1.1

    prompt = CHATML.format(system=BENCH_SYSTEM, user=BENCH_USER)
    t0 = time.time()
    # List-in, DecodedResults-out — the form that carries measured perf_metrics.
    result = pipe.generate([prompt], gen)
    reply_s = time.time() - t0

    try:
        text = result.texts[0]
    except Exception:
        text = str(result)
    # Did the reply contain parseable JSON? A "no" here is honest signal, not
    # failure: the engine's repair re-asks recover exactly this case, and a
    # 1.5B INT4 model will say no more often than a hosted 70B would.
    try:
        json.loads(text[text.find("{"): text.rfind("}") + 1])
        json_ok = "yes"
    except Exception:
        json_ok = "no"
    row = {"device": device, "load_s": load_s, "reply_s": reply_s,
           "ttft_ms": None, "tok_s": None, "json_ok": json_ok}
    # wall-clock above stands on its own if this build has no perf_metrics
    try:
        pm = result.perf_metrics
        row["ttft_ms"] = pm.get_ttft().mean
        row["tok_s"] = pm.get_throughput().mean
    except Exception:
        pass
    del pipe
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the model per OpenVINO device")
    ap.add_argument("--model-dir", default="models/openvino/qwen2.5-1.5b-instruct-int4-ov")
    ap.add_argument("--devices", nargs="*", default=None,
                    help="subset to test (default: every device OpenVINO reports)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    try:
        import openvino
        import openvino_genai
    except ImportError:
        print(
            "error: the OpenVINO runtime is required.\n"
            "    pip install openvino-genai huggingface_hub",
            file=sys.stderr,
        )
        return 2

    model_dir = Path(args.model_dir)
    if not (model_dir / "openvino_model.xml").exists():
        print(
            f"error: no OpenVINO model at {model_dir}. Fetch one with:\n"
            "    python scripts/fetch_openvino_model.py",
            file=sys.stderr,
        )
        return 2

    core = openvino.Core()
    visible = core.available_devices
    names = {}
    for d in visible:
        try:
            names[d] = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            names[d] = "?"

    print(f"host: {platform.machine()} / {platform.processor() or platform.system()}")
    print(f"model: {model_dir.name}")
    print("devices OpenVINO can see:")
    for d in visible:
        print(f"  {d:6s} {names[d]}")
    if not any("NPU" in d for d in visible):
        print(
            "  (no NPU visible — on an Intel Core Ultra 'AI PC' this same "
            "command benchmarks it with zero code changes)"
        )
    print()

    targets = args.devices if args.devices else visible
    rows = []
    for device in targets:
        print(f"benchmarking {device} ...", flush=True)
        try:
            rows.append(bench_device(openvino_genai, str(model_dir), device,
                                     args.max_new_tokens))
        except Exception as exc:
            print(f"  {device}: FAILED — {exc}")

    if not rows:
        print("no device produced a result", file=sys.stderr)
        return 1

    print(f"\n{'device':8s} {'load s':>7s} {'reply s':>8s} {'ttft ms':>8s} {'tok/s':>7s} {'json':>5s}")
    for r in rows:
        ttft = f"{r['ttft_ms']:.0f}" if r["ttft_ms"] is not None else "-"
        toks = f"{r['tok_s']:.1f}" if r["tok_s"] is not None else "-"
        print(f"{r['device']:8s} {r['load_s']:7.1f} {r['reply_s']:8.1f} {ttft:>8s} {toks:>7s} {r["json_ok"]:>5s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
