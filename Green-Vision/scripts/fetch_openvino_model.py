"""Fetch a pre-compressed OpenVINO model for local, offline reasoning.

Intel publishes instruct models already converted to OpenVINO IR and weight-
compressed to INT4, so nothing here needs PyTorch, a conversion step, or a GPU.
An INT4 1.5B model is roughly 1 GB on disk against ~3 GB at FP16, loads in a
couple of seconds, and answers in a few more on an ordinary laptop CPU — which
is the whole point: the engine runs with no API key, no per-token cost and no
data leaving the machine.

Usage:
    python scripts/fetch_openvino_model.py
    python scripts/fetch_openvino_model.py --model qwen1.5b --out models/openvino
    python scripts/fetch_openvino_model.py --list

Then in config/city.yaml:
    model:
      provider: openvino
      model_dir: models/openvino/qwen2.5-1.5b-instruct-int4-ov
      device: CPU          # or GPU / NPU / AUTO
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Pre-converted, INT4-compressed instruct models published by Intel's OpenVINO
# team. Sizes are approximate on-disk footprints after download.
CATALOGUE: dict[str, dict[str, str]] = {
    "qwen1.5b": {
        "repo": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",
        "size": "~1.0 GB",
        "template": "chatml",
        "note": "Default. Best size-to-JSON-reliability trade-off on CPU.",
    },
    "tinyllama": {
        "repo": "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov",
        "size": "~0.7 GB",
        "template": "chatml",
        "note": "Smallest. For very constrained hardware; weaker at strict JSON.",
    },
    "phi3.5": {
        "repo": "OpenVINO/Phi-3.5-mini-instruct-int4-ov",
        "size": "~2.2 GB",
        "template": "chatml",
        "note": "Strongest reasoning of the three; wants ~8 GB RAM.",
    },
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Download a pre-compressed OpenVINO model")
    ap.add_argument("--model", default="qwen1.5b", choices=sorted(CATALOGUE))
    ap.add_argument("--out", default="models/openvino", help="parent directory")
    ap.add_argument("--list", action="store_true", help="show the catalogue and exit")
    args = ap.parse_args()

    if args.list:
        print("Pre-compressed OpenVINO models:\n")
        for key, m in CATALOGUE.items():
            print(f"  {key:11s} {m['size']:9s} {m['repo']}")
            print(f"  {'':11s} {m['note']}\n")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "error: huggingface_hub is required.\n    pip install openvino-genai huggingface_hub",
            file=sys.stderr,
        )
        return 2

    entry = CATALOGUE[args.model]
    dest = Path(args.out) / entry["repo"].split("/")[-1].lower()

    if (dest / "openvino_model.xml").exists():
        print(f"already present: {dest}")
    else:
        print(f"downloading {entry['repo']} ({entry['size']}) -> {dest}")
        snapshot_download(
            entry["repo"],
            local_dir=str(dest),
            # Weights, graph and tokenizer only — skip PyTorch duplicates.
            allow_patterns=["*.xml", "*.bin", "*.json", "*.txt", "*.model"],
        )
        print(f"done: {dest}")

    print(
        "\nSet in config/city.yaml:\n"
        "    model:\n"
        "      provider: openvino\n"
        f"      model_dir: {dest.as_posix()}\n"
        f"      chat_template: {entry['template']}\n"
        "      device: CPU"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
