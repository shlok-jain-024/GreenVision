"""Compress a base instruct model to INT4 OpenVINO IR with NNCF — ourselves.

`fetch_openvino_model.py` downloads models Intel already compressed. This
script is the same operation done locally: take an original FP16 model from
Hugging Face and produce the INT4 OpenVINO IR with Intel's NNCF (Neural
Network Compression Framework), via `optimum-cli export openvino`. The output
drops into `model_dir` in config/city.yaml exactly like a fetched one.

Why bother when Intel hosts pre-compressed copies? Two honest reasons:
  * any model — including one not in Intel's catalogue — can be compressed
    and run locally the same way;
  * the ~3-4x size cut (FP16 -> INT4) is the step that makes a language model
    fit on an ordinary municipal laptop, and doing it here makes that step
    reproducible instead of taken on faith.

NNCF weight compression is plain Python on the CPU: no GPU, no Intel hardware
required to RUN the compression (the result runs on any OpenVINO device).

Requires (one-time, ~2 GB of wheels — deliberately not in requirements.txt):
    pip install "optimum[openvino]"

Usage:
    python scripts/compress_model_nncf.py                        # Qwen2.5-1.5B -> INT4
    python scripts/compress_model_nncf.py --base Qwen/Qwen2.5-0.5B-Instruct
    python scripts/compress_model_nncf.py --weight-format int8   # milder, larger
    python scripts/compress_model_nncf.py --verify               # load + one reply
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def find_optimum_cli() -> str | None:
    """Prefer the optimum-cli that belongs to THIS interpreter's environment
    (venv scripts are not on PATH when the venv python is invoked directly)."""
    local = Path(sys.executable).parent / "optimum-cli"
    if local.exists():
        return str(local)
    return shutil.which("optimum-cli")


def main() -> int:
    ap = argparse.ArgumentParser(description="NNCF-compress a model to OpenVINO IR")
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Hugging Face id of the original (uncompressed) model")
    ap.add_argument("--weight-format", default="int4", choices=["int4", "int8"])
    ap.add_argument("--out", default="models/openvino", help="parent directory")
    ap.add_argument("--verify", action="store_true",
                    help="after export, load the result and generate one reply")
    args = ap.parse_args()

    optimum_cli = find_optimum_cli()
    if optimum_cli is None:
        print(
            "error: optimum-cli not found. Install the export stack with:\n"
            '    pip install "optimum[openvino]"',
            file=sys.stderr,
        )
        return 2

    name = args.base.split("/")[-1].lower()
    dest = Path(args.out) / f"{name}-{args.weight_format}-nncf-ov"

    if (dest / "openvino_model.xml").exists():
        print(f"already present: {dest} ({dir_size_mb(dest):.0f} MB)")
    else:
        print(f"exporting {args.base} -> {dest} ({args.weight_format}, NNCF)")
        print("downloads the FP16 original once, then compresses on CPU; "
              "expect several minutes.\n")
        t0 = time.time()
        # optimum-cli drives: HF download -> OpenVINO IR conversion -> NNCF
        # weight compression. --sym int4 matches Intel's published -int4-ov repos.
        cmd = [
            optimum_cli, "export", "openvino",
            "--model", args.base,
            "--weight-format", args.weight_format,
            "--sym",
            str(dest),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"\nexport failed (exit {proc.returncode}). Command was:\n  "
                  + " ".join(cmd), file=sys.stderr)
            return proc.returncode
        print(f"\ncompressed in {time.time() - t0:.0f}s -> {dest} "
              f"({dir_size_mb(dest):.0f} MB on disk)")

    if args.verify:
        try:
            import openvino_genai
        except ImportError:
            print("verify skipped: pip install openvino-genai", file=sys.stderr)
            return 2
        print("loading compressed model on CPU ...")
        t0 = time.time()
        pipe = openvino_genai.LLMPipeline(str(dest), "CPU")
        gen = openvino_genai.GenerationConfig()
        gen.max_new_tokens = 32
        gen.do_sample = False
        reply = pipe.generate(
            "<|im_start|>user\nReply with exactly: OK\n<|im_end|>\n"
            "<|im_start|>assistant\n",
            gen,
        )
        print(f"loaded + replied in {time.time() - t0:.1f}s: {str(reply)[:60]!r}")

    print(
        "\nSet in config/city.yaml:\n"
        "    model:\n"
        "      provider: openvino\n"
        f"      model_dir: {dest.as_posix()}\n"
        "      chat_template: chatml\n"
        "      device: CPU"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
