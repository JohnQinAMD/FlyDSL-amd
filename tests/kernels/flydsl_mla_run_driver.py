#!/usr/bin/env python3
"""KernelForge run/loop driver for the FlyDSL MLA decode kernel (gfx950).

The orchestrator's ``test`` / ``bench`` MCP tools call this as:
    python flydsl_mla_run_driver.py correctness  [--shape batch=32,ctx_len=8192] [--mode smoke]
    python flydsl_mla_run_driver.py performance  [--shape ...] --bench-mode [--warmup N --iters N]
and also tolerate the bare loop-style form (no subcommand + --mode / --bench-mode).

It emits the markers those tools parse, backed by the FROZEN reference harness
(``test_mla_decode.run_single`` — torch MLA golden + aiter metadata):
    correctness -> "SNR: <db> dB" + "allclose: <bool>" + "max_diff: <x>"   (exit 0/1)
    performance -> "median_ms: <x>"   (device p50 from torch.profiler CUDA events, NOT wall)

Cache hygiene: FlyDSL's build cache keys on kernel source+closure but IGNORES env
flags and seqlen, so a stale compiled kernel can survive an edit. We clear it every
call so the number always reflects the CURRENT kernel the fellow just edited.

Only the KERNEL under tests/../kernels/attention is meant to change; do not edit this
driver or test_mla_decode.py (they are the frozen scoring harness).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# Resolve + enter the FlyDSL repo root so test_mla_decode's relative imports
# ("kernels.*", "tests.*", sys.path.insert(1,".")) resolve regardless of the CWD
# the MCP tool spawns us from.
_HERE = os.path.dirname(os.path.abspath(__file__))          # tests/kernels
_REPO = os.path.dirname(os.path.dirname(_HERE))             # FlyDSL repo root
os.chdir(_REPO)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

_CACHE_DIR = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "/root/.flydsl")
# The kernel this driver scores (override for other kernels, e.g. prefill).
_KERNEL = os.environ.get(
    "FORGE_FLYDSL_KERNEL",
    os.path.join(_REPO, "kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py"),
)
_STAMP = os.path.join(_CACHE_DIR, ".driver_kernel_mtime")


def _clear_flydsl_cache() -> None:
    """Clear FlyDSL's env/seqlen-blind build cache — but ONLY when the kernel
    source changed since the last call. Clearing on EVERY call forces a full JIT
    recompile per correctness/perf invocation, which blows past the MCP test/bench
    tool timeouts (the fellow then sees empty results). mtime-gating keeps the
    cache-hazard fix (a stale kernel never survives an edit) while letting repeated
    calls on the SAME kernel reuse the compiled artifact."""
    try:
        kmt = os.path.getmtime(_KERNEL) if os.path.exists(_KERNEL) else 0.0
        prev = None
        if os.path.exists(_STAMP):
            try:
                prev = float(open(_STAMP).read().strip() or 0)
            except Exception:
                prev = None
        if prev is not None and abs(prev - kmt) < 1e-6:
            return  # kernel unchanged -> keep the JIT cache (no recompile)
        if os.path.isdir(_CACHE_DIR):
            for name in os.listdir(_CACHE_DIR):
                if name == os.path.basename(_STAMP):
                    continue
                p = os.path.join(_CACHE_DIR, name)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(_STAMP, "w") as f:
            f.write(repr(kmt))
    except Exception:
        pass


def _parse_shape(shape_str: str) -> dict:
    """'batch=32,ctx_len=8192,decode_qlen=1' -> {batch:32, ctx_len:8192, ...}."""
    out = {"batch": 32, "ctx_len": 8192, "decode_qlen": 1}
    if not shape_str or shape_str == "default":
        return out
    for part in shape_str.split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        # accept both the task's canonical keys and a couple of aliases
        key = {"seqlen": "ctx_len", "ctx": "ctx_len", "b": "batch"}.get(k, k)
        try:
            out[key] = int(v)
        except ValueError:
            out[key] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="FlyDSL MLA decode — KernelForge driver")
    ap.add_argument("action", nargs="?", default=None,
                    choices=[None, "correctness", "performance"],
                    help="correctness | performance (inferred from --bench-mode if omitted)")
    ap.add_argument("--shape", default="default")
    ap.add_argument("--mode", default=None, help="smoke|stability|determinism (all = correctness)")
    ap.add_argument("--bench-mode", dest="bench_mode", action="store_true")
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None)
    args = ap.parse_args()

    # Resolve the action: explicit subcommand wins, else --bench-mode => performance.
    action = args.action or ("performance" if args.bench_mode else "correctness")

    shape = _parse_shape(args.shape)
    batch, ctx_len, qlen = shape["batch"], shape["ctx_len"], shape.get("decode_qlen", 1)

    _clear_flydsl_cache()

    # Import AFTER chdir/path setup so the frozen harness resolves.
    from tests.kernels.test_mla_decode import run_single  # noqa: E402

    kw = {}
    if args.warmup is not None:
        kw["bench_warmup"] = args.warmup
    if args.iters is not None:
        kw["bench_iters"] = args.iters

    try:
        m = run_single(batch, ctx_len, decode_qlen=qlen, raise_on_mismatch=False, **kw)
    except Exception as e:  # a broken edit -> report as failure, don't crash the tool
        print(f"allclose: False")
        print(f"DRIVER_ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    cos_diff = m["cos_diff"]
    snr_db = m["snr_db"]
    us = m["us"]
    passed = cos_diff < 3e-2

    if action == "performance":
        # Device p50 (torch.profiler CUDA-event duration), reported as median_ms so
        # KernelForge's keep/revert gate runs on DEVICE time, not wall time.
        print(f"median_ms: {us / 1000.0:.6f}")
        print(f"wall_ms: {us / 1000.0:.6f}")
        # correctness still surfaced so a fast-but-wrong kernel is visible
        print(f"SNR: {snr_db:.2f} dB")
        print(f"allclose: {passed}")
        return 0

    # correctness
    print(f"SNR: {snr_db:.2f} dB")
    print(f"allclose: {passed}")
    print(f"max_diff: {cos_diff:.3e}")
    print(f"median_ms: {us / 1000.0:.6f}")  # harmless extra; lets one call serve both
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
