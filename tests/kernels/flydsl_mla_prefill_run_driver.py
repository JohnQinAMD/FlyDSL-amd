#!/usr/bin/env python3
"""KernelForge run-mode driver for the FlyDSL query-tiled MLA PREFILL kernel (gfx950, bf16).

Contract:
    python flydsl_mla_prefill_run_driver.py correctness [--shape seqlen=512] [--ref causal|noncausal]
    python flydsl_mla_prefill_run_driver.py performance [--shape ...] --bench-mode

Full-prefill workload: batch=1, kv_len == q_len == S, causal-append == lower-triangular.
Calls the STABLE interface kernels.attention.mla_prefill_qtiled.flydsl_mla_prefill (query-tiled,
writes out [S,128,512] directly — NO decode split-KV metadata, so no O(S²) buffer blowup).
Correctness oracle = torch SDPA is_causal on the SAME data cast to fp32 (q,k=576 / v=512, MQA 128:1),
memory-safe even at large S. Prints:
    correctness -> "SNR: <db> dB" + "allclose: <bool>" + "max_diff: <cos_diff>"
    performance -> "median_ms: <device p50>"

Do not edit this driver — it is the frozen scoring harness. The kernel under development is
kernels/attention/mla_prefill_qtiled*.py; it starts as a stub (raises) so correctness fails until
the fellow lands a real kernel.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_REPO)
sys.path.insert(0, _REPO)
torch.set_default_device("cuda")

from kernels.attention.mla_prefill_qtiled import flydsl_mla_prefill  # noqa: E402

QK, VD, NH, NKV, PAGE = 576, 512, 128, 1, 1
_CACHE_DIR = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR", "/root/.flydsl")
_KERNEL = os.environ.get("FORGE_FLYDSL_KERNEL",
                         os.path.join(_REPO, "kernels/attention/mla_prefill_qtiled.py"))
_STAMP = os.path.join(_CACHE_DIR, ".prefill_qtiled_mtime")


def _clear_cache_if_changed():
    try:
        kmt = os.path.getmtime(_KERNEL) if os.path.exists(_KERNEL) else 0.0
        prev = None
        if os.path.exists(_STAMP):
            try:
                prev = float(open(_STAMP).read().strip() or 0)
            except Exception:
                prev = None
        if prev is not None and abs(prev - kmt) < 1e-6:
            return
        if os.path.isdir(_CACHE_DIR):
            for n in os.listdir(_CACHE_DIR):
                if n == os.path.basename(_STAMP):
                    continue
                p = os.path.join(_CACHE_DIR, n)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        open(_STAMP, "w").write(repr(kmt))
    except Exception:
        pass


def _parse_shape(s):
    S = 512
    if s and s != "default":
        for part in s.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip() in ("seqlen", "S", "ctx_len", "ctx"):
                    S = int(v)
    return S


def _build(S):
    torch.manual_seed(1234 + S)
    num_page = S
    q = (torch.randn(S, NH, QK, dtype=torch.bfloat16) * 0.5)
    kv = (torch.randn(num_page, PAGE, NKV, QK, dtype=torch.bfloat16) * 0.5)
    kv_indices = torch.randperm(num_page, dtype=torch.int)
    qo_indptr = torch.tensor([0, S], dtype=torch.int)
    kv_indptr = torch.tensor([0, S], dtype=torch.int)
    out = torch.empty(S, NH, VD, dtype=torch.bfloat16).fill_(float("nan"))
    sm_scale = 1.0 / (QK ** 0.5)
    return q, kv, kv_indices, qo_indptr, kv_indptr, out, sm_scale, num_page


def _oracle(q, kv, kv_indices, num_page, sm_scale, S, causal, head_chunk=0):
    """Compute the fp32 SDPA reference without materializing all heads at long S."""
    kv_g = kv.to(torch.float32).view(num_page, QK)[kv_indices][:S]
    k1 = kv_g.unsqueeze(0).unsqueeze(0)
    v1 = k1[..., :VD]
    chunk = head_chunk or (4 if S >= 8192 else NH)
    ref = torch.empty((S, NH, VD), dtype=torch.float32)
    for hs in range(0, NH, chunk):
        he = min(hs + chunk, NH)
        hc = he - hs
        qf = q[:, hs:he, :].to(torch.float32).permute(1, 0, 2).unsqueeze(0)
        ref[:, hs:he, :] = (
            torch.nn.functional.scaled_dot_product_attention(
                qf,
                k1.expand(1, hc, S, QK),
                v1.expand(1, hc, S, VD),
                is_causal=causal,
                scale=sm_scale,
            )
            .squeeze(0)
            .permute(1, 0, 2)
        )
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default=None, choices=[None, "correctness", "performance"])
    ap.add_argument("--shape", default="default")
    ap.add_argument("--mode", default=None)
    ap.add_argument("--ref", default="causal", choices=["causal", "noncausal"])
    ap.add_argument("--bench-mode", dest="bench_mode", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument(
        "--head-chunk",
        type=int,
        default=0,
        help="SDPA oracle heads per chunk (0 selects 4 for S>=8192, else all)",
    )
    args = ap.parse_args()
    action = args.action or ("performance" if args.bench_mode else "correctness")
    S = _parse_shape(args.shape)
    causal = (args.ref == "causal")

    _clear_cache_if_changed()
    q, kv, kvi, qo, kvp, out, sm, npg = _build(S)

    def launch():
        flydsl_mla_prefill(q, kv, kvi, qo, kvp, out, sm, causal=causal)

    try:
        launch()
        torch.cuda.synchronize()
    except Exception as e:
        print("allclose: False")
        print(f"DRIVER_ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if action == "performance":
        for _ in range(args.warmup):
            launch()
        torch.cuda.synchronize()
        N = max(args.iters, 3)
        st = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(N)]
        for i in range(N):
            st[i].record()
            launch()
            en[i].record()
        torch.cuda.synchronize()
        us = sorted(a.elapsed_time(b) * 1000 for a, b in zip(st, en))[N // 2]
        print(f"median_ms: {us / 1000.0:.6f}")
        print(f"wall_ms: {us / 1000.0:.6f}")
        # Roofline inputs (opt-in): KernelForge's bench tool folds these into a
        # MAF-roofline steering directive. Causal prefill FLOPs = 0.5*S^2*NH*(QK+VD)*2;
        # analytic HBM lower bound = Q + KV(once) + O, bf16.
        flops = 0.5 * S * S * NH * (QK + VD) * 2
        hbm_bytes = (S * NH * QK + S * QK + S * NH * VD) * 2
        print(f"flops: {flops:.6e}")
        print(f"hbm_bytes: {hbm_bytes:.6e}")
        print("dtype: bf16")

    ref = _oracle(q, kv, kvi, npg, sm, S, causal, head_chunk=args.head_chunk)
    x, y = ref.double(), out.double()
    cos = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    noise = (x - y).norm().item()
    snr = 20.0 * math.log10(x.norm().item() / max(noise, 1e-20)) if noise > 0 else 999.0
    passed = cos < 3e-2
    print(f"SNR: {snr:.2f} dB")
    print(f"allclose: {passed}")
    print(f"max_diff: {cos:.3e}  (ref={args.ref}, S={S})")
    return 0 if (passed or action == "performance") else 1


if __name__ == "__main__":
    raise SystemExit(main())
