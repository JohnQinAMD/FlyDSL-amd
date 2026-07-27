# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 sparse-MLA training correctness and historical-SOTA benchmark on gfx950.

The device tests are opt-in because the production matrix compiles several
large kernels. Correctness uses an independent chunked Torch implementation.
The benchmark records current PR-head latency and compares it with the fastest
published FlyDSL row from AMD-AGI/Primus PR #873. That published row is rounded
to 0.01 ms and comes from a different software snapshot, so the ratio is
historical regression context rather than a same-host paired comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

ROOT = Path(__file__).resolve().parents[2]
DIM = 512
ROPE_DIM = 64
DQK = DIM + ROPE_DIM
WINDOW = 128
SCALE = 1.0 / math.sqrt(DIM)
VARIANTS = {
    "flash": {"heads": 64, "index_topk": 512},
    "pro": {"heads": 128, "index_topk": 1024},
}

# AMD-AGI/Primus PR #873, merge e1a541da2, MI355X, B=1/S=4096.
# Fastest published `_turbo_flydsl` cells, rounded to 0.01 ms.
PR873_FASTEST_FLYDSL_MS = {
    ("flash", 0): {"fwd": 0.20, "bwd": 0.67},
    ("flash", 4): {"fwd": 0.53, "bwd": 2.55},
    ("flash", 128): {"fwd": 0.22, "bwd": 0.78},
    ("pro", 0): {"fwd": 0.38, "bwd": 1.29},
    ("pro", 4): {"fwd": 1.41, "bwd": 6.32},
    ("pro", 128): {"fwd": 0.43, "bwd": 1.49},
}


@dataclass(frozen=True)
class Shape:
    variant: str
    compression_ratio: int
    sequence: int

    @property
    def label(self) -> str:
        return f"{self.variant}_cr{self.compression_ratio}_s{self.sequence}"


@dataclass
class Runtime:
    torch: Any
    fwd_module: Any
    bwd_module: Any
    fwd: Callable[..., Any]
    bwd: Callable[..., Any]


@dataclass
class Inputs:
    q: Any
    kv: Any
    topk: Any
    sink: Any
    grad_output: Any


SMOKE_MATRIX = (
    Shape("flash", 0, 512),
    Shape("flash", 4, 512),
    Shape("pro", 128, 512),
)
RELEASE_MATRIX = tuple(
    Shape(variant, compression_ratio, 4096) for variant in ("flash", "pro") for compression_ratio in (0, 4, 128)
)


def _require_opt_in(name: str) -> None:
    if os.environ.get(name) != "1":
        pytest.skip(f"set {name}=1 to run the gfx950 DSV4 training test")


def _load_runtime() -> Runtime:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"DSV4 sparse-MLA training requires gfx950, got {arch}")

    fwd_module = importlib.import_module("kernels.attention.mla_v4_sparse_train_fwd")
    bwd_module = importlib.import_module("kernels.attention.mla_v4_sparse_train_bwd")
    return Runtime(
        torch=torch,
        fwd_module=fwd_module,
        bwd_module=bwd_module,
        fwd=fwd_module.sparse_mla_fwd_flydsl,
        bwd=bwd_module.sparse_mla_bwd_flydsl,
    )


def _shape_config(shape: Shape) -> tuple[int, int, int]:
    heads = VARIANTS[shape.variant]["heads"]
    if shape.compression_ratio == 0:
        return heads, 0, 0
    pool_rows = max(shape.sequence // shape.compression_ratio, 1)
    selected_rows = min(VARIANTS[shape.variant]["index_topk"], pool_rows) if shape.compression_ratio == 4 else 0
    return heads, pool_rows, selected_rows


def _build_inputs(runtime: Runtime, shape: Shape, seed: int) -> Inputs:
    torch = runtime.torch
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    heads, pool_rows, selected_rows = _shape_config(shape)

    latent = torch.randn(
        (shape.sequence, DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    q = torch.randn(
        (shape.sequence, heads, DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    q = torch.cat(
        (
            q,
            torch.zeros(
                (shape.sequence, heads, ROPE_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
        ),
        dim=-1,
    ).contiguous()
    sink = (
        torch.randn(
            (heads,),
            generator=generator,
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    ).contiguous()
    grad_output = torch.randn(
        (shape.sequence, heads, DIM),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )

    token = torch.arange(shape.sequence, device=device).view(shape.sequence, 1)
    window = token - WINDOW + 1 + torch.arange(WINDOW, device=device).view(1, WINDOW)
    window = torch.where(window >= 0, window, torch.full_like(window, -1))
    if shape.compression_ratio == 0:
        kv = latent.unsqueeze(1)
        topk = window
    else:
        pool = torch.randn(
            (pool_rows, DIM),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        )
        kv = torch.cat((latent, pool), dim=0).unsqueeze(1)
        if shape.compression_ratio == 4:
            pool_topk = shape.sequence + torch.randint(
                0,
                pool_rows,
                (shape.sequence, selected_rows),
                generator=generator,
                device=device,
            )
        else:
            pool_slot = torch.arange(pool_rows, device=device).view(1, pool_rows)
            visible = ((pool_slot + 1) * shape.compression_ratio - 1) <= token
            pool_topk = torch.where(
                visible,
                shape.sequence + pool_slot,
                torch.full_like(pool_slot.expand(shape.sequence, pool_rows), -1),
            )
        topk = torch.cat((window, pool_topk), dim=1)

    padding = (-topk.shape[1]) % 64
    if padding:
        topk = torch.cat(
            (
                topk,
                torch.full(
                    (shape.sequence, padding),
                    -1,
                    dtype=topk.dtype,
                    device=device,
                ),
            ),
            dim=1,
        )
    kv = torch.cat(
        (
            kv,
            torch.zeros(
                (kv.shape[0], 1, ROPE_DIM),
                dtype=torch.bfloat16,
                device=device,
            ),
        ),
        dim=-1,
    ).contiguous()
    return Inputs(
        q=q,
        kv=kv,
        topk=topk.to(torch.int32).contiguous(),
        sink=sink,
        grad_output=grad_output,
    )


def _torch_reference(runtime: Runtime, inputs: Inputs) -> tuple[Any, ...]:
    """Explicit sparse-softmax forward/backward, chunked over query tokens."""
    torch = runtime.torch
    chunk_size = int(os.environ.get("FLYDSL_V4_TRAIN_REFERENCE_CHUNK", "16"))
    if chunk_size <= 0:
        raise ValueError("FLYDSL_V4_TRAIN_REFERENCE_CHUNK must be positive")

    q = inputs.q[..., :DIM].float()
    kv = inputs.kv[:, 0, :DIM].float()
    grad_output = inputs.grad_output.float()
    tokens, heads, _ = q.shape
    out = torch.empty((tokens, heads, DIM), dtype=torch.float32, device=q.device)
    lse = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    dq = torch.zeros_like(inputs.q, dtype=torch.float32)
    dkv = torch.zeros_like(inputs.kv, dtype=torch.float32)
    dsink = torch.zeros_like(inputs.sink)

    for start in range(0, tokens, chunk_size):
        end = min(start + chunk_size, tokens)
        indices = inputs.topk[start:end]
        valid = indices >= 0
        safe = indices.clamp_min(0).long()
        keys = kv.index_select(0, safe.reshape(-1)).reshape(end - start, indices.shape[1], DIM)
        scores = torch.einsum("chd,crd->chr", q[start:end], keys) * SCALE
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        local_lse = torch.logaddexp(
            torch.logsumexp(scores, dim=-1),
            inputs.sink.view(1, heads),
        )
        probabilities = torch.exp(scores - local_lse.unsqueeze(-1))
        probabilities = torch.where(valid[:, None, :], probabilities, 0.0)
        local_out = torch.einsum("chr,crd->chd", probabilities, keys)
        delta = (local_out * grad_output[start:end]).sum(dim=-1)
        dscores = probabilities * (torch.einsum("chd,crd->chr", grad_output[start:end], keys) - delta.unsqueeze(-1))

        out[start:end] = local_out
        lse[start:end] = local_lse
        dq[start:end, :, :DIM] = torch.einsum("chr,crd->chd", dscores, keys) * SCALE
        contributions = torch.einsum(
            "chr,chd->crd",
            probabilities,
            grad_output[start:end],
        )
        contributions += torch.einsum("chr,chd->crd", dscores, q[start:end]) * SCALE
        flat_valid = valid.reshape(-1)
        dkv[:, 0, :DIM].index_add_(
            0,
            safe.reshape(-1)[flat_valid],
            contributions.reshape(-1, DIM)[flat_valid],
        )
        sink_probability = torch.exp(inputs.sink.view(1, heads) - local_lse)
        dsink -= (sink_probability * delta).sum(dim=0)

    return out, lse, dq, dkv, dsink


def _snr_db(torch: Any, reference: Any, candidate: Any) -> float:
    ref = reference.float()
    out = candidate.float()
    if not bool(torch.isfinite(out).all().item()):
        return float("-inf")
    signal = ref.square().sum()
    noise = (ref - out).square().sum()
    if float(noise.item()) == 0.0:
        return 999.0
    return float((10.0 * torch.log10(signal / noise)).item())


def _validate_case(
    runtime: Runtime,
    inputs: Inputs,
    *,
    canonical_topk: bool,
) -> tuple[tuple[Any, Any], tuple[Any, ...], dict[str, Any]]:
    torch = runtime.torch
    output, lse = runtime.fwd(
        inputs.q,
        inputs.kv,
        inputs.topk,
        attn_sink=inputs.sink,
        scale=SCALE,
        canonical_topk=canonical_topk,
    )
    gradients = runtime.bwd(
        inputs.q,
        inputs.kv,
        output,
        inputs.grad_output,
        inputs.topk,
        lse,
        attn_sink=inputs.sink,
        scale=SCALE,
        canonical_topk=canonical_topk,
    )
    torch.cuda.synchronize()
    ref_output, ref_lse, ref_dq, ref_dkv, ref_dsink = _torch_reference(runtime, inputs)
    forward_snr = _snr_db(torch, ref_output, output)
    lse_snr = _snr_db(torch, ref_lse, lse)
    backward_snr = {
        "dq": _snr_db(torch, ref_dq, gradients[0]),
        "dkv": _snr_db(torch, ref_dkv, gradients[1]),
        "dsink": _snr_db(torch, ref_dsink, gradients[2]),
    }
    assert forward_snr >= 40.0, forward_snr
    assert lse_snr >= 40.0, lse_snr
    assert min(backward_snr.values()) >= 35.0, backward_snr
    assert bool(torch.count_nonzero(gradients[0][..., DIM:]).item()) is False
    assert bool(torch.count_nonzero(gradients[1][..., DIM:]).item()) is False

    output2, lse2 = runtime.fwd(
        inputs.q,
        inputs.kv,
        inputs.topk,
        attn_sink=inputs.sink,
        scale=SCALE,
        canonical_topk=canonical_topk,
    )
    gradients2 = runtime.bwd(
        inputs.q,
        inputs.kv,
        output2,
        inputs.grad_output,
        inputs.topk,
        lse2,
        attn_sink=inputs.sink,
        scale=SCALE,
        canonical_topk=canonical_topk,
    )
    torch.cuda.synchronize()
    deterministic = (
        torch.equal(output, output2)
        and torch.equal(lse, lse2)
        and all(torch.equal(first, second) for first, second in zip(gradients, gradients2, strict=True))
    )
    assert deterministic
    metrics = {
        "forward_snr_db": forward_snr,
        "lse_snr_db": lse_snr,
        "backward_snr_db": backward_snr,
        "deterministic": deterministic,
        "dq_rope_zero": True,
        "dkv_rope_zero": True,
    }
    return (output, lse), gradients, metrics


def _time_ms(
    torch: Any,
    operation: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    repeat_medians = []
    for _ in range(repeats):
        events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(iterations)
        ]
        for start, end in events:
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize()
        repeat_medians.append(statistics.median(float(start.elapsed_time(end)) for start, end in events))
    return statistics.median(repeat_medians), repeat_medians


def _run_shape(runtime: Runtime, shape: Shape, *, benchmark: bool, case_index: int) -> dict[str, Any]:
    inputs = _build_inputs(runtime, shape, seed=20260725 + case_index * 1009)
    (output, lse), _, correctness = _validate_case(
        runtime,
        inputs,
        canonical_topk=True,
    )
    result: dict[str, Any] = {
        "shape": shape.label,
        "variant": shape.variant,
        "compression_ratio": shape.compression_ratio,
        "sequence": shape.sequence,
        "heads": VARIANTS[shape.variant]["heads"],
        "padded_rank": inputs.topk.shape[1],
        "correctness": correctness,
    }
    if not benchmark:
        print("V4_TRAIN_CORRECTNESS " + json.dumps(result, sort_keys=True), flush=True)
        return result

    warmup = int(os.environ.get("FLYDSL_V4_TRAIN_BENCH_WARMUP", "8"))
    iterations = int(os.environ.get("FLYDSL_V4_TRAIN_BENCH_ITERS", "21"))
    repeats = int(os.environ.get("FLYDSL_V4_TRAIN_BENCH_REPEATS", "3"))
    if min(warmup, iterations, repeats) <= 0:
        raise ValueError("benchmark warmup, iterations, and repeats must be positive")

    def run_fwd() -> Any:
        return runtime.fwd(
            inputs.q,
            inputs.kv,
            inputs.topk,
            attn_sink=inputs.sink,
            scale=SCALE,
            canonical_topk=True,
        )

    def run_bwd() -> Any:
        return runtime.bwd(
            inputs.q,
            inputs.kv,
            output,
            inputs.grad_output,
            inputs.topk,
            lse,
            attn_sink=inputs.sink,
            scale=SCALE,
            canonical_topk=True,
        )

    fwd_ms, fwd_repeats = _time_ms(
        runtime.torch,
        run_fwd,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    bwd_ms, bwd_repeats = _time_ms(
        runtime.torch,
        run_bwd,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
    )
    historical = PR873_FASTEST_FLYDSL_MS[(shape.variant, shape.compression_ratio)]
    result.update(
        {
            "flydsl_fwd_ms": fwd_ms,
            "flydsl_bwd_ms": bwd_ms,
            "flydsl_fwd_repeat_ms": fwd_repeats,
            "flydsl_bwd_repeat_ms": bwd_repeats,
            "pr873_fastest_published_flydsl_ms": historical,
            "historical_pr873_speedup": {
                "fwd": historical["fwd"] / fwd_ms,
                "bwd": historical["bwd"] / bwd_ms,
            },
        }
    )
    print("V4_TRAIN_BENCH " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _run_generic_topk_guard(runtime: Runtime) -> dict[str, Any]:
    shape = Shape("flash", 0, 512)
    inputs = _build_inputs(runtime, shape, seed=20262727)
    valid = inputs.topk >= 0
    shifted = (inputs.topk + 1) % shape.sequence
    inputs.topk = runtime.torch.where(valid, shifted, inputs.topk).contiguous()
    _, _, correctness = _validate_case(
        runtime,
        inputs,
        canonical_topk=False,
    )
    result = {
        "shape": "flash_custom_r128_s512",
        "canonical_topk": False,
        "correctness": correctness,
    }
    print("V4_TRAIN_GENERIC_TOPK " + json.dumps(result, sort_keys=True), flush=True)
    return result


def _driver(mode: str) -> None:
    runtime = _load_runtime()
    matrix = SMOKE_MATRIX if mode == "correctness" else RELEASE_MATRIX
    results = [
        _run_shape(runtime, shape, benchmark=mode == "benchmark", case_index=index)
        for index, shape in enumerate(matrix)
    ]
    if mode == "correctness":
        results.append(_run_generic_topk_guard(runtime))
    summary: dict[str, Any] = {
        "mode": mode,
        "cases": len(results),
        "gpu": runtime.torch.cuda.get_device_name(),
        "arch": runtime.torch.cuda.get_device_properties(0).gcnArchName,
        "torch": runtime.torch.__version__,
        "hip": runtime.torch.version.hip,
        "flydsl": importlib.import_module("flydsl").__version__,
        "fwd_source_sha256": hashlib.sha256(Path(runtime.fwd_module.__file__).read_bytes()).hexdigest(),
        "bwd_source_sha256": hashlib.sha256(Path(runtime.bwd_module.__file__).read_bytes()).hexdigest(),
    }
    if mode == "benchmark":
        speedups = [result["historical_pr873_speedup"][phase] for result in results for phase in ("fwd", "bwd")]
        summary.update(
            {
                "historical_pr873_all_cell_geomean": statistics.geometric_mean(speedups),
                "historical_pr873_weakest_cell": min(speedups),
                "comparison": "rounded cross-snapshot context; not paired roofline evidence",
            }
        )
        if os.environ.get("FLYDSL_V4_TRAIN_REQUIRE_PR873_WIN") == "1":
            assert min(speedups) > 1.0, summary
    print("V4_TRAIN_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


@pytest.mark.large_shape
def test_mla_v4_sparse_train_correctness() -> None:
    _require_opt_in("FLYDSL_RUN_V4_TRAIN_TESTS")
    _driver("correctness")


@pytest.mark.large_shape
@pytest.mark.benchmark
def test_mla_v4_sparse_train_release_benchmark() -> None:
    _require_opt_in("FLYDSL_BENCH")
    _driver("benchmark")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("correctness", "benchmark"), required=True)
    args = parser.parse_args()
    _driver(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
