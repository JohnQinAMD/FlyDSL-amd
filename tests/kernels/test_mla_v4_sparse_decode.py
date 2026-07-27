# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse decode correctness and AITER ASM comparison on gfx950."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.kernels.benchmark_common import maybe_enable_aiter

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

BATCH = 256
ROW_LEN = 1152
NUM_HEADS = 16
NUM_KV_HEADS = 1
DIM_NOPE = 448
DIM_ROPE = 64
DIM_QK = DIM_NOPE + DIM_ROPE
DIM_PACKED = 512
V_HEAD_DIM = 512
NUM_SPLITS = 1
SOFTMAX_SCALE = 1.0 / math.sqrt(DIM_QK)
SCALE_TILES = 7
SCALE_TILE_SIZE = 64
SCALE_BYTES = SCALE_TILES * 2
AITER_ASM_SYMBOL = "mla_a8w8_qh64_qseqlen1_gqaratio64_nm"
AITER_ASM_CODE_OBJECT = f"{AITER_ASM_SYMBOL}.co"
ROOT = Path(__file__).resolve().parents[2]


@dataclass
class V4Case:
    q_packed: Any
    q_rope: Any
    kv_packed: Any
    kv_rope: Any
    qo_indptr: Any
    kv_indptr: Any
    kv_indices: Any
    kv_last_page_lens: Any
    sink: Any
    split_indptr: Any
    fly_output: Any
    fly_logits: Any
    fly_lse: Any
    aiter_output: Any
    aiter_logits: Any
    aiter_lse: Any


@dataclass(frozen=True)
class V4Runtime:
    torch: Any
    aiter: Any | None
    flydsl_module: Any
    flydsl_launcher: Callable[..., Any]


def _require_opt_in(name: str) -> None:
    if os.environ.get(name) != "1":
        pytest.skip(f"set {name}=1 to run the exact gfx950 V4 sparse-decode test")


def _load_runtime(*, require_aiter: bool) -> V4Runtime:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("a ROCm GPU is required")

    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"the V4 sparse-decode candidate requires gfx950, got {arch}")
    if torch.float8_e4m3fn is None:
        raise RuntimeError("gfx950 OCP FP8 support is unavailable")

    module = importlib.import_module("kernels.attention.mla_v4_sparse_decode")
    identity = module.FORGE_REGISTRATION_IDENTITY
    assert identity["mode"] == "selected"
    assert identity["variant"] == "m16s536"
    assert identity["registered_heavy_count"] == 1

    aiter = None
    if require_aiter:
        if not maybe_enable_aiter():
            raise RuntimeError("AITER is not importable; set AITER_REPO to an AITER checkout")
        import aiter as imported_aiter
        import aiter.mla  # noqa: F401

        if not callable(getattr(imported_aiter.mla, "mla_decode_fwd_v4_nm", None)):
            raise RuntimeError("AITER does not expose mla_decode_fwd_v4_nm")
        if not callable(getattr(imported_aiter, "mla_decode_v4_asm", None)):
            raise RuntimeError("AITER does not expose the underlying mla_decode_v4_asm kernel")
        if imported_aiter.dtypes.fp8 != torch.float8_e4m3fn:
            raise RuntimeError(f"expected OCP FP8 on gfx950, got AITER dtype {imported_aiter.dtypes.fp8}")
        aiter = imported_aiter

    return V4Runtime(
        torch=torch,
        aiter=aiter,
        flydsl_module=module,
        flydsl_launcher=module.launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic,
    )


def _pack_v4_2buff(torch: Any, source: Any) -> tuple[Any, Any]:
    """Pack BF16 as [448 FP8 | 14 duplicated E8M0 | 50 pad] + BF16 RoPE64.

    This mirrors the MIT-licensed AITER V4 ASM test contract.
    """
    assert source.dtype == torch.bfloat16
    assert source.shape[-1] == DIM_QK

    leading = source.shape[:-1]
    nope = source[..., :DIM_NOPE].float().reshape(*leading, SCALE_TILES, SCALE_TILE_SIZE)
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    scale = torch.pow(
        2.0,
        torch.clamp_min(nope.abs().amax(dim=-1) / fp8_max, 1.0e-4).log2().ceil(),
    )
    quantized = (nope / scale.unsqueeze(-1)).to(torch.float8_e4m3fn).reshape(*leading, DIM_NOPE)

    packed_u8 = torch.zeros((*leading, DIM_PACKED), dtype=torch.uint8, device=source.device)
    packed = packed_u8.view(torch.float8_e4m3fn)
    packed[..., :DIM_NOPE].copy_(quantized)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp_(0, 254).to(torch.uint8)
    packed_u8[..., DIM_NOPE : DIM_NOPE + SCALE_BYTES].copy_(e8m0.repeat_interleave(2, dim=-1))
    rope = source[..., DIM_NOPE:].contiguous()
    return packed, rope


def _unpack_v4_2buff(torch: Any, packed: Any, rope: Any) -> Any:
    packed_u8 = packed.view(torch.uint8)
    leading = packed.shape[:-1]
    nope = packed[..., :DIM_NOPE].float().reshape(*leading, SCALE_TILES, SCALE_TILE_SIZE)
    e8m0 = packed_u8[..., DIM_NOPE : DIM_NOPE + SCALE_BYTES][..., 0::2].to(torch.int32)
    scale = torch.pow(2.0, (e8m0 - 127).float())
    nope = (nope * scale.unsqueeze(-1)).reshape(*leading, DIM_NOPE).to(torch.bfloat16)
    return torch.cat((nope, rope.to(torch.bfloat16)), dim=-1)


def _make_case(runtime: V4Runtime, seed: int) -> V4Case:
    torch = runtime.torch
    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    total_kv = BATCH * ROW_LEN
    q_native = torch.randn((BATCH, NUM_HEADS, DIM_QK), dtype=torch.bfloat16, device=device) * 0.5
    kv_native = (
        torch.randn(
            (total_kv, 1, NUM_KV_HEADS, DIM_QK),
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.5
    )
    q_packed, q_rope = _pack_v4_2buff(torch, q_native)
    kv_packed, kv_rope = _pack_v4_2buff(torch, kv_native)
    del q_native, kv_native

    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device) * ROW_LEN
    kv_indices = torch.randperm(total_kv, dtype=torch.int64, device=device).to(torch.int32)
    kv_last_page_lens = torch.ones(BATCH, dtype=torch.int32, device=device)
    split_indptr = torch.arange(BATCH + 1, dtype=torch.int32, device=device)
    sink = (torch.randn(NUM_HEADS, dtype=torch.float32, device=device) * 3.0).contiguous()
    assert torch.equal(qo_indptr, torch.arange(BATCH + 1, dtype=torch.int32, device=device))
    assert bool(torch.all(kv_indptr[1:] - kv_indptr[:-1] == ROW_LEN).item())
    assert kv_indices.numel() == total_kv
    assert int(kv_indices.min().item()) == 0
    assert int(kv_indices.max().item()) == total_kv - 1
    assert torch.equal(split_indptr, torch.arange(BATCH + 1, dtype=torch.int32, device=device))

    output_shape = (BATCH, NUM_HEADS, V_HEAD_DIM)
    logits_shape = (BATCH, NUM_SPLITS, NUM_HEADS, V_HEAD_DIM)
    lse_shape = (BATCH, NUM_SPLITS, NUM_HEADS, 1)
    return V4Case(
        q_packed=q_packed.contiguous(),
        q_rope=q_rope.contiguous(),
        kv_packed=kv_packed.contiguous(),
        kv_rope=kv_rope.contiguous(),
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices.contiguous(),
        kv_last_page_lens=kv_last_page_lens,
        sink=sink,
        split_indptr=split_indptr,
        fly_output=torch.empty(output_shape, dtype=torch.bfloat16, device=device),
        fly_logits=torch.empty(logits_shape, dtype=torch.float32, device=device),
        fly_lse=torch.empty(lse_shape, dtype=torch.float32, device=device),
        aiter_output=torch.empty(output_shape, dtype=torch.bfloat16, device=device),
        aiter_logits=torch.empty(logits_shape, dtype=torch.float32, device=device),
        aiter_lse=torch.empty(lse_shape, dtype=torch.float32, device=device),
    )


def _launch_flydsl(runtime: V4Runtime, case: V4Case, *, output: Any | None = None) -> None:
    runtime.flydsl_launcher(
        case.q_packed,
        case.q_rope,
        case.kv_packed,
        case.kv_rope,
        case.qo_indptr,
        case.kv_indptr,
        case.kv_indices,
        case.sink,
        case.split_indptr,
        case.fly_output if output is None else output,
        case.fly_logits,
        case.fly_lse,
        NUM_SPLITS,
        float(SOFTMAX_SCALE),
        num_sequences=BATCH,
        stream=runtime.torch.cuda.current_stream(case.q_packed.device),
    )


def _launch_aiter_asm(runtime: V4Runtime, case: V4Case, *, output: Any | None = None) -> None:
    assert runtime.aiter is not None
    runtime.aiter.mla.mla_decode_fwd_v4_nm(
        q=case.q_packed,
        qrope=case.q_rope,
        kv_buffer=case.kv_packed,
        kvrope=case.kv_rope,
        output=case.aiter_output if output is None else output,
        qo_indptr=case.qo_indptr,
        kv_indptr=case.kv_indptr,
        kv_page_indices=case.kv_indices,
        kv_last_page_lens=case.kv_last_page_lens,
        max_seqlen_q=1,
        sink=case.sink,
        split_indptr=case.split_indptr,
        sm_scale=SOFTMAX_SCALE,
        out_16_nosplit=0,
        num_kv_splits=NUM_SPLITS,
        logits=case.aiter_logits,
        attn_lse=case.aiter_lse,
    )


def _torch_reference(runtime: V4Runtime, case: V4Case, *, chunk_rows: int = 8) -> Any:
    """Independent sink-aware reference over the exact packed input bytes."""
    torch = runtime.torch
    q = _unpack_v4_2buff(torch, case.q_packed, case.q_rope).float()
    kv = _unpack_v4_2buff(torch, case.kv_packed, case.kv_rope).reshape(-1, DIM_QK)
    indices = case.kv_indices.reshape(BATCH, ROW_LEN)
    output = torch.empty((BATCH, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=q.device)

    for start in range(0, BATCH, chunk_rows):
        end = min(start + chunk_rows, BATCH)
        logical_kv = kv.index_select(0, indices[start:end].reshape(-1).long())
        logical_kv = logical_kv.reshape(end - start, ROW_LEN, DIM_QK).float()
        scores = torch.einsum("bhd,bld->bhl", q[start:end], logical_kv) * SOFTMAX_SCALE
        maximum = torch.maximum(scores.amax(dim=-1), case.sink.view(1, NUM_HEADS))
        weights = torch.exp(scores - maximum.unsqueeze(-1))
        denominator = weights.sum(dim=-1) + torch.exp(case.sink.view(1, NUM_HEADS) - maximum)
        values = torch.einsum("bhl,bld->bhd", weights, logical_kv[..., :V_HEAD_DIM])
        output[start:end] = (values / denominator.unsqueeze(-1)).to(torch.bfloat16)
    return output


def _quality_metrics(torch: Any, reference: Any, candidate: Any) -> dict[str, Any]:
    ref = reference.float()
    out = candidate.float()
    diff = ref - out
    ref_rows = ref.reshape(-1, V_HEAD_DIM)
    out_rows = out.reshape(-1, V_HEAD_DIM)
    diff_rows = diff.reshape(-1, V_HEAD_DIM)

    signal = ref_rows.square().sum(dim=-1)
    candidate_energy = out_rows.square().sum(dim=-1)
    noise = diff_rows.square().sum(dim=-1)
    row_snr = 10.0 * torch.log10(signal / torch.clamp_min(noise, 1.0e-30))
    row_cos_diff = 1.0 - (ref_rows * out_rows).sum(dim=-1) / torch.clamp_min(
        torch.sqrt(signal * candidate_energy),
        1.0e-30,
    )
    global_snr = 10.0 * torch.log10(signal.sum() / torch.clamp_min(noise.sum(), 1.0e-30))
    global_cos_diff = 1.0 - (ref * out).sum() / torch.clamp_min(
        torch.sqrt(signal.sum() * candidate_energy.sum()),
        1.0e-30,
    )
    return {
        "snr_db": float(global_snr.item()),
        "min_row_snr_db": float(row_snr.min().item()),
        "cos_diff": float(global_cos_diff.item()),
        "max_row_cos_diff": float(row_cos_diff.max().item()),
        "finite": bool(torch.isfinite(out).all().item()),
        "nonzero": bool((candidate_energy > 0).all().item()),
    }


def _assert_quality(metrics: dict[str, Any], label: str) -> None:
    for name in ("snr_db", "min_row_snr_db", "cos_diff", "max_row_cos_diff"):
        assert math.isfinite(metrics[name]), f"{label} produced non-finite {name}"
    assert metrics["finite"], f"{label} produced non-finite output"
    assert metrics["nonzero"], f"{label} produced a zero output row"
    assert metrics["snr_db"] >= 20.0, f"{label} global SNR is {metrics['snr_db']:.3f} dB"
    assert metrics["min_row_snr_db"] >= 20.0, f"{label} worst-row SNR is {metrics['min_row_snr_db']:.3f} dB"
    assert metrics["cos_diff"] <= 3.0e-2, f"{label} cosine difference is {metrics['cos_diff']:.6f}"
    assert (
        metrics["max_row_cos_diff"] <= 3.0e-2
    ), f"{label} worst-row cosine difference is {metrics['max_row_cos_diff']:.6f}"


def _quality_failures(metrics: dict[str, Any]) -> list[str]:
    failures = []
    for name in ("snr_db", "min_row_snr_db", "cos_diff", "max_row_cos_diff"):
        if not math.isfinite(metrics[name]):
            failures.append(f"non-finite {name}")
    if not metrics["finite"]:
        failures.append("non-finite output")
    if not metrics["nonzero"]:
        failures.append("zero output row")
    if metrics["snr_db"] < 20.0:
        failures.append(f"global SNR {metrics['snr_db']:.3f} dB")
    if metrics["min_row_snr_db"] < 20.0:
        failures.append(f"worst-row SNR {metrics['min_row_snr_db']:.3f} dB")
    if metrics["cos_diff"] > 3.0e-2:
        failures.append(f"cosine difference {metrics['cos_diff']:.6f}")
    if metrics["max_row_cos_diff"] > 3.0e-2:
        failures.append(f"worst-row cosine difference {metrics['max_row_cos_diff']:.6f}")
    return failures


def _run_flydsl_correctness(runtime: V4Runtime, case: V4Case, reference: Any) -> dict[str, Any]:
    snapshots = []
    for _ in range(2):
        case.fly_output.fill_(float("nan"))
        case.fly_logits.fill_(float("nan"))
        case.fly_lse.fill_(float("nan"))
        _launch_flydsl(runtime, case)
        runtime.torch.cuda.synchronize()
        snapshots.append(case.fly_output.clone())
    assert runtime.torch.equal(snapshots[0], snapshots[1]), "FlyDSL output is not bitwise deterministic"
    metrics = _quality_metrics(runtime.torch, reference, snapshots[-1])
    _assert_quality(metrics, "FlyDSL")
    return metrics


def _run_aiter_health_audit(runtime: V4Runtime, case: V4Case, reference: Any) -> dict[str, Any]:
    """Validate a fixed, non-retried sequence before reporting ASM timing."""
    call_count = int(os.environ.get("FLYDSL_V4_AITER_HEALTH_CALLS", "32"))
    if call_count < 32:
        raise ValueError("FLYDSL_V4_AITER_HEALTH_CALLS must be at least 32")
    calls = []
    failures = []
    for index in range(call_count):
        case.aiter_output.fill_(float("nan"))
        case.aiter_logits.fill_(float("nan"))
        case.aiter_lse.fill_(float("nan"))
        _launch_aiter_asm(runtime, case)
        runtime.torch.cuda.synchronize()
        metrics = _quality_metrics(runtime.torch, reference, case.aiter_output)
        calls.append(metrics)
        errors = _quality_failures(metrics)
        if errors:
            failures.append({"call": index + 1, "errors": errors})
    health: dict[str, Any] = {
        "call_count": call_count,
        "failure_count": len(failures),
        "failures": failures,
        "calls": [
            {
                "call": index,
                "snr_db": call["snr_db"],
                "min_row_snr_db": call["min_row_snr_db"],
                "cos_diff": call["cos_diff"],
                "max_row_cos_diff": call["max_row_cos_diff"],
            }
            for index, call in enumerate(calls, start=1)
        ],
        "worst": {
            "snr_db": min(call["snr_db"] for call in calls),
            "min_row_snr_db": min(call["min_row_snr_db"] for call in calls),
            "cos_diff": max(call["cos_diff"] for call in calls),
            "max_row_cos_diff": max(call["max_row_cos_diff"] for call in calls),
        },
    }
    print("V4_AITER_ASM_HEALTH " + json.dumps(health, sort_keys=True))
    assert not failures, "AITER ASM comparator is unhealthy; timing and speedup are invalid: " + json.dumps(
        failures, sort_keys=True
    )
    return health


def _device_kernel_names(
    runtime: V4Runtime,
    launch: Callable[[], None],
    *,
    label: str,
    expected: str,
) -> list[str]:
    torch = runtime.torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
        launch()
        torch.cuda.synchronize()
    names = sorted(
        {
            event.name
            for event in profile.events()
            if "CUDA" in str(event.device_type) and float(event.self_device_time_total) > 0
        }
    )
    assert any(expected in name for name in names), (
        f"{label} did not expose expected device-symbol fragment {expected!r}; " f"captured device kernels: {names}"
    )
    return names


def _validate_output_pool(
    runtime: V4Runtime,
    reference: Any,
    outputs: Any,
    *,
    label: str,
    repeat: int,
) -> dict[str, Any]:
    calls = []
    failures = []
    for index, output in enumerate(outputs, start=1):
        metrics = _quality_metrics(runtime.torch, reference, output)
        calls.append(metrics)
        errors = _quality_failures(metrics)
        if errors:
            failures.append({"call": index, "errors": errors})
    summary = {
        "label": label,
        "repeat": repeat,
        "call_count": len(calls),
        "failure_count": len(failures),
        "failures": failures,
        "worst": {
            "snr_db": min(call["snr_db"] for call in calls),
            "min_row_snr_db": min(call["min_row_snr_db"] for call in calls),
            "cos_diff": max(call["cos_diff"] for call in calls),
            "max_row_cos_diff": max(call["max_row_cos_diff"] for call in calls),
        },
    }
    print("V4_TIMED_OUTPUT_HEALTH " + json.dumps(summary, sort_keys=True))
    assert not failures, f"{label} had invalid measured outputs: " + json.dumps(failures, sort_keys=True)
    return summary


def _paired_benchmark(runtime: V4Runtime, case: V4Case, reference: Any) -> dict[str, Any]:
    warmup = int(os.environ.get("FLYDSL_V4_BENCH_WARMUP", "10"))
    iterations = int(os.environ.get("FLYDSL_V4_BENCH_ITERS", "100"))
    repeats = int(os.environ.get("FLYDSL_V4_BENCH_REPEATS", "4"))
    if warmup < 1 or iterations < 1:
        raise ValueError("benchmark warmup and iteration counts must be positive")
    if repeats < 2 or repeats % 2:
        raise ValueError("FLYDSL_V4_BENCH_REPEATS must be a positive even number of at least 2")

    launchers = {
        "flydsl": lambda output: _launch_flydsl(runtime, case, output=output),
        "aiter_asm": lambda output: _launch_aiter_asm(runtime, case, output=output),
    }
    output_shape = (iterations, BATCH, NUM_HEADS, V_HEAD_DIM)
    output_pools = {
        name: runtime.torch.empty(
            output_shape,
            dtype=runtime.torch.bfloat16,
            device=case.q_packed.device,
        )
        for name in launchers
    }

    for _ in range(warmup):
        _launch_flydsl(runtime, case)
        _launch_aiter_asm(runtime, case)
    runtime.torch.cuda.synchronize()
    _assert_quality(_quality_metrics(runtime.torch, reference, case.fly_output), "FlyDSL warmup")
    _assert_quality(_quality_metrics(runtime.torch, reference, case.aiter_output), "AITER ASM warmup")

    samples = {name: [] for name in launchers}
    validation = {name: [] for name in launchers}
    start = runtime.torch.cuda.Event(enable_timing=True)
    end = runtime.torch.cuda.Event(enable_timing=True)
    for repeat in range(repeats):
        order = ("flydsl", "aiter_asm") if repeat % 2 == 0 else ("aiter_asm", "flydsl")
        for name in order:
            outputs = output_pools[name]
            outputs.fill_(float("nan"))
            start.record()
            for output in outputs:
                launchers[name](output)
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1.0e3 / iterations)
            validation[name].append(
                _validate_output_pool(
                    runtime,
                    reference,
                    outputs,
                    label=name,
                    repeat=repeat + 1,
                )
            )
    flydsl_us = statistics.median(samples["flydsl"])
    aiter_us = statistics.median(samples["aiter_asm"])
    return {
        "shape": "B256_L1152_H16_split1",
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "flydsl_samples_us": samples["flydsl"],
        "aiter_asm_samples_us": samples["aiter_asm"],
        "measured_output_validation": validation,
        "flydsl_us": flydsl_us,
        "aiter_asm_us": aiter_us,
        "speedup_vs_aiter_asm": aiter_us / flydsl_us,
        "latency_reduction_pct": 100.0 * (aiter_us - flydsl_us) / aiter_us,
    }


def _aiter_provenance(runtime: V4Runtime) -> dict[str, Any]:
    assert runtime.aiter is not None
    module_path = Path(runtime.aiter.__file__).resolve(strict=True)
    repo_root = module_path.parent.parent
    code_object = repo_root / "hsa" / "gfx950" / "mla_v4" / AITER_ASM_CODE_OBJECT
    if not code_object.is_file():
        raise RuntimeError(f"the expected AITER ASM code object is missing: {code_object}")
    code_object_sha256 = hashlib.sha256(code_object.read_bytes()).hexdigest()
    expected_sha256 = os.environ.get("FLYDSL_V4_AITER_ASM_SHA256")
    if expected_sha256 and expected_sha256 != code_object_sha256:
        raise RuntimeError(
            "AITER ASM code-object hash mismatch: " f"expected {expected_sha256}, got {code_object_sha256}"
        )

    git_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    version = getattr(runtime.aiter, "__version__", None)
    return {
        "module_path": str(module_path),
        "version": None if version is None else str(version),
        "git_head": git_head.stdout.strip() if git_head.returncode == 0 else None,
        "asm_symbol": AITER_ASM_SYMBOL,
        "code_object": str(code_object.relative_to(repo_root)),
        "code_object_sha256": code_object_sha256,
    }


def _parent_preflight(*, require_aiter: bool) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch is not installed")
    if not torch.cuda.is_available():
        pytest.skip("a ROCm GPU is required")
    arch = torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        pytest.skip(f"the V4 sparse-decode candidate requires gfx950, got {arch}")
    if require_aiter and not maybe_enable_aiter():
        pytest.fail("AITER is not importable; set AITER_REPO to an AITER checkout", pytrace=False)


def _run_driver(mode: str) -> None:
    env = os.environ.copy()
    for name in (
        "ARCH",
        "COMPILE_ONLY",
        "FLYDSL_DUMP_IR",
        "FLYDSL_EXTRA_SOURCE_DIRS",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FLYDSL_RUNTIME_RUN_ONLY",
    ):
        env.pop(name, None)
    for name in tuple(env):
        if name.startswith("FLYDSL_V4_N256_"):
            env.pop(name)
    env["FLYDSL_V4_FORGE_SELECTED_REGISTRATION"] = "1"
    env["FLYDSL_V4_FORGE_EXPECTED_VARIANT"] = "m16s536"
    env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    env["GPU_TARGET"] = "gfx950"
    python_paths = [str(ROOT / "build-fly" / "python_packages"), str(ROOT)]
    if env.get("AITER_REPO"):
        python_paths.append(env["AITER_REPO"])
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--mode", mode],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    assert completed.returncode == 0, f"V4 sparse-decode {mode} driver exited {completed.returncode}"


def _driver_correctness() -> None:
    runtime = _load_runtime(require_aiter=False)
    for seed in (20260725, 20260726):
        case = _make_case(runtime, seed)
        reference = _torch_reference(runtime, case)
        metrics = _run_flydsl_correctness(runtime, case, reference)
        print("V4_FLYDSL_CORRECTNESS " + json.dumps({"seed": seed, **metrics}, sort_keys=True))
        del case, reference
        runtime.torch.cuda.empty_cache()


def _driver_benchmark() -> None:
    runtime = _load_runtime(require_aiter=True)
    aiter_provenance = _aiter_provenance(runtime)
    case = _make_case(runtime, 20260725)
    reference = _torch_reference(runtime, case)
    _run_flydsl_correctness(runtime, case, reference)
    aiter_health = _run_aiter_health_audit(runtime, case, reference)
    flydsl_kernel_names = _device_kernel_names(
        runtime,
        lambda: _launch_flydsl(runtime, case),
        label="FlyDSL",
        expected="m16s536_lds_stride",
    )
    _assert_quality(
        _quality_metrics(runtime.torch, reference, case.fly_output),
        "FlyDSL symbol probe",
    )
    aiter_kernel_names = _device_kernel_names(
        runtime,
        lambda: _launch_aiter_asm(runtime, case),
        label="AITER ASM",
        expected=AITER_ASM_SYMBOL,
    )
    _assert_quality(
        _quality_metrics(runtime.torch, reference, case.aiter_output),
        "AITER ASM symbol probe",
    )

    result = _paired_benchmark(runtime, case, reference)
    source_path = Path(runtime.flydsl_module.__file__).resolve()
    result["flydsl_source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    result["registration"] = dict(runtime.flydsl_module.FORGE_REGISTRATION_IDENTITY)
    result["gpu_arch"] = "gfx950"
    result["gpu_name"] = runtime.torch.cuda.get_device_name()
    result["torch_version"] = runtime.torch.__version__
    result["rocm_version"] = runtime.torch.version.hip
    result["aiter_health"] = aiter_health
    result["aiter"] = aiter_provenance
    result["flydsl_device_kernels"] = flydsl_kernel_names
    result["aiter_device_kernels"] = aiter_kernel_names
    print("V4_SPARSE_DECODE_BENCH " + json.dumps(result, sort_keys=True))

    if os.environ.get("FLYDSL_V4_REQUIRE_AITER_WIN") == "1":
        assert os.environ.get(
            "FLYDSL_V4_AITER_ASM_SHA256"
        ), "FLYDSL_V4_REQUIRE_AITER_WIN=1 requires a pinned FLYDSL_V4_AITER_ASM_SHA256"
        assert (
            result["speedup_vs_aiter_asm"] > 1.0
        ), f"FlyDSL did not beat AITER ASM: speedup={result['speedup_vs_aiter_asm']:.6f}x"


@pytest.mark.large_shape
def test_mla_v4_sparse_decode_exact_correctness() -> None:
    _require_opt_in("FLYDSL_RUN_V4_MLA_TESTS")
    _parent_preflight(require_aiter=False)
    _run_driver("correctness")


@pytest.mark.large_shape
@pytest.mark.benchmark
def test_mla_v4_sparse_decode_benchmark_against_aiter_asm() -> None:
    _require_opt_in("FLYDSL_BENCH")
    _parent_preflight(require_aiter=True)
    _run_driver("benchmark")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("correctness", "benchmark"))
    args = parser.parse_args()
    if args.mode == "correctness":
        _driver_correctness()
    else:
        _driver_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
