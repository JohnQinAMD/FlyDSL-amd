# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse MLA decode -- promoted M16/S536 gfx950 path.

One workgroup owns one decode row and all 16 query heads. The release wrapper is
validated for B=256, KV length=1152, split=1.

Do not add ``from __future__ import annotations`` here: FlyDSL needs concrete JIT
launcher annotations while tracing.
"""

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.attention._mla_v4_sparse_decode_common import (
    BLOCK_N,
    DIM_PACKED,
    DIM_ROPE,
    M16S536_DYNAMIC_LDS_BYTES,
    M16S536_KV_LDS_STRIDE,
    NUM_HEADS,
    NUM_KV_HEADS,
    NUM_THREADS,
    OCCUPANCY,
    PAGE_SIZE,
    V_HEAD_DIM,
)
from kernels.attention._mla_v4_sparse_decode_kernel import _kn_mla_v4_sparse_decode_mfma_impl

# Release geometry

KERNEL_NAME = "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride"
KERNEL_BLOCK = (512, 1, 1)
RELEASE_BATCH = 256
RELEASE_ROW_LEN = 1152
RELEASE_SPLITS = 1


# KernelForge registration contract

FORGE_VARIANT = "m16s536"
FORGE_SELECTED_REGISTRATION_ENV = "FLYDSL_V4_FORGE_SELECTED_REGISTRATION"
FORGE_EXPECTED_VARIANT_ENV = "FLYDSL_V4_FORGE_EXPECTED_VARIANT"
FORGE_REGISTRATION_SCHEMA = "kernelforge.v4_decode_registration.v1"
FORGE_REGISTRATION_CONTRACT = (
    (
        "m16s536",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        (512, 1, 1),
    ),
)
FORGE_HEAVY_KERNEL_GLOBALS = ("kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",)


def _registration_mode() -> str:
    selected = os.environ.get(FORGE_SELECTED_REGISTRATION_ENV, "0")
    expected = os.environ.get(FORGE_EXPECTED_VARIANT_ENV)

    if selected == "0":
        if expected is not None:
            raise RuntimeError(f"{FORGE_EXPECTED_VARIANT_ENV} requires {FORGE_SELECTED_REGISTRATION_ENV}=1")
        return "all"
    if selected != "1":
        raise RuntimeError(f"{FORGE_SELECTED_REGISTRATION_ENV} must be exactly '0' or '1'; got {selected!r}")
    if expected is None:
        raise RuntimeError(f"{FORGE_EXPECTED_VARIANT_ENV} is required for selected registration")
    if expected != FORGE_VARIANT:
        raise RuntimeError(f"{FORGE_EXPECTED_VARIANT_ENV}={expected!r} does not match {FORGE_VARIANT!r}")
    return "selected"


FORGE_REGISTRATION_IDENTITY = {
    "schema": FORGE_REGISTRATION_SCHEMA,
    "mode": _registration_mode(),
    "variant": FORGE_VARIANT,
    "kernel_global": KERNEL_NAME,
    "symbol": KERNEL_NAME,
    "block": KERNEL_BLOCK,
    "registered_heavy_count": len(FORGE_HEAVY_KERNEL_GLOBALS),
}


# JIT launcher

kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride = flyc.kernel(
    _kn_mla_v4_sparse_decode_mfma_impl,
    name=KERNEL_NAME,
    known_block_size=list(KERNEL_BLOCK),
)


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    output: fx.Tensor,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch the exact-shape M16/S536 sparse-decode kernel."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indptr,
        kv_indices,
        sink,
        output,
        softmax_scale,
        value_attrs={
            "passthrough": [
                ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(grid=(num_sequences, 1, 1), block=KERNEL_BLOCK, stream=stream)


launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


# Host-side validation and public wrapper


def _require_tensor(name, tensor, *, dtype, shape=None):
    if tensor.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: expected contiguous storage, got stride={tensor.stride()}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")


def _require_release_shape(num_sequences: int, num_kv_splits: int, kv_indices) -> None:
    failures = []
    if num_sequences != RELEASE_BATCH:
        failures.append(f"num_sequences={num_sequences}, expected {RELEASE_BATCH}")
    if num_kv_splits != RELEASE_SPLITS:
        failures.append(f"num_kv_splits={num_kv_splits}, expected {RELEASE_SPLITS}")

    expected_indices = RELEASE_BATCH * RELEASE_ROW_LEN
    if kv_indices.numel() != expected_indices:
        failures.append(f"kv_indices.numel()={kv_indices.numel()}, expected {expected_indices}")
    if failures:
        raise NotImplementedError(
            "DSV4 sparse decode validated release path supports only B=256, "
            "uniform KV row length=1152, num_kv_splits=1; failed: " + "; ".join(failures)
        )


def _require_identity_indptr(name, tensor, count: int) -> None:
    expected = torch.arange(count + 1, dtype=tensor.dtype, device=tensor.device)
    if not torch.equal(tensor, expected):
        raise NotImplementedError(
            f"{name}: validated release path requires identity mapping [0..{count}] "
            "because the single-split kernel maps one CTA directly to one sequence row"
        )


def _require_uniform_kv_indptr(name, tensor, count: int, row_len: int) -> None:
    expected = torch.arange(count + 1, dtype=tensor.dtype, device=tensor.device) * row_len
    if not torch.equal(tensor, expected):
        raise NotImplementedError(
            f"{name}: validated release path requires uniform row length {row_len} for {count} rows"
        )


def flydsl_mla_v4_sparse_decode(
    *,
    q_packed,
    q_rope,
    kv_packed,
    kv_rope,
    output,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sink,
    split_indptr,
    num_kv_splits,
    logits,
    attn_lse,
    softmax_scale,
):
    """KernelForge-compatible DeepSeek-V4 sparse decode wrapper.

    Validated release constraints: gfx950, ``B=256``, uniform KV row length
    ``1152``, ``num_kv_splits=1``, identity ``qo_indptr`` / ``split_indptr``.
    The kernel writes ``output`` directly; ``logits`` and ``attn_lse`` must be ``None``.
    """
    if not isinstance(num_kv_splits, int) or num_kv_splits <= 0:
        raise ValueError(f"num_kv_splits must be a positive int, got {num_kv_splits!r}")
    if logits is not None or attn_lse is not None:
        raise ValueError(
            "logits and attn_lse must be None: this validated single-split path "
            "writes output directly and does not populate split-KV scratch buffers"
        )

    device = q_packed.device
    if device.type != "cuda":
        raise ValueError(f"q_packed must be on an AMD GPU, got {device}")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if not arch.startswith("gfx950"):
        raise NotImplementedError(f"DeepSeek-V4 sparse decode currently requires gfx950, got {arch}")

    if q_packed.ndim != 3:
        raise ValueError("q_packed must be [num_sequences, 16, 512]")
    if kv_packed.ndim != 4:
        raise ValueError("kv_packed must be [num_physical_rows, 1, 1, 512]")

    num_sequences = q_packed.size(0)
    num_physical_rows = kv_packed.size(0)
    tensor_specs = (
        ("q_packed", q_packed, torch.float8_e4m3fn, (num_sequences, NUM_HEADS, DIM_PACKED)),
        ("q_rope", q_rope, torch.bfloat16, (num_sequences, NUM_HEADS, DIM_ROPE)),
        ("kv_packed", kv_packed, torch.float8_e4m3fn, (num_physical_rows, PAGE_SIZE, NUM_KV_HEADS, DIM_PACKED)),
        ("kv_rope", kv_rope, torch.bfloat16, (num_physical_rows, PAGE_SIZE, NUM_KV_HEADS, DIM_ROPE)),
        ("output", output, torch.bfloat16, (num_sequences, NUM_HEADS, V_HEAD_DIM)),
        ("qo_indptr", qo_indptr, torch.int32, (num_sequences + 1,)),
        ("kv_indptr", kv_indptr, torch.int32, (num_sequences + 1,)),
        ("split_indptr", split_indptr, torch.int32, (num_sequences + 1,)),
        ("kv_indices", kv_indices, torch.int32, None),
        ("sink", sink, torch.float32, (NUM_HEADS,)),
    )
    for name, tensor, dtype, shape in tensor_specs:
        _require_tensor(name, tensor, dtype=dtype, shape=shape)
        if tensor.device != device:
            raise ValueError(f"{name}: expected device {device}, got {tensor.device}")

    _require_release_shape(num_sequences, num_kv_splits, kv_indices)
    _require_identity_indptr("qo_indptr", qo_indptr, num_sequences)
    _require_identity_indptr("split_indptr", split_indptr, num_sequences)
    _require_uniform_kv_indptr("kv_indptr", kv_indptr, num_sequences, RELEASE_ROW_LEN)
    launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        kv_indptr,
        kv_indices,
        sink,
        output,
        float(softmax_scale),
        num_sequences=num_sequences,
        stream=torch.cuda.current_stream(device),
    )
    if os.environ.get("FLYDSL_V4_DIAGNOSTIC_SYNC_STAGE1") == "1":
        torch.cuda.synchronize(device)


__all__ = [
    "BLOCK_N",
    "FORGE_EXPECTED_VARIANT_ENV",
    "FORGE_HEAVY_KERNEL_GLOBALS",
    "FORGE_REGISTRATION_CONTRACT",
    "FORGE_REGISTRATION_IDENTITY",
    "FORGE_REGISTRATION_SCHEMA",
    "FORGE_SELECTED_REGISTRATION_ENV",
    "FORGE_VARIANT",
    "M16S536_DYNAMIC_LDS_BYTES",
    "M16S536_KV_LDS_STRIDE",
    "NUM_HEADS",
    "NUM_THREADS",
    "OCCUPANCY",
    "flydsl_mla_v4_sparse_decode",
    "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic",
]
