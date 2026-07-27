# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 sparse MLA decode candidate for gfx950.

This is the correctness-first H16 schedule used by the KernelForge campaign:

* packed 512-byte FP8 NoPE records (448 data bytes, duplicated E8M0 scales);
* a separate BF16 RoPE64 buffer;
* page-size-one flat CSR indirection;
* one FP32 attention sink per query head;
* runtime split-KV with direct BF16 output for one split and normalized FP32
  partial output/LSE for multiple splits.

The stage-1 grid is direct ``(sequence, split)``.  The default workgroup has
four physical waves and reduces four 16-token QK wave tiles into one 64-token
online-softmax step through LDS.  Narrowly dispatched N128 specializations
instead use eight waves and a 128-token step for the saturated N256
single-split and uniform N64/split4 campaign points.  For PV, each wave owns a
disjoint output slice and consumes the probabilities produced by all waves.
Both products use ``v_mfma_f32_16x16x32_bf16`` after exact FP8/E8M0
dequantization.

Each KV tile is dequantized once into BF16 LDS and then reused by QK and PV.
The default uses 68,352 bytes and permits two workgroups per CU on gfx950; the
baseline N256 specialization uses exactly 136,704 bytes for one eight-wave
workgroup.  The opt-in M16 S536 experiment pads only the decoded KV rows and
uses 142,848 bytes in the same one-workgroup residency class.  Q stays resident
in VGPRs, and no persistent scheduler or metadata kernel is in the timed path.

Do not add ``from __future__ import annotations``: FlyDSL uses the concrete
annotations on the JIT launcher while tracing.
"""

import functools
import os
import types

import torch
import triton
import triton.language as tl

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr import math as fmath
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from kernels.common import dpp_utils

# KernelForge changes only this selector between frozen control commits.  New
# mechanisms keep the selected launcher stable and modify its implementation,
# so matched timing binds both the source and the compiled runtime artifact.
FORGE_VARIANT: str = "m16s536"

FORGE_SELECTED_REGISTRATION_ENV: str = "FLYDSL_V4_FORGE_SELECTED_REGISTRATION"
FORGE_EXPECTED_VARIANT_ENV: str = "FLYDSL_V4_FORGE_EXPECTED_VARIANT"
FORGE_REGISTRATION_SCHEMA: str = "kernelforge.v4_decode_registration.v1"
FORGE_REGISTRATION_CONTRACT = (
    (
        "m8",
        "kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode",
        "kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode",
        (256, 1, 1),
    ),
    (
        "m9",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill",
        (512, 1, 1),
    ),
    (
        "m10",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead",
        (512, 1, 1),
    ),
    (
        "m11",
        "kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline",
        "kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline",
        (256, 1, 1),
    ),
    (
        "m12l",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax",
        (512, 1, 1),
    ),
    (
        "m15",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder",
        (512, 1, 1),
    ),
    (
        "m16s536",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
        (512, 1, 1),
    ),
)
FORGE_HEAVY_KERNEL_GLOBALS = (
    "kn_mla_v4_sparse_decode_mfma",
    "kn_mla_v4_sparse_decode_mfma_n128",
    "kn_mla_v4_sparse_decode_mfma_n128_token4",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr",
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l",
    "kn_mla_v4_sparse_decode_mfma_n128_split4",
    "kn_mla_v4_sparse_decode_mfma_qh64_b32",
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m1",
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode",
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline",
)
_FORGE_CANONICAL_REGISTRATIONS = types.MappingProxyType(
    {variant: (kernel_global, symbol, block) for variant, kernel_global, symbol, block in FORGE_REGISTRATION_CONTRACT}
)


def _resolve_forge_registration(variant, raw_mode, expected_variant):
    """Resolve the import-time heavy-registration mode without ambient fallback."""
    if variant not in _FORGE_CANONICAL_REGISTRATIONS:
        raise RuntimeError(f"unsupported literal FORGE_VARIANT={variant!r}")
    if raw_mode not in {"0", "1"}:
        raise RuntimeError(f"{FORGE_SELECTED_REGISTRATION_ENV} must be exactly '0' or '1'; got {raw_mode!r}")
    if raw_mode == "0":
        if expected_variant is not None:
            raise RuntimeError(
                f"{FORGE_EXPECTED_VARIANT_ENV} requires {FORGE_SELECTED_REGISTRATION_ENV}=1; "
                f"got expected variant {expected_variant!r}"
            )
        return False
    if expected_variant is None:
        raise RuntimeError(f"{FORGE_EXPECTED_VARIANT_ENV} is required for selected registration")
    if expected_variant != variant:
        raise RuntimeError(
            f"{FORGE_EXPECTED_VARIANT_ENV}={expected_variant!r} does not match literal FORGE_VARIANT={variant!r}"
        )
    return True


_FORGE_SELECTED_REGISTRATION = _resolve_forge_registration(
    FORGE_VARIANT,
    os.environ.get(FORGE_SELECTED_REGISTRATION_ENV, "0"),
    os.environ.get(FORGE_EXPECTED_VARIANT_ENV),
)

NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
DIM_NOPE: int = 448
DIM_ROPE: int = 64
DIM_QK: int = DIM_NOPE + DIM_ROPE
DIM_PACKED: int = 512
V_HEAD_DIM: int = 512
PAGE_SIZE: int = 1

NUM_WARPS: int = 4
WARP_SIZE: int = 64
NUM_THREADS: int = NUM_WARPS * WARP_SIZE
HEADS_PER_WAVE: int = 4
LANES_PER_HEAD: int = 16
DIMS_PER_LANE: int = DIM_QK // LANES_PER_HEAD
BLOCK_N: int = 64
OCCUPANCY: int = 2

PACKED_DWORDS: int = DIM_PACKED // 4
NOPE_DWORDS_PER_LANE: int = DIMS_PER_LANE // 4
NOPE_SCALE_OFFSET: int = DIM_NOPE
MFMA_K: int = 32
MFMA_INPUT_VALUES: int = 8
MFMA_OUTPUT_VALUES: int = 4
TOKENS_PER_WAVE: int = 16
N256_SINGLE_NUM_WARPS: int = 8
N256_SINGLE_NUM_THREADS: int = N256_SINGLE_NUM_WARPS * WARP_SIZE
N256_SINGLE_BLOCK_N: int = N256_SINGLE_NUM_WARPS * TOKENS_PER_WAVE
M16S536_KV_LDS_STRIDE: int = 536
M16S536_DYNAMIC_LDS_BYTES: int = 142_848
QH64_B32_NUM_WARPS: int = 4
QH64_B32_NUM_THREADS: int = QH64_B32_NUM_WARPS * WARP_SIZE
QH64_B32_BLOCK_N: int = 32
QH64_B32_KV_TILE_ELEMS: int = QH64_B32_BLOCK_N * DIM_QK
QH64_PHYSICAL_QUERY_ROWS: int = QH64_B32_NUM_WARPS * NUM_HEADS
QH64_OUTPUT_MFMA_TILES: int = V_HEAD_DIM // 16
QH64_QUERY_ELEMS: int = NUM_HEADS * DIM_QK
QH64_M11_RAW_BANKS: int = 4
QH64_M11_RAW_PACKED_BYTES: int = QH64_B32_BLOCK_N * DIM_PACKED
QH64_M11_RAW_ROPE_BYTES: int = QH64_B32_BLOCK_N * DIM_ROPE * 2
QH64_M11_RAW_BANK_BYTES: int = QH64_M11_RAW_PACKED_BYTES + QH64_M11_RAW_ROPE_BYTES
QH64_M11_RAW_TOTAL_BYTES: int = QH64_M11_RAW_BANKS * QH64_M11_RAW_BANK_BYTES
QH64_M11_DECODED_PONGS: int = 2
QH64_M11_DECODED_ELEMS: int = QH64_M11_DECODED_PONGS * QH64_B32_KV_TILE_ELEMS
OUTPUT_DIMS_PER_WAVE: int = V_HEAD_DIM // NUM_WARPS
OUTPUT_MFMA_TILES_PER_WAVE: int = OUTPUT_DIMS_PER_WAVE // 16
KV_TILE_ELEMS: int = BLOCK_N * DIM_QK
P_TILE_ELEMS: int = NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES
LOG2E: float = 1.4426950408889634
INV_LOG2E: float = 1.0 / LOG2E

assert DIM_QK == V_HEAD_DIM == 512
assert DIMS_PER_LANE == 32
assert DIM_NOPE // DIMS_PER_LANE == 14
assert NUM_HEADS == NUM_WARPS * HEADS_PER_WAVE
assert BLOCK_N == NUM_WARPS * TOKENS_PER_WAVE
assert DIM_QK % MFMA_K == 0
assert OUTPUT_DIMS_PER_WAVE == 128
assert N256_SINGLE_BLOCK_N == 128
assert QH64_PHYSICAL_QUERY_ROWS == 64
assert QH64_B32_KV_TILE_ELEMS * 2 == 32 * 1024
assert QH64_QUERY_ELEMS * 2 == 16 * 1024
assert _FORGE_CANONICAL_REGISTRATIONS["m8"][2] == (QH64_B32_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m9"][2] == (N256_SINGLE_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m10"][2] == (N256_SINGLE_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m11"][2] == (QH64_B32_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m12l"][2] == (N256_SINGLE_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m15"][2] == (N256_SINGLE_NUM_THREADS, 1, 1)
assert _FORGE_CANONICAL_REGISTRATIONS["m16s536"][2] == (N256_SINGLE_NUM_THREADS, 1, 1)
assert M16S536_KV_LDS_STRIDE % MFMA_INPUT_VALUES == 0
assert (
    N256_SINGLE_BLOCK_N * M16S536_KV_LDS_STRIDE * 2
    + N256_SINGLE_NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES * 2
    + 2 * N256_SINGLE_NUM_WARPS * NUM_HEADS * 4
    + N256_SINGLE_BLOCK_N * 4
    == M16S536_DYNAMIC_LDS_BYTES
)
assert QH64_M11_RAW_PACKED_BYTES == 16 * 1024
assert QH64_M11_RAW_ROPE_BYTES == 4 * 1024
assert QH64_M11_RAW_TOTAL_BYTES + QH64_M11_DECODED_ELEMS * 2 + QH64_QUERY_ELEMS * 2 == 160 * 1024


@fx.struct
class SparseDecodeShared:
    """LDS reused by every 64-token online-softmax iteration."""

    kv: fx.Array[fx.BFloat16, KV_TILE_ELEMS, 16]
    probability: fx.Array[fx.BFloat16, P_TILE_ELEMS, 16]
    reduction: fx.Array[fx.Float32, 2 * NUM_WARPS * NUM_HEADS, 16]
    physical_rows: fx.Array[fx.Int32, BLOCK_N, 16]


@fx.struct
class SparseDecodeSharedN128:
    """One-CU-wide LDS tile for the narrow N128 specializations."""

    kv: fx.Array[
        fx.BFloat16,
        N256_SINGLE_BLOCK_N * DIM_QK,
        16,
    ]
    probability: fx.Array[
        fx.BFloat16,
        N256_SINGLE_NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES,
        16,
    ]
    reduction: fx.Array[
        fx.Float32,
        2 * N256_SINGLE_NUM_WARPS * NUM_HEADS,
        16,
    ]
    physical_rows: fx.Array[fx.Int32, N256_SINGLE_BLOCK_N, 16]


@fx.struct
class SparseDecodeSharedN128M16S536:
    """N128 workspace with a 24-BF16 bank-breaking decoded-KV row pad."""

    kv: fx.Array[
        fx.BFloat16,
        N256_SINGLE_BLOCK_N * M16S536_KV_LDS_STRIDE,
        16,
    ]
    probability: fx.Array[
        fx.BFloat16,
        N256_SINGLE_NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES,
        16,
    ]
    reduction: fx.Array[
        fx.Float32,
        2 * N256_SINGLE_NUM_WARPS * NUM_HEADS,
        16,
    ]
    physical_rows: fx.Array[fx.Int32, N256_SINGLE_BLOCK_N, 16]


@fx.struct
class SparseDecodeSharedN128RawG2L:
    """N128 tile plus one wave-partitioned raw pair staging slot."""

    kv: fx.Array[
        fx.BFloat16,
        N256_SINGLE_BLOCK_N * DIM_QK,
        16,
    ]
    probability: fx.Array[
        fx.BFloat16,
        N256_SINGLE_NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES,
        16,
    ]
    reduction: fx.Array[
        fx.Float32,
        2 * N256_SINGLE_NUM_WARPS * NUM_HEADS,
        16,
    ]
    physical_rows: fx.Array[fx.Int32, N256_SINGLE_BLOCK_N, 16]
    raw_packed: fx.Array[fx.Int8, 16 * 1024, 16]
    raw_rope: fx.Array[fx.Int8, 8 * 1024, 16]


@fx.struct
class SparseDecodeSharedQH64B32:
    """Synchronous workspace for the diagnostic padded-query path."""

    kv: fx.Array[
        fx.BFloat16,
        QH64_B32_KV_TILE_ELEMS,
        16,
    ]
    query: fx.Array[
        fx.BFloat16,
        QH64_QUERY_ELEMS,
        16,
    ]


@fx.struct
class SparseDecodeSharedQH64B32M11:
    """Four raw banks, two decoded pongs, and resident Q in exactly 160 KiB."""

    raw: fx.Array[
        fx.Int8,
        QH64_M11_RAW_TOTAL_BYTES,
        16,
    ]
    kv: fx.Array[
        fx.BFloat16,
        QH64_M11_DECODED_ELEMS,
        16,
    ]
    query: fx.Array[
        fx.BFloat16,
        QH64_QUERY_ELEMS,
        16,
    ]


def _i32(value):
    """Cast an integer-like DSL value to i32."""
    raw = _raw(value) if not isinstance(value, ir.Value) else value
    if raw.type == T.i32:
        return raw
    return _raw(fx.Int32(raw))


def _idx(value):
    """Cast an integer-like DSL value to index."""
    if isinstance(value, fx.Index):
        return value
    return fx.Index(value)


def _select(cond, true_value, false_value):
    """Typed SSA select returning the raw result."""
    return _raw(
        ArithValue(_raw(cond)).select(
            _raw(true_value),
            _raw(false_value),
        )
    )


def _fast_exp(value):
    """Natural exponential through the gfx950 base-2 instruction."""
    scaled = arith.mulf(
        _raw(value),
        _raw(fx.Float32(LOG2E)),
        fastmath=arith.FastMathFlags.fast,
    )
    return rocdl.exp2(T.f32, scaled)


def _e8m0_to_f32(scale_i32):
    """Convert an E8M0 biased exponent byte to its exact f32 power of two."""
    return llvm.inline_asm(
        T.f32,
        [_raw(scale_i32)],
        "v_lshlrev_b32 $0, 23, $1",
        "=v,v",
        has_side_effects=True,
    )


@flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
def kn_mla_v4_sparse_decode(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
):
    """Four-wave H16 page-one sparse decode stage 1."""

    fm_fast = arith.FastMathFlags.fast
    fm_no_inf = (
        arith.FastMathFlags.nnan
        | arith.FastMathFlags.nsz
        | arith.FastMathFlags.arcp
        | arith.FastMathFlags.contract
        | arith.FastMathFlags.afn
        | arith.FastMathFlags.reassoc
    )

    c_zero = fx.Float32(0.0)
    c_one = fx.Float32(1.0)
    c_neg_inf = fx.Float32(float("-inf"))

    seq_idx = gpu.block_id("x")
    split_idx = gpu.block_id("y")
    tid = gpu.thread_id("x")

    seq_i32 = _i32(seq_idx)
    split_i32 = _i32(split_idx)
    tid_i32 = _i32(tid)
    wave_i32 = _raw(ArithValue(tid_i32) // fx.Int32(WARP_SIZE))
    lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(WARP_SIZE))
    head_in_wave_i32 = _raw(ArithValue(lane_i32) // fx.Int32(LANES_PER_HEAD))
    lane_in_head_i32 = _raw(ArithValue(lane_i32) % fx.Int32(LANES_PER_HEAD))
    head_i32 = _raw(ArithValue(wave_i32) * fx.Int32(HEADS_PER_WAVE) + ArithValue(head_in_wave_i32))
    dim_base_i32 = _raw(ArithValue(lane_in_head_i32) * fx.Int32(DIMS_PER_LANE))

    q_packed_rsrc = buffer_ops.create_buffer_resource(q_packed)
    q_rope_rsrc = buffer_ops.create_buffer_resource(q_rope)
    kv_packed_rsrc = buffer_ops.create_buffer_resource(kv_packed)
    kv_rope_rsrc = buffer_ops.create_buffer_resource(kv_rope)
    qo_indptr_rsrc = buffer_ops.create_buffer_resource(qo_indptr)
    kv_indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr)
    kv_indices_rsrc = buffer_ops.create_buffer_resource(kv_indices)
    sink_rsrc = buffer_ops.create_buffer_resource(sink)
    split_indptr_rsrc = buffer_ops.create_buffer_resource(split_indptr)
    output_rsrc = buffer_ops.create_buffer_resource(output)
    logits_rsrc = buffer_ops.create_buffer_resource(logits)
    attn_lse_rsrc = buffer_ops.create_buffer_resource(attn_lse)

    # These pointers are wave-uniform.  The scalar-buffer path avoids issuing
    # 64 identical vector loads per wave.
    q_row_i32 = buffer_ops.buffer_load(
        qo_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    kv_start_i32 = buffer_ops.buffer_load(
        kv_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    kv_end_i32 = buffer_ops.buffer_load(
        kv_indptr_rsrc,
        _raw(ArithValue(seq_i32) + fx.Int32(1)),
        vec_width=1,
        is_scalar=True,
    )
    seq_split_start_i32 = buffer_ops.buffer_load(
        split_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    seq_split_end_i32 = buffer_ops.buffer_load(
        split_indptr_rsrc,
        _raw(ArithValue(seq_i32) + fx.Int32(1)),
        vec_width=1,
        is_scalar=True,
    )

    kv_len_i32 = _raw(ArithValue(kv_end_i32) - ArithValue(kv_start_i32))
    num_tiles_i32 = _raw((ArithValue(kv_len_i32) + fx.Int32(BLOCK_N - 1)).with_signedness(False) // fx.Int32(BLOCK_N))
    configured_splits_i32 = _raw(ArithValue(seq_split_end_i32) - ArithValue(seq_split_start_i32))

    splits_capped_i32 = _select(
        ArithValue(configured_splits_i32) < ArithValue(num_kv_splits),
        configured_splits_i32,
        _raw(num_kv_splits.ir_value()),
    )
    tiles_capped_i32 = _select(
        ArithValue(num_tiles_i32) < ArithValue(splits_capped_i32),
        num_tiles_i32,
        splits_capped_i32,
    )
    # A zero-length CSR row is the virtual-sink-only case.  It still needs one
    # valid partial so the split reducer writes a deterministic all-zero row.
    valid_splits_i32 = _select(
        ArithValue(tiles_capped_i32) < fx.Int32(1),
        fx.Int32(1),
        tiles_capped_i32,
    )
    tiles_per_split_i32 = _raw(
        (ArithValue(num_tiles_i32) + ArithValue(valid_splits_i32) - fx.Int32(1)).with_signedness(False)
        // ArithValue(valid_splits_i32)
    )

    split_token_candidate_i32 = _raw(
        ArithValue(kv_start_i32) + ArithValue(split_i32) * ArithValue(tiles_per_split_i32) * fx.Int32(BLOCK_N)
    )
    split_start_i32 = _select(
        ArithValue(split_token_candidate_i32) < ArithValue(kv_end_i32),
        split_token_candidate_i32,
        kv_end_i32,
    )
    split_end_candidate_i32 = _raw(ArithValue(split_start_i32) + ArithValue(tiles_per_split_i32) * fx.Int32(BLOCK_N))
    split_end_i32 = _select(
        ArithValue(split_end_candidate_i32) < ArithValue(kv_end_i32),
        split_end_candidate_i32,
        kv_end_i32,
    )
    is_valid_split = _raw(ArithValue(split_i32) < ArithValue(valid_splits_i32))
    is_last_valid_split = _raw(ArithValue(split_i32) == ArithValue(valid_splits_i32) - fx.Int32(1))

    is_nope_lane = _raw(ArithValue(lane_in_head_i32) < fx.Int32(14))
    is_rope_lane = _raw(ArithValue(lane_in_head_i32) >= fx.Int32(14))
    rope_chunk_i32 = _select(
        is_rope_lane,
        _raw(ArithValue(lane_in_head_i32) - fx.Int32(14)),
        fx.Int32(0),
    )

    def _decode_nope32(record_i32, packed_rsrc):
        """Load/dequantize this lane's 32 NoPE values and round to BF16."""
        dword_base_i32 = _raw(
            ArithValue(record_i32) * fx.Int32(PACKED_DWORDS)
            + ArithValue(lane_in_head_i32) * fx.Int32(NOPE_DWORDS_PER_LANE)
        )
        words_lo = Vec(
            buffer_ops.buffer_load(
                packed_rsrc,
                dword_base_i32,
                vec_width=4,
                dtype=T.i32,
            ),
            (4,),
            fx.Int32,
        )
        words_hi = Vec(
            buffer_ops.buffer_load(
                packed_rsrc,
                _raw(ArithValue(dword_base_i32) + fx.Int32(4)),
                vec_width=4,
                dtype=T.i32,
            ),
            (4,),
            fx.Int32,
        )
        scale_byte_i8 = buffer_ops.buffer_load(
            packed_rsrc,
            _raw(
                ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                + fx.Int32(NOPE_SCALE_OFFSET)
                + ArithValue(lane_in_head_i32)
            ),
            vec_width=1,
            dtype=T.i8,
        )
        scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))

        values = []
        for word_idx in range_constexpr(8):
            word = words_lo[word_idx] if word_idx < 4 else words_hi[word_idx - 4]
            low = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(word),
                    False,
                ),
                (2,),
                fx.Float32,
            )
            high = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(word),
                    True,
                ),
                (2,),
                fx.Float32,
            )
            decoded = low.shuffle(high, [0, 1, 2, 3])
            # Gen1 consumes dequantized BF16 operands.  Preserve that rounding
            # boundary even though this bootstrap computes the dot with VALU.
            rounded = (decoded * scale_f32).to(fx.BFloat16).to(fx.Float32)
            for elem_idx in range_constexpr(4):
                values.append(_raw(rounded[elem_idx]))
        return values

    def _load_rope32(record_i32, rope_rsrc):
        """Load this lane's 32 BF16 RoPE values and extend to f32."""
        rope_base_i32 = _raw(
            ArithValue(record_i32) * fx.Int32(DIM_ROPE) + ArithValue(rope_chunk_i32) * fx.Int32(DIMS_PER_LANE)
        )
        values = []
        for vec_idx in range_constexpr(DIMS_PER_LANE // 4):
            raw_vec = buffer_ops.buffer_load(
                rope_rsrc,
                _raw(ArithValue(rope_base_i32) + fx.Int32(vec_idx * 4)),
                vec_width=4,
                dtype=T.bf16,
            )
            f32_vec = Vec(raw_vec, (4,), fx.BFloat16).to(fx.Float32)
            for elem_idx in range_constexpr(4):
                values.append(_raw(f32_vec[elem_idx]))
        return values

    q_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
    q_nope_values = _decode_nope32(
        q_record_i32,
        q_packed_rsrc,
    )
    q_rope_values = _load_rope32(
        q_record_i32,
        q_rope_rsrc,
    )
    q_values = [_select(is_nope_lane, q_nope_values[d], q_rope_values[d]) for d in range_constexpr(DIMS_PER_LANE)]

    sink_lane = buffer_ops.buffer_load(
        sink_rsrc,
        head_i32,
        vec_width=1,
        dtype=T.f32,
    )
    row_max_init = _select(
        is_last_valid_split,
        sink_lane,
        c_neg_inf,
    )
    row_sum_init = _select(
        is_last_valid_split,
        c_one,
        c_zero,
    )
    state_init = [row_max_init, row_sum_init] + [_raw(c_zero) for _ in range_constexpr(DIMS_PER_LANE)]

    for logical_pos, state in range(
        _idx(split_start_i32),
        _idx(split_end_i32),
        _idx(1),
        init=state_init,
    ):
        logical_pos_i32 = _i32(logical_pos)
        physical_row_i32 = buffer_ops.buffer_load(
            kv_indices_rsrc,
            logical_pos_i32,
            vec_width=1,
            is_scalar=True,
        )
        kv_nope_values = _decode_nope32(
            physical_row_i32,
            kv_packed_rsrc,
        )
        kv_rope_values = _load_rope32(
            physical_row_i32,
            kv_rope_rsrc,
        )
        kv_values = [
            _select(is_nope_lane, kv_nope_values[d], kv_rope_values[d]) for d in range_constexpr(DIMS_PER_LANE)
        ]

        dot_local = _raw(c_zero)
        for d in range_constexpr(DIMS_PER_LANE):
            product = arith.mulf(
                q_values[d],
                kv_values[d],
                fastmath=fm_fast,
            )
            dot_local = arith.addf(
                dot_local,
                product,
                fastmath=fm_fast,
            )

        score = fx.Float32(dot_local)
        for shuffle_offset in (8, 4, 2, 1):
            score = score + score.shuffle_xor(
                fx.Int32(shuffle_offset),
                fx.Int32(LANES_PER_HEAD),
            )
        score = score * softmax_scale

        row_max_old = fx.Float32(state[0])
        row_sum_old = fx.Float32(state[1])
        row_max_new = fx.Float32(
            arith.maximumf(
                _raw(row_max_old),
                _raw(score),
                fastmath=fm_no_inf,
            )
        )
        old_scale = fx.Float32(
            _fast_exp(
                arith.subf(
                    _raw(row_max_old),
                    _raw(row_max_new),
                    fastmath=fm_no_inf,
                )
            )
        )
        probability = fx.Float32(
            _fast_exp(
                arith.subf(
                    _raw(score),
                    _raw(row_max_new),
                    fastmath=fm_no_inf,
                )
            )
        )
        row_sum_new = row_sum_old * old_scale + probability

        next_state = [_raw(row_max_new), _raw(row_sum_new)]
        for d in range_constexpr(DIMS_PER_LANE):
            out_old = fx.Float32(state[2 + d])
            out_new = out_old * old_scale + probability * fx.Float32(kv_values[d])
            next_state.append(_raw(out_new))
        results = yield next_state

    if is_valid_split:
        row_max_final = fx.Float32(results[0])
        row_sum_final = fx.Float32(results[1])
        inv_sum = fx.Float32(rocdl.rcp(T.f32, _raw(row_sum_final)))
        normalized = [_raw(fx.Float32(results[2 + d]) * inv_sum) for d in range_constexpr(DIMS_PER_LANE)]

        is_direct_output = _raw(ArithValue(_raw(num_kv_splits.ir_value())) == fx.Int32(1))
        output_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
        partial_record_i32 = _raw(
            (ArithValue(q_row_i32) * ArithValue(num_kv_splits) + ArithValue(split_i32)) * fx.Int32(NUM_HEADS)
            + ArithValue(head_i32)
        )

        if is_direct_output:
            for vec_idx in range_constexpr(DIMS_PER_LANE // 4):
                base = vec_idx * 4
                out_vec = Vec.from_elements(
                    normalized[base : base + 4],
                    fx.Float32,
                ).to(fx.BFloat16)
                output_offset_i32 = _raw(
                    ArithValue(output_record_i32) * fx.Int32(V_HEAD_DIM) + ArithValue(dim_base_i32) + fx.Int32(base)
                )
                buffer_ops.buffer_store(
                    out_vec,
                    output_rsrc,
                    output_offset_i32,
                )
        else:
            for vec_idx in range_constexpr(DIMS_PER_LANE // 4):
                base = vec_idx * 4
                out_vec = Vec.from_elements(
                    normalized[base : base + 4],
                    fx.Float32,
                )
                logits_offset_i32 = _raw(
                    ArithValue(partial_record_i32) * fx.Int32(V_HEAD_DIM) + ArithValue(dim_base_i32) + fx.Int32(base)
                )
                buffer_ops.buffer_store(
                    out_vec,
                    logits_rsrc,
                    logits_offset_i32,
                )

            if ArithValue(lane_in_head_i32) == fx.Int32(0):
                log2_sum = fmath.log2(row_sum_final, fastmath=fm_no_inf)
                lse_value = row_max_final + log2_sum * fx.Float32(INV_LOG2E)
                buffer_ops.buffer_store(
                    lse_value,
                    attn_lse_rsrc,
                    partial_record_i32,
                )


def _kn_mla_v4_sparse_decode_mfma_impl(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    block_n: fx.Constexpr[int],
    num_warps: fx.Constexpr[int],
    balanced_splits: fx.Constexpr[bool],
    token_stationary: fx.Constexpr[bool],
    padded_qh64: fx.Constexpr[bool],
    wide_pv_qh64: fx.Constexpr[bool] = False,
    wave_local_qk: fx.Constexpr[bool] = False,
    progressive_vgpr_prefetch: fx.Constexpr[bool] = False,
    progressive_raw_g2l: fx.Constexpr[bool] = False,
    native_fp8_bf16_decode: fx.Constexpr[bool] = False,
    batched_token4_fill: fx.Constexpr[bool] = False,
    lookahead_token4_prefetch: fx.Constexpr[bool] = False,
    qh64_raw_pipeline: fx.Constexpr[bool] = False,
    log2_domain_softmax: fx.Constexpr[bool] = False,
    opaque_payload_ladder: fx.Constexpr[bool] = False,
    kv_lds_stride: fx.Constexpr[int] = DIM_QK,
):
    """Compile-time tiled H16 BF16-MFMA sparse-decode stage 1."""

    assert block_n == num_warps * TOKENS_PER_WAVE or (
        padded_qh64 and block_n == QH64_B32_BLOCK_N and num_warps == QH64_B32_NUM_WARPS
    )
    assert num_warps in (NUM_WARPS, N256_SINGLE_NUM_WARPS)
    assert not token_stationary or (block_n == N256_SINGLE_BLOCK_N and num_warps == N256_SINGLE_NUM_WARPS)
    assert not padded_qh64 or (
        block_n == QH64_B32_BLOCK_N and num_warps == QH64_B32_NUM_WARPS and not balanced_splits and not token_stationary
    )
    assert (
        not wide_pv_qh64
        or padded_qh64
        or (token_stationary and block_n == N256_SINGLE_BLOCK_N and num_warps == N256_SINGLE_NUM_WARPS)
    )
    assert not wave_local_qk or (
        token_stationary and wide_pv_qh64 and block_n == N256_SINGLE_BLOCK_N and num_warps == N256_SINGLE_NUM_WARPS
    )
    assert not progressive_vgpr_prefetch or (
        token_stationary
        and wide_pv_qh64
        and wave_local_qk
        and block_n == N256_SINGLE_BLOCK_N
        and num_warps == N256_SINGLE_NUM_WARPS
    )
    assert not progressive_raw_g2l or (
        token_stationary
        and wide_pv_qh64
        and wave_local_qk
        and not progressive_vgpr_prefetch
        and block_n == N256_SINGLE_BLOCK_N
        and num_warps == N256_SINGLE_NUM_WARPS
    )
    assert not native_fp8_bf16_decode or (
        not progressive_vgpr_prefetch
        and not progressive_raw_g2l
        and (
            (
                token_stationary
                and wide_pv_qh64
                and wave_local_qk
                and block_n == N256_SINGLE_BLOCK_N
                and num_warps == N256_SINGLE_NUM_WARPS
            )
            or (
                padded_qh64
                and wide_pv_qh64
                and not wave_local_qk
                and block_n == QH64_B32_BLOCK_N
                and num_warps == QH64_B32_NUM_WARPS
            )
        )
    )
    # Keep long constexpr implications independent. FlyDSL's boolean AST
    # rewriter duplicates the accumulated LHS of a long `and`, making one
    # combined assertion expand exponentially before MLIR generation.
    assert not batched_token4_fill or native_fp8_bf16_decode
    assert not batched_token4_fill or token_stationary
    assert not batched_token4_fill or wide_pv_qh64
    assert not batched_token4_fill or wave_local_qk
    assert not batched_token4_fill or not padded_qh64
    assert not batched_token4_fill or not progressive_vgpr_prefetch
    assert not batched_token4_fill or not progressive_raw_g2l
    assert not batched_token4_fill or block_n == N256_SINGLE_BLOCK_N
    assert not batched_token4_fill or num_warps == N256_SINGLE_NUM_WARPS
    assert not (batched_token4_fill and lookahead_token4_prefetch)
    assert not lookahead_token4_prefetch or native_fp8_bf16_decode
    assert not lookahead_token4_prefetch or token_stationary
    assert not lookahead_token4_prefetch or wide_pv_qh64
    assert not lookahead_token4_prefetch or wave_local_qk
    assert not lookahead_token4_prefetch or not padded_qh64
    assert not lookahead_token4_prefetch or not progressive_vgpr_prefetch
    assert not lookahead_token4_prefetch or not progressive_raw_g2l
    assert not lookahead_token4_prefetch or block_n == N256_SINGLE_BLOCK_N
    assert not lookahead_token4_prefetch or num_warps == N256_SINGLE_NUM_WARPS
    assert not qh64_raw_pipeline or padded_qh64
    assert not qh64_raw_pipeline or wide_pv_qh64
    assert not qh64_raw_pipeline or native_fp8_bf16_decode
    assert not qh64_raw_pipeline or not token_stationary
    assert not qh64_raw_pipeline or not wave_local_qk
    assert not qh64_raw_pipeline or not progressive_vgpr_prefetch
    assert not qh64_raw_pipeline or not progressive_raw_g2l
    assert not qh64_raw_pipeline or not batched_token4_fill
    assert not qh64_raw_pipeline or not lookahead_token4_prefetch
    assert not qh64_raw_pipeline or block_n == QH64_B32_BLOCK_N
    assert not qh64_raw_pipeline or num_warps == QH64_B32_NUM_WARPS
    assert not log2_domain_softmax or token_stationary
    assert not log2_domain_softmax or wide_pv_qh64
    assert not log2_domain_softmax or wave_local_qk
    assert not log2_domain_softmax or native_fp8_bf16_decode
    assert not log2_domain_softmax or lookahead_token4_prefetch
    assert not log2_domain_softmax or not balanced_splits
    assert not log2_domain_softmax or not padded_qh64
    assert not log2_domain_softmax or not progressive_vgpr_prefetch
    assert not log2_domain_softmax or not progressive_raw_g2l
    assert not log2_domain_softmax or not batched_token4_fill
    assert not log2_domain_softmax or not qh64_raw_pipeline
    assert not log2_domain_softmax or block_n == N256_SINGLE_BLOCK_N
    assert not log2_domain_softmax or num_warps == N256_SINGLE_NUM_WARPS
    # Keep every M15 topology implication linear.  FlyDSL's boolean AST
    # rewriter otherwise duplicates accumulated operands in long expressions.
    assert not opaque_payload_ladder or token_stationary
    assert not opaque_payload_ladder or wide_pv_qh64
    assert not opaque_payload_ladder or wave_local_qk
    assert not opaque_payload_ladder or native_fp8_bf16_decode
    assert not opaque_payload_ladder or lookahead_token4_prefetch
    assert not opaque_payload_ladder or not balanced_splits
    assert not opaque_payload_ladder or not padded_qh64
    assert not opaque_payload_ladder or not progressive_vgpr_prefetch
    assert not opaque_payload_ladder or not progressive_raw_g2l
    assert not opaque_payload_ladder or not batched_token4_fill
    assert not opaque_payload_ladder or not qh64_raw_pipeline
    assert not opaque_payload_ladder or not log2_domain_softmax
    assert not opaque_payload_ladder or block_n == N256_SINGLE_BLOCK_N
    assert not opaque_payload_ladder or num_warps == N256_SINGLE_NUM_WARPS
    # M16 changes one compile-time decoded-KV pitch on the otherwise exact M10
    # topology.  Keep the implications linear so RewriteBoolOps cannot expand
    # a long accumulated boolean expression.
    assert kv_lds_stride in (DIM_QK, M16S536_KV_LDS_STRIDE)
    assert kv_lds_stride == DIM_QK or token_stationary
    assert kv_lds_stride == DIM_QK or wide_pv_qh64
    assert kv_lds_stride == DIM_QK or wave_local_qk
    assert kv_lds_stride == DIM_QK or native_fp8_bf16_decode
    assert kv_lds_stride == DIM_QK or lookahead_token4_prefetch
    assert kv_lds_stride == DIM_QK or not balanced_splits
    assert kv_lds_stride == DIM_QK or not padded_qh64
    assert kv_lds_stride == DIM_QK or not progressive_vgpr_prefetch
    assert kv_lds_stride == DIM_QK or not progressive_raw_g2l
    assert kv_lds_stride == DIM_QK or not batched_token4_fill
    assert kv_lds_stride == DIM_QK or not qh64_raw_pipeline
    assert kv_lds_stride == DIM_QK or not log2_domain_softmax
    assert kv_lds_stride == DIM_QK or not opaque_payload_ladder
    assert kv_lds_stride == DIM_QK or block_n == N256_SINGLE_BLOCK_N
    assert kv_lds_stride == DIM_QK or num_warps == N256_SINGLE_NUM_WARPS
    num_threads = num_warps * WARP_SIZE
    kv_tile_elems = block_n * kv_lds_stride
    p_tile_elems = num_warps * WARP_SIZE * MFMA_OUTPUT_VALUES
    output_dims_per_wave = V_HEAD_DIM if padded_qh64 else V_HEAD_DIM // num_warps
    output_mfma_tiles_per_wave = output_dims_per_wave // 16

    fm_fast = arith.FastMathFlags.fast
    fm_no_inf = (
        arith.FastMathFlags.nnan
        | arith.FastMathFlags.nsz
        | arith.FastMathFlags.arcp
        | arith.FastMathFlags.contract
        | arith.FastMathFlags.afn
        | arith.FastMathFlags.reassoc
    )

    c_zero = fx.Float32(0.0)
    c_neg_inf = fx.Float32(float("-inf"))
    zero_bf16x8 = Vec.filled(MFMA_INPUT_VALUES, 0.0, fx.BFloat16)
    zero_f32x4 = Vec.filled(MFMA_OUTPUT_VALUES, 0.0, fx.Float32)
    softmax_score_scale = softmax_scale
    if const_expr(log2_domain_softmax):
        # M12-L pays the natural-log to log2 conversion once per launch
        # argument, instead of once for every online-softmax exponential.
        softmax_scale_log2 = fx.Float32(
            arith.mulf(
                _raw(softmax_scale),
                _raw(fx.Float32(LOG2E)),
                fastmath=fm_fast,
            )
        )
        softmax_score_scale = softmax_scale_log2

    def _softmax_exp_difference(value):
        """Exponentiate a difference expressed in the selected max domain."""
        if const_expr(log2_domain_softmax):
            return rocdl.exp2(T.f32, value)
        return _fast_exp(value)

    seq_idx = gpu.block_id("x")
    split_idx = gpu.block_id("y")
    tid = gpu.thread_id("x")

    seq_i32 = _i32(seq_idx)
    split_i32 = _i32(split_idx)
    tid_i32 = _i32(tid)
    wave_i32 = _raw(ArithValue(tid_i32) // fx.Int32(WARP_SIZE))
    lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(WARP_SIZE))
    head_i32 = _raw(ArithValue(lane_i32) % fx.Int32(NUM_HEADS))
    lane_group_i32 = _raw(ArithValue(lane_i32) // fx.Int32(NUM_HEADS))

    q_packed_rsrc = buffer_ops.create_buffer_resource(q_packed)
    q_rope_rsrc = buffer_ops.create_buffer_resource(q_rope)
    kv_packed_rsrc = buffer_ops.create_buffer_resource(kv_packed)
    kv_rope_rsrc = buffer_ops.create_buffer_resource(kv_rope)
    qo_indptr_rsrc = buffer_ops.create_buffer_resource(qo_indptr)
    kv_indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr)
    kv_indices_rsrc = buffer_ops.create_buffer_resource(kv_indices)
    sink_rsrc = buffer_ops.create_buffer_resource(sink)
    split_indptr_rsrc = buffer_ops.create_buffer_resource(split_indptr)
    output_rsrc = buffer_ops.create_buffer_resource(output)
    logits_rsrc = buffer_ops.create_buffer_resource(logits)
    attn_lse_rsrc = buffer_ops.create_buffer_resource(attn_lse)

    if const_expr(qh64_raw_pipeline):
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedQH64B32M11).peek()
    elif const_expr(padded_qh64):
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedQH64B32).peek()
    elif const_expr(block_n == BLOCK_N):
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeShared).peek()
    elif const_expr(kv_lds_stride == M16S536_KV_LDS_STRIDE):
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedN128M16S536).peek()
    elif const_expr(progressive_raw_g2l):
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedN128RawG2L).peek()
    else:
        shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedN128).peek()
    kv_lds = shared.kv.view(fx.make_layout(kv_tile_elems, 1))
    if const_expr(not padded_qh64):
        p_lds = shared.probability.view(fx.make_layout(p_tile_elems, 1))
        red_lds = shared.reduction.view(fx.make_layout(2 * num_warps * NUM_HEADS, 1))
        phys_lds = shared.physical_rows.view(fx.make_layout(block_n, 1))

    class Vec8Bf16:
        ir_type = Vec.make_type(MFMA_INPUT_VALUES, fx.BFloat16)

    class Vec4Bf16:
        ir_type = Vec.make_type(MFMA_OUTPUT_VALUES, fx.BFloat16)

    class Vec2Bf16:
        ir_type = Vec.make_type(2, fx.BFloat16)

    class Vec4I32:
        ir_type = Vec.make_type(4, fx.Int32)

    def _native_scaled_fp8_pair(word_i32, scale_f32, word_sel: bool):
        """Decode two scaled OCP-FP8 values directly to packed BF16."""
        word_sel_i1 = arith.constant(
            word_sel,
            type=ir.IntegerType.get_signless(1),
        )
        raw = llvm.call_intrinsic(
            Vec2Bf16.ir_type,
            "llvm.amdgcn.cvt.scalef32.pk.bf16.fp8",
            [
                _raw(word_i32),
                _raw(scale_f32),
                _raw(word_sel_i1),
            ],
            [],
            [],
        )
        return Vec(raw, (2,), fx.BFloat16)

    def _lds_ptr_at(array, byte_offset):
        base = fx.Int64(fx.ptrtoint(array.ptr))
        address = base + fx.Int64(byte_offset)
        return buffer_ops.create_llvm_ptr(address, address_space=3)

    def _m6_raw_buffer_load_lds(rsrc, array, lds_byte_offset_i32, global_byte_offset_i32):
        """Issue one opaque 16-B buffer-to-LDS copy for M6 overlap."""
        lds_address_i64 = fx.Int64(fx.ptrtoint(array.ptr)) + fx.Int64(lds_byte_offset_i32)
        lds_address_uniform_i64 = rocdl.readfirstlane(
            T.i64,
            _raw(lds_address_i64),
        )
        lds_ptr = llvm.inttoptr(
            ir.Type.parse("!llvm.ptr<3>"),
            lds_address_uniform_i64,
        )
        llvm.InlineAsmOp(
            None,
            [
                lds_ptr,
                _raw(global_byte_offset_i32),
                _raw(rsrc),
            ],
            "s_mov_b32 m0, $0\n\ts_nop 0\n\tbuffer_load_dwordx4 $1, $2, 0 offen sc0 lds",
            "s,v,s",
            has_side_effects=True,
        )

    def _m6_raw_lds_load_i32x4(array, byte_offset_i32):
        raw = llvm.LoadOp(
            Vec4I32.ir_type,
            _lds_ptr_at(array, byte_offset_i32),
            alignment=16,
        ).result
        return Vec(raw, (4,), fx.Int32)

    def _m6_raw_lds_load_i8(array, byte_offset_i32):
        return llvm.LoadOp(
            T.i8,
            _lds_ptr_at(array, byte_offset_i32),
            alignment=1,
        ).result

    def _m6_raw_lds_load_bf16x8(array, byte_offset_i32):
        raw = llvm.LoadOp(
            Vec8Bf16.ir_type,
            _lds_ptr_at(array, byte_offset_i32),
            alignment=16,
        ).result
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _lds_store_bf16x8(elem_offset_i32, value):
        llvm.StoreOp(
            _raw(value),
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(elem_offset_i32) * fx.Int32(2)),
            ),
            alignment=16,
        )

    def _lds_load_bf16x8(elem_offset_i32):
        raw = llvm.LoadOp(
            Vec8Bf16.ir_type,
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(elem_offset_i32) * fx.Int32(2)),
            ),
            alignment=16,
        ).result
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _q_lds_store_bf16x8(elem_offset_i32, value):
        llvm.StoreOp(
            _raw(value),
            _lds_ptr_at(
                shared.query,
                _raw(ArithValue(elem_offset_i32) * fx.Int32(2)),
            ),
            alignment=16,
        )

    def _q_lds_load_bf16x8(elem_offset_i32):
        raw = llvm.LoadOp(
            Vec8Bf16.ir_type,
            _lds_ptr_at(
                shared.query,
                _raw(ArithValue(elem_offset_i32) * fx.Int32(2)),
            ),
            alignment=16,
        ).result
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _qh64_ds_read_tr16_bf16x4(elem_offset_i32):
        """Transpose-read four V values from the existing row-major KV tile."""
        raw = rocdl.ds_read_tr16_b64(
            Vec4Bf16.ir_type,
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(elem_offset_i32) * fx.Int32(2)),
            ),
        ).result
        return Vec(raw, (MFMA_OUTPUT_VALUES,), fx.BFloat16)

    def _m11_decoded_store_bf16x8(pong_base_elem_i32, elem_offset_i32, value):
        """Store one decoded fragment into an M11 row-major pong."""
        absolute_elem_i32 = _raw(ArithValue(pong_base_elem_i32) + ArithValue(elem_offset_i32))
        llvm.StoreOp(
            _raw(value),
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(absolute_elem_i32) * fx.Int32(2)),
            ),
            alignment=16,
        )

    def _m11_decoded_load_bf16x8(pong_base_elem_i32, elem_offset_i32):
        """Load one QK fragment from the selected M11 decoded pong."""
        absolute_elem_i32 = _raw(ArithValue(pong_base_elem_i32) + ArithValue(elem_offset_i32))
        raw = llvm.LoadOp(
            Vec8Bf16.ir_type,
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(absolute_elem_i32) * fx.Int32(2)),
            ),
            alignment=16,
        ).result
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _m11_decoded_ds_read_tr16_bf16x4(pong_base_elem_i32, elem_offset_i32):
        """Transpose-read four V values from the selected M11 decoded pong."""
        absolute_elem_i32 = _raw(ArithValue(pong_base_elem_i32) + ArithValue(elem_offset_i32))
        raw = rocdl.ds_read_tr16_b64(
            Vec4Bf16.ir_type,
            _lds_ptr_at(
                shared.kv,
                _raw(ArithValue(absolute_elem_i32) * fx.Int32(2)),
            ),
        ).result
        return Vec(raw, (MFMA_OUTPUT_VALUES,), fx.BFloat16)

    def _decode_nope8(record_i32, dim_i32, packed_rsrc):
        """Decode one naturally aligned eight-value NoPE fragment."""
        dword_offset_i32 = _raw(
            ArithValue(record_i32) * fx.Int32(PACKED_DWORDS) + ArithValue(dim_i32).with_signedness(False) // fx.Int32(4)
        )
        raw_words = buffer_ops.buffer_load(
            packed_rsrc,
            dword_offset_i32,
            vec_width=2,
            dtype=T.i32,
        )
        words = Vec(raw_words, (2,), fx.Int32)
        if const_expr(native_fp8_bf16_decode):
            scale_byte_i8 = buffer_ops.buffer_load(
                packed_rsrc,
                _raw(
                    ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                    + fx.Int32(NOPE_SCALE_OFFSET)
                    + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
                ),
                vec_width=1,
                dtype=T.i8,
            )
            scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
            scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
            decoded_words = []
            for word_idx in range_constexpr(2):
                low = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    False,
                )
                high = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    True,
                )
                decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
            return decoded_words[0].shuffle(
                decoded_words[1],
                list(range(MFMA_INPUT_VALUES)),
            )

        decoded_words = []
        for word_idx in range_constexpr(2):
            low = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    False,
                ),
                (2,),
                fx.Float32,
            )
            high = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    True,
                ),
                (2,),
                fx.Float32,
            )
            decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
        decoded = decoded_words[0].shuffle(
            decoded_words[1],
            list(range(MFMA_INPUT_VALUES)),
        )

        scale_byte_i8 = buffer_ops.buffer_load(
            packed_rsrc,
            _raw(
                ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                + fx.Int32(NOPE_SCALE_OFFSET)
                + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
            ),
            vec_width=1,
            dtype=T.i8,
        )
        scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
        return (decoded * scale_f32).to(fx.BFloat16)

    def _decode_nope16(record_i32, dim_i32, packed_rsrc):
        """Decode two adjacent vec8 fragments with one supported dwordx4."""
        dword_offset_i32 = _raw(
            ArithValue(record_i32) * fx.Int32(PACKED_DWORDS) + ArithValue(dim_i32).with_signedness(False) // fx.Int32(4)
        )
        raw_words = buffer_ops.buffer_load(
            packed_rsrc,
            dword_offset_i32,
            vec_width=4,
            dtype=T.i32,
        )
        words = Vec(raw_words, (4,), fx.Int32)
        if const_expr(native_fp8_bf16_decode):
            scale_byte_i8 = buffer_ops.buffer_load(
                packed_rsrc,
                _raw(
                    ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                    + fx.Int32(NOPE_SCALE_OFFSET)
                    + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
                ),
                vec_width=1,
                dtype=T.i8,
            )
            scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
            scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
            decoded_words = []
            for word_idx in range_constexpr(4):
                low = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    False,
                )
                high = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    True,
                )
                decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
            return (
                decoded_words[0].shuffle(
                    decoded_words[1],
                    list(range(MFMA_INPUT_VALUES)),
                ),
                decoded_words[2].shuffle(
                    decoded_words[3],
                    list(range(MFMA_INPUT_VALUES)),
                ),
            )

        decoded_words = []
        for word_idx in range_constexpr(4):
            low = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    False,
                ),
                (2,),
                fx.Float32,
            )
            high = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    True,
                ),
                (2,),
                fx.Float32,
            )
            decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))

        scale_byte_i8 = buffer_ops.buffer_load(
            packed_rsrc,
            _raw(
                ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                + fx.Int32(NOPE_SCALE_OFFSET)
                + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
            ),
            vec_width=1,
            dtype=T.i8,
        )
        scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
        first = decoded_words[0].shuffle(
            decoded_words[1],
            list(range(MFMA_INPUT_VALUES)),
        )
        second = decoded_words[2].shuffle(
            decoded_words[3],
            list(range(MFMA_INPUT_VALUES)),
        )
        return (
            (first * scale_f32).to(fx.BFloat16),
            (second * scale_f32).to(fx.BFloat16),
        )

    def _decode_nope16_prefetched(
        words,
        scale_byte_i8,
        scale_is_i32: fx.Constexpr[bool] = False,
    ):
        """Decode one M6 NoPE dwordx4 already resident in VGPRs."""
        if const_expr(native_fp8_bf16_decode):
            if const_expr(scale_is_i32):
                scale_i32 = arith.andi(
                    _raw(scale_byte_i8),
                    _raw(fx.Int32(0xFF)),
                )
            else:
                scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
            scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
            decoded_words = []
            for word_idx in range_constexpr(4):
                low = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    False,
                )
                high = _native_scaled_fp8_pair(
                    words[word_idx],
                    scale_f32,
                    True,
                )
                decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
            return (
                decoded_words[0].shuffle(
                    decoded_words[1],
                    list(range(MFMA_INPUT_VALUES)),
                ),
                decoded_words[2].shuffle(
                    decoded_words[3],
                    list(range(MFMA_INPUT_VALUES)),
                ),
            )

        decoded_words = []
        for word_idx in range_constexpr(4):
            low = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    False,
                ),
                (2,),
                fx.Float32,
            )
            high = Vec(
                rocdl.cvt_pk_f32_fp8(
                    Vec.make_type(2, fx.Float32),
                    _raw(words[word_idx]),
                    True,
                ),
                (2,),
                fx.Float32,
            )
            decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))

        if const_expr(scale_is_i32):
            scale_i32 = arith.andi(
                _raw(scale_byte_i8),
                _raw(fx.Int32(0xFF)),
            )
        else:
            scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
        first = decoded_words[0].shuffle(
            decoded_words[1],
            list(range(MFMA_INPUT_VALUES)),
        )
        second = decoded_words[2].shuffle(
            decoded_words[3],
            list(range(MFMA_INPUT_VALUES)),
        )
        return (
            (first * scale_f32).to(fx.BFloat16),
            (second * scale_f32).to(fx.BFloat16),
        )

    def _load_rope8(record_i32, dim_i32, rope_rsrc):
        raw = buffer_ops.buffer_load(
            rope_rsrc,
            _raw(ArithValue(record_i32) * fx.Int32(DIM_ROPE) + ArithValue(dim_i32)),
            vec_width=MFMA_INPUT_VALUES,
            dtype=T.bf16,
        )
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _m15_opaque_load_i32x4(rsrc, byte_offset_i32):
        """Issue one ordered 16-byte VMEM request whose wait is manual."""
        raw = llvm.inline_asm(
            Vec4I32.ir_type,
            [_raw(byte_offset_i32), _raw(rsrc)],
            "buffer_load_dwordx4 $0, $1, $2, 0 offen",
            "=v,v,s,~{memory}",
            has_side_effects=True,
        )
        return Vec(raw, (4,), fx.Int32)

    def _m15_opaque_load_scale_i32(rsrc, byte_offset_i32):
        """Issue one ordered scale request with distinct VDATA and VADDR.

        The four-byte read stays within the 512-byte packed record for every
        E8M0 offset.  Its low byte is selected only after the matching VMEM
        wait, avoiding the tied byte-load destination used by rejected M15.
        """
        return llvm.inline_asm(
            T.i32,
            [_raw(byte_offset_i32), _raw(rsrc)],
            "buffer_load_dword $0, $1, $2, 0 offen",
            "=&v,v,s,~{memory}",
            has_side_effects=True,
        )

    def _m15_wait_vmcnt(count):
        """Retire only the oldest requests in M15's exact VMEM ledger."""
        llvm.inline_asm(
            None,
            [],
            f"s_waitcnt vmcnt({count})",
            "~{memory}",
            has_side_effects=True,
        )

    def _mask_bf16x8(value, valid):
        selected = arith.select(valid, _raw(value), _raw(zero_bf16x8))
        return Vec(selected, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _mfma_bf16(a, b, accumulator):
        raw = rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(MFMA_OUTPUT_VALUES, fx.Float32),
            [_raw(a), _raw(b), _raw(accumulator), 0, 0, 0],
        )
        rocdl.sched_mfma(1)
        return Vec(raw, (MFMA_OUTPUT_VALUES,), fx.Float32)

    q_row_i32 = buffer_ops.buffer_load(
        qo_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    kv_start_i32 = buffer_ops.buffer_load(
        kv_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    kv_end_i32 = buffer_ops.buffer_load(
        kv_indptr_rsrc,
        _raw(ArithValue(seq_i32) + fx.Int32(1)),
        vec_width=1,
        is_scalar=True,
    )
    seq_split_start_i32 = buffer_ops.buffer_load(
        split_indptr_rsrc,
        seq_i32,
        vec_width=1,
        is_scalar=True,
    )
    seq_split_end_i32 = buffer_ops.buffer_load(
        split_indptr_rsrc,
        _raw(ArithValue(seq_i32) + fx.Int32(1)),
        vec_width=1,
        is_scalar=True,
    )

    kv_len_i32 = _raw(ArithValue(kv_end_i32) - ArithValue(kv_start_i32))
    num_tiles_i32 = _raw((ArithValue(kv_len_i32) + fx.Int32(block_n - 1)).with_signedness(False) // fx.Int32(block_n))
    configured_splits_i32 = _raw(ArithValue(seq_split_end_i32) - ArithValue(seq_split_start_i32))
    splits_capped_i32 = _select(
        ArithValue(configured_splits_i32) < ArithValue(num_kv_splits),
        configured_splits_i32,
        _raw(num_kv_splits.ir_value()),
    )
    tiles_capped_i32 = _select(
        ArithValue(num_tiles_i32) < ArithValue(splits_capped_i32),
        num_tiles_i32,
        splits_capped_i32,
    )
    valid_splits_i32 = _select(
        ArithValue(tiles_capped_i32) < fx.Int32(1),
        fx.Int32(1),
        tiles_capped_i32,
    )
    if const_expr(balanced_splits):
        # Distribute the remainder across the earliest splits instead of
        # leaving the final split empty.  For N64/L1152 this maps nine B128
        # tiles as 3/2/2/2 while retaining all 256 grid workgroups.
        base_tiles_i32 = _raw(ArithValue(num_tiles_i32).with_signedness(False) // ArithValue(valid_splits_i32))
        extra_tiles_i32 = _raw(ArithValue(num_tiles_i32) % ArithValue(valid_splits_i32))
        extra_before_i32 = _select(
            ArithValue(split_i32) < ArithValue(extra_tiles_i32),
            split_i32,
            extra_tiles_i32,
        )
        split_tile_start_i32 = _raw(ArithValue(split_i32) * ArithValue(base_tiles_i32) + ArithValue(extra_before_i32))
        split_tile_count_i32 = _raw(
            ArithValue(base_tiles_i32)
            + ArithValue(
                _select(
                    ArithValue(split_i32) < ArithValue(extra_tiles_i32),
                    fx.Int32(1),
                    fx.Int32(0),
                )
            )
        )
        split_candidate_i32 = _raw(ArithValue(kv_start_i32) + ArithValue(split_tile_start_i32) * fx.Int32(block_n))
        split_end_candidate_i32 = _raw(
            ArithValue(split_candidate_i32) + ArithValue(split_tile_count_i32) * fx.Int32(block_n)
        )
    else:
        tiles_per_split_i32 = _raw(
            (ArithValue(num_tiles_i32) + ArithValue(valid_splits_i32) - fx.Int32(1)).with_signedness(False)
            // ArithValue(valid_splits_i32)
        )
        split_candidate_i32 = _raw(
            ArithValue(kv_start_i32) + ArithValue(split_i32) * ArithValue(tiles_per_split_i32) * fx.Int32(block_n)
        )
        split_end_candidate_i32 = _raw(
            ArithValue(split_candidate_i32) + ArithValue(tiles_per_split_i32) * fx.Int32(block_n)
        )
    split_start_i32 = _select(
        ArithValue(split_candidate_i32) < ArithValue(kv_end_i32),
        split_candidate_i32,
        kv_end_i32,
    )
    split_end_i32 = _select(
        ArithValue(split_end_candidate_i32) < ArithValue(kv_end_i32),
        split_end_candidate_i32,
        kv_end_i32,
    )
    is_valid_split = _raw(ArithValue(split_i32) < ArithValue(valid_splits_i32))
    is_last_valid_split = _raw(ArithValue(split_i32) == ArithValue(valid_splits_i32) - fx.Int32(1))

    # The B operand is Q because QK is evaluated as K @ Q^T.  In the
    # 16x16x32 MFMA result, lane%16 is the head and each lane owns four
    # token scores.
    q_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
    real_query = _raw(ArithValue(wave_i32) == fx.Int32(0))
    q_fragments = []
    if const_expr(padded_qh64):
        # Cooperatively materialize all 16 real H16 query rows once.  Sixteen
        # threads per head each own one aligned vec8 in four 128-value bands.
        # The QK loop reloads one vec8 per k-step, shortening Q live ranges by
        # 64 VGPR while both B32 token halves reuse the same fragment.
        qh64_q_group_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(NUM_HEADS))
        for q_load_iter in range_constexpr(3):
            qh64_q_dim_i32 = _raw(
                ArithValue(qh64_q_group_i32) * fx.Int32(MFMA_INPUT_VALUES) + fx.Int32(q_load_iter * 128)
            )
            qh64_q_fragment = _decode_nope8(
                q_record_i32,
                qh64_q_dim_i32,
                q_packed_rsrc,
            )
            qh64_q_lds_offset_i32 = _raw(ArithValue(head_i32) * fx.Int32(DIM_QK) + ArithValue(qh64_q_dim_i32))
            _q_lds_store_bf16x8(
                qh64_q_lds_offset_i32,
                qh64_q_fragment,
            )

        qh64_q_tail_dim_i32 = _raw(fx.Int32(384) + ArithValue(qh64_q_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
        qh64_q_tail_lds_offset_i32 = _raw(ArithValue(head_i32) * fx.Int32(DIM_QK) + ArithValue(qh64_q_tail_dim_i32))
        if ArithValue(qh64_q_group_i32) < fx.Int32(8):
            _q_lds_store_bf16x8(
                qh64_q_tail_lds_offset_i32,
                _decode_nope8(
                    q_record_i32,
                    qh64_q_tail_dim_i32,
                    q_packed_rsrc,
                ),
            )
        else:
            _q_lds_store_bf16x8(
                qh64_q_tail_lds_offset_i32,
                _load_rope8(
                    q_record_i32,
                    _raw(ArithValue(qh64_q_tail_dim_i32) - fx.Int32(DIM_NOPE)),
                    q_rope_rsrc,
                ),
            )
        gpu.barrier()
    else:
        for k_step in range_constexpr(DIM_QK // MFMA_K):
            dim_i32 = _raw(ArithValue(lane_group_i32) * fx.Int32(MFMA_INPUT_VALUES) + fx.Int32(k_step * MFMA_K))
            if k_step < DIM_NOPE // MFMA_K:
                q_fragments.append(
                    _decode_nope8(
                        q_record_i32,
                        dim_i32,
                        q_packed_rsrc,
                    )
                )
            else:
                q_fragments.append(
                    _load_rope8(
                        q_record_i32,
                        _raw(ArithValue(dim_i32) - fx.Int32(DIM_NOPE)),
                        q_rope_rsrc,
                    )
                )

    sink_loaded = buffer_ops.buffer_load(
        sink_rsrc,
        head_i32,
        vec_width=1,
        dtype=T.f32,
    )
    sink_lane = _select(real_query, sink_loaded, c_zero) if padded_qh64 else sink_loaded

    def _qh64_resolve_tile(tile_i32):
        """Resolve one B32 tile with one CSR lookup per eight-lane group."""
        token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(8))
        fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(8))
        logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_i32))
        valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
        # split_start is a valid row for every shape admitted by the focused
        # diagnostic.  It also keeps speculative next-tile resolution in
        # bounds on the final iteration.
        safe_logical_i32 = _select(
            valid,
            logical_i32,
            split_start_i32,
        )
        physical_lane_i32 = fx.Int32(0).ir_value()
        if ArithValue(fragment_lane_i32) == fx.Int32(0):
            physical_lane_i32 = buffer_ops.buffer_load(
                kv_indices_rsrc,
                safe_logical_i32,
                vec_width=1,
                dtype=T.i32,
            )
        leader_lane_i32 = _raw(ArithValue(lane_i32) - ArithValue(fragment_lane_i32))
        physical_i32 = rocdl.ds_bpermute(
            T.i32,
            _raw(ArithValue(leader_lane_i32) * fx.Int32(4)),
            _raw(physical_lane_i32),
        )
        return (
            token_i32,
            fragment_lane_i32,
            valid,
            _raw(physical_i32),
        )

    def _qh64_fill_tile_eager(tile_i32):
        """Decode one B32 tile into the shared row-major K/V slab."""
        (
            token_i32,
            fragment_lane_i32,
            valid,
            physical_i32,
        ) = _qh64_resolve_tile(tile_i32)

        for load_iter in range_constexpr(4):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 8))
            if ArithValue(dim_group_i32) < fx.Int32(DIM_NOPE // (2 * MFMA_INPUT_VALUES)):
                dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                decoded_fragments = _decode_nope16(
                    physical_i32,
                    dim_i32,
                    kv_packed_rsrc,
                )
                for fragment_idx in range_constexpr(2):
                    decoded = _mask_bf16x8(
                        decoded_fragments[fragment_idx],
                        valid,
                    )
                    lds_offset_i32 = _raw(
                        ArithValue(token_i32) * fx.Int32(DIM_QK)
                        + ArithValue(dim_i32)
                        + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                    )
                    _lds_store_bf16x8(lds_offset_i32, decoded)

        rope_dim_i32 = _raw(ArithValue(fragment_lane_i32) * fx.Int32(MFMA_INPUT_VALUES))
        rope = _mask_bf16x8(
            _load_rope8(
                physical_i32,
                rope_dim_i32,
                kv_rope_rsrc,
            ),
            valid,
        )
        rope_lds_offset_i32 = _raw(
            ArithValue(token_i32) * fx.Int32(DIM_QK) + fx.Int32(DIM_NOPE) + ArithValue(rope_dim_i32)
        )
        _lds_store_bf16x8(rope_lds_offset_i32, rope)

    def _m11_issue_raw_tile(tile_i32, bank_base_bytes_i32):
        """Issue one B32 tile into a wave-owned chunk-major raw bank."""
        (
            _token_i32,
            fragment_lane_i32,
            _valid,
            physical_i32,
        ) = _qh64_resolve_tile(tile_i32)

        # Each wave owns eight rows.  One dwordx4 per lane covers a 128-byte
        # packed chunk, while a wave-uniform m0 base gives every lane a
        # distinct 16-byte destination.
        for raw_chunk in range_constexpr(4):
            _m6_raw_buffer_load_lds(
                kv_packed_rsrc,
                shared.raw,
                _raw(
                    ArithValue(bank_base_bytes_i32) + fx.Int32(raw_chunk * 4096) + ArithValue(wave_i32) * fx.Int32(1024)
                ),
                _raw(
                    ArithValue(physical_i32) * fx.Int32(DIM_PACKED)
                    + fx.Int32(raw_chunk * 128)
                    + ArithValue(fragment_lane_i32) * fx.Int32(16)
                ),
            )

        _m6_raw_buffer_load_lds(
            kv_rope_rsrc,
            shared.raw,
            _raw(
                ArithValue(bank_base_bytes_i32)
                + fx.Int32(QH64_M11_RAW_PACKED_BYTES)
                + ArithValue(wave_i32) * fx.Int32(1024)
            ),
            _raw(ArithValue(physical_i32) * fx.Int32(DIM_ROPE * 2) + ArithValue(fragment_lane_i32) * fx.Int32(16)),
        )

    def _m11_decode_raw_tile(tile_i32, bank_base_bytes_i32, pong_base_elem_i32):
        """Decode one wave-owned raw bank into a cross-wave row-major pong."""
        token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(8))
        fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(8))
        valid = _raw(ArithValue(tile_i32) + ArithValue(token_i32) < ArithValue(split_end_i32))

        for raw_chunk in range_constexpr(4):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(raw_chunk * 8))
            if ArithValue(dim_group_i32) < fx.Int32(DIM_NOPE // (2 * MFMA_INPUT_VALUES)):
                dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                raw_words = _m6_raw_lds_load_i32x4(
                    shared.raw,
                    _raw(
                        ArithValue(bank_base_bytes_i32)
                        + fx.Int32(raw_chunk * 4096)
                        + ArithValue(tid_i32) * fx.Int32(16)
                    ),
                )
                scale_byte_i8 = _m6_raw_lds_load_i8(
                    shared.raw,
                    _raw(
                        ArithValue(bank_base_bytes_i32)
                        + fx.Int32(3 * 4096)
                        + ArithValue(token_i32) * fx.Int32(128)
                        + fx.Int32(64)
                        + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
                    ),
                )
                decoded_fragments = _decode_nope16_prefetched(
                    raw_words,
                    scale_byte_i8,
                )
                for fragment_idx in range_constexpr(2):
                    decoded = _mask_bf16x8(
                        decoded_fragments[fragment_idx],
                        valid,
                    )
                    decoded_offset_i32 = _raw(
                        ArithValue(token_i32) * fx.Int32(DIM_QK)
                        + ArithValue(dim_i32)
                        + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                    )
                    _m11_decoded_store_bf16x8(
                        pong_base_elem_i32,
                        decoded_offset_i32,
                        decoded,
                    )

        rope = _m6_raw_lds_load_bf16x8(
            shared.raw,
            _raw(
                ArithValue(bank_base_bytes_i32)
                + fx.Int32(QH64_M11_RAW_PACKED_BYTES)
                + ArithValue(tid_i32) * fx.Int32(16)
            ),
        )
        rope = _mask_bf16x8(
            rope,
            valid,
        )
        rope_offset_i32 = _raw(
            ArithValue(token_i32) * fx.Int32(DIM_QK)
            + fx.Int32(DIM_NOPE)
            + ArithValue(fragment_lane_i32) * fx.Int32(MFMA_INPUT_VALUES)
        )
        _m11_decoded_store_bf16x8(
            pong_base_elem_i32,
            rope_offset_i32,
            rope,
        )

    def _m6_fill_initial_tile(tile_i32):
        """Materialize M6's first B128 tile with the accepted token4 map."""
        token4_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
        token4_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
        token4_logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token4_token_i32))
        token4_valid = _raw(ArithValue(token4_logical_i32) < ArithValue(split_end_i32))
        token4_safe_logical_i32 = _select(
            token4_valid,
            token4_logical_i32,
            tile_i32,
        )
        token4_physical_lane_i32 = fx.Int32(0).ir_value()
        if ArithValue(token4_fragment_lane_i32) == fx.Int32(0):
            token4_physical_lane_i32 = buffer_ops.buffer_load(
                kv_indices_rsrc,
                token4_safe_logical_i32,
                vec_width=1,
                dtype=T.i32,
            )
        token4_physical_i32 = dpp_utils.update_dpp_i32(
            token4_physical_lane_i32,
            token4_physical_lane_i32,
            0,
        )

        for load_iter in range_constexpr(7):
            token4_dim_group_i32 = _raw(ArithValue(token4_fragment_lane_i32) + fx.Int32(load_iter * 4))
            token4_dim_i32 = _raw(ArithValue(token4_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
            token4_decoded_fragments = _decode_nope16(
                _raw(token4_physical_i32),
                token4_dim_i32,
                kv_packed_rsrc,
            )
            for fragment_idx in range_constexpr(2):
                token4_decoded = _mask_bf16x8(
                    token4_decoded_fragments[fragment_idx],
                    token4_valid,
                )
                token4_lds_offset_i32 = _raw(
                    ArithValue(token4_token_i32) * fx.Int32(DIM_QK)
                    + ArithValue(token4_dim_i32)
                    + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                )
                _lds_store_bf16x8(
                    token4_lds_offset_i32,
                    token4_decoded,
                )

        for load_iter in range_constexpr(2):
            token4_dim_group_i32 = _raw(ArithValue(token4_fragment_lane_i32) + fx.Int32(load_iter * 4))
            token4_rope_dim_i32 = _raw(ArithValue(token4_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
            token4_decoded = _load_rope8(
                _raw(token4_physical_i32),
                token4_rope_dim_i32,
                kv_rope_rsrc,
            )
            token4_decoded = _mask_bf16x8(
                token4_decoded,
                token4_valid,
            )
            token4_lds_offset_i32 = _raw(
                ArithValue(token4_token_i32) * fx.Int32(DIM_QK) + fx.Int32(DIM_NOPE) + ArithValue(token4_rope_dim_i32)
            )
            _lds_store_bf16x8(
                token4_lds_offset_i32,
                token4_decoded,
            )

    def _m9_fill_batched_tile(tile_i32):
        """Issue the complete token4 tile before native decode and LDS store."""
        token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
        fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
        logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_i32))
        valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
        safe_logical_i32 = _select(
            valid,
            logical_i32,
            tile_i32,
        )
        physical_lane_i32 = fx.Int32(0).ir_value()
        if ArithValue(fragment_lane_i32) == fx.Int32(0):
            physical_lane_i32 = buffer_ops.buffer_load(
                kv_indices_rsrc,
                safe_logical_i32,
                vec_width=1,
                dtype=T.i32,
            )
        physical_i32 = dpp_utils.update_dpp_i32(
            physical_lane_i32,
            physical_lane_i32,
            0,
        )

        packed_groups = []
        scale_bytes = []
        for load_iter in range_constexpr(7):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
            dword_offset_i32 = _raw(
                ArithValue(physical_i32) * fx.Int32(PACKED_DWORDS)
                + ArithValue(dim_i32).with_signedness(False) // fx.Int32(4)
            )
            packed_groups.append(
                Vec(
                    buffer_ops.buffer_load(
                        kv_packed_rsrc,
                        dword_offset_i32,
                        vec_width=4,
                        dtype=T.i32,
                    ),
                    (4,),
                    fx.Int32,
                )
            )
            scale_bytes.append(
                buffer_ops.buffer_load(
                    kv_packed_rsrc,
                    _raw(
                        ArithValue(physical_i32) * fx.Int32(DIM_PACKED)
                        + fx.Int32(NOPE_SCALE_OFFSET)
                        + (ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
                    ),
                    vec_width=1,
                    dtype=T.i8,
                )
            )

        rope_groups = []
        for load_iter in range_constexpr(2):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            rope_dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
            rope_groups.append(
                _load_rope8(
                    _raw(physical_i32),
                    rope_dim_i32,
                    kv_rope_rsrc,
                )
            )

        for load_iter in range_constexpr(7):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
            decoded_fragments = _decode_nope16_prefetched(
                packed_groups[load_iter],
                scale_bytes[load_iter],
            )
            for fragment_idx in range_constexpr(2):
                decoded = _mask_bf16x8(
                    decoded_fragments[fragment_idx],
                    valid,
                )
                lds_offset_i32 = _raw(
                    ArithValue(token_i32) * fx.Int32(kv_lds_stride)
                    + ArithValue(dim_i32)
                    + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                )
                _lds_store_bf16x8(
                    lds_offset_i32,
                    decoded,
                )

        for load_iter in range_constexpr(2):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            rope_dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
            lds_offset_i32 = _raw(
                ArithValue(token_i32) * fx.Int32(kv_lds_stride) + fx.Int32(DIM_NOPE) + ArithValue(rope_dim_i32)
            )
            _lds_store_bf16x8(
                lds_offset_i32,
                _mask_bf16x8(
                    rope_groups[load_iter],
                    valid,
                ),
            )

    if const_expr(qh64_raw_pipeline):
        # Prime four natural B32 raw banks.  split_start is the sequence-local
        # absolute CSR offset; literal logical zero would read another row.
        for prologue_tile in range_constexpr(QH64_M11_RAW_BANKS):
            _m11_issue_raw_tile(
                _raw(ArithValue(split_start_i32) + fx.Int32(prologue_tile * QH64_B32_BLOCK_N)),
                _raw(fx.Int32(prologue_tile * QH64_M11_RAW_BANK_BYTES)),
            )
        llvm.inline_asm(
            None,
            [],
            "s_waitcnt vmcnt(0)",
            "~{memory}",
            has_side_effects=True,
        )
        _m11_decode_raw_tile(
            split_start_i32,
            _raw(fx.Int32(0)),
            _raw(fx.Int32(0)),
        )
        llvm.inline_asm(
            None,
            [],
            "s_waitcnt lgkmcnt(0)",
            "~{memory}",
            has_side_effects=True,
        )
        gpu.barrier()

    # Keep the real-token softmax independent of the virtual sink.  Folding
    # sink into the initial state makes every BF16 P value on the last split
    # inherit sink's scale and is especially inaccurate after split merging.
    row_max_init = _raw(c_neg_inf)
    row_sum_init = _raw(c_zero)
    state_init = [
        row_max_init,
        row_sum_init,
    ] + [_raw(zero_f32x4) for _ in range_constexpr(output_mfma_tiles_per_wave)]

    for tile_pos, state in range(
        _idx(split_start_i32),
        _idx(split_end_i32),
        _idx(block_n),
        init=state_init,
    ):
        tile_i32 = _i32(tile_pos)
        m11_tile_ordinal_i32 = fx.Int32(0).ir_value()
        m11_current_pong_base_i32 = fx.Int32(0).ir_value()
        if const_expr(qh64_raw_pipeline):
            m11_tile_ordinal_i32 = _raw(
                (ArithValue(tile_i32) - ArithValue(split_start_i32)).with_signedness(False)
                // fx.Int32(QH64_B32_BLOCK_N)
            )
            m11_current_bank_i32 = _raw(ArithValue(m11_tile_ordinal_i32) % fx.Int32(QH64_M11_RAW_BANKS))
            m11_current_pong_i32 = _raw(ArithValue(m11_tile_ordinal_i32) % fx.Int32(QH64_M11_DECODED_PONGS))
            m11_current_pong_base_i32 = _raw(ArithValue(m11_current_pong_i32) * fx.Int32(QH64_B32_KV_TILE_ELEMS))
            m11_has_future_tile = _raw(
                ArithValue(tile_i32) + fx.Int32(QH64_M11_RAW_BANKS * QH64_B32_BLOCK_N) < ArithValue(split_end_i32)
            )
            if m11_has_future_tile:
                _m11_issue_raw_tile(
                    _raw(ArithValue(tile_i32) + fx.Int32(QH64_M11_RAW_BANKS * QH64_B32_BLOCK_N)),
                    _raw(ArithValue(m11_current_bank_i32) * fx.Int32(QH64_M11_RAW_BANK_BYTES)),
                )

        if const_expr(lookahead_token4_prefetch):
            # M10 eagerly fills only the first tile.  Every later tile was
            # prefetched during the preceding tile's PV chain and published
            # after its end-of-tile safety barrier.
            if ArithValue(tile_i32) == ArithValue(split_start_i32):
                _m9_fill_batched_tile(tile_i32)
        elif const_expr(batched_token4_fill):
            _m9_fill_batched_tile(tile_i32)
        elif const_expr(progressive_vgpr_prefetch):
            # Later M6 tiles are populated pair-by-pair during the preceding
            # iteration.  Only the first tile uses the accepted token4 fill.
            if ArithValue(tile_i32) == ArithValue(split_start_i32):
                _m6_fill_initial_tile(tile_i32)
        elif const_expr(progressive_raw_g2l):
            if ArithValue(tile_i32) == ArithValue(split_start_i32):
                _m6_fill_initial_tile(tile_i32)
        elif const_expr(padded_qh64):
            if const_expr(not qh64_raw_pipeline):
                _qh64_fill_tile_eager(tile_i32)
        elif const_expr(token_stationary):
            # Four adjacent lanes own one token.  The quad leader resolves its
            # CSR row, then a convergent DPP quad broadcast keeps the physical
            # row in a VGPR for all seven NoPE and two RoPE fragments.
            token4_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
            token4_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
            token4_logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token4_token_i32))
            token4_valid = _raw(ArithValue(token4_logical_i32) < ArithValue(split_end_i32))
            token4_safe_logical_i32 = _select(
                token4_valid,
                token4_logical_i32,
                tile_i32,
            )
            token4_physical_lane_i32 = fx.Int32(0).ir_value()
            if ArithValue(token4_fragment_lane_i32) == fx.Int32(0):
                token4_physical_lane_i32 = buffer_ops.buffer_load(
                    kv_indices_rsrc,
                    token4_safe_logical_i32,
                    vec_width=1,
                    dtype=T.i32,
                )
            token4_physical_i32 = dpp_utils.update_dpp_i32(
                token4_physical_lane_i32,
                token4_physical_lane_i32,
                0,
            )

            # Each quad partitions all 28 NoPE groups into seven contiguous
            # four-fragment rounds without changing decode or LDS layout.
            for load_iter in range_constexpr(7):
                token4_dim_group_i32 = _raw(ArithValue(token4_fragment_lane_i32) + fx.Int32(load_iter * 4))
                token4_dim_i32 = _raw(ArithValue(token4_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                token4_decoded_fragments = _decode_nope16(
                    _raw(token4_physical_i32),
                    token4_dim_i32,
                    kv_packed_rsrc,
                )
                for fragment_idx in range_constexpr(2):
                    token4_decoded = _mask_bf16x8(
                        token4_decoded_fragments[fragment_idx],
                        token4_valid,
                    )
                    token4_lds_offset_i32 = _raw(
                        ArithValue(token4_token_i32) * fx.Int32(DIM_QK)
                        + ArithValue(token4_dim_i32)
                        + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                    )
                    _lds_store_bf16x8(
                        token4_lds_offset_i32,
                        token4_decoded,
                    )

            # The same quad mapping covers all eight RoPE vec8 groups in two
            # contiguous four-fragment rounds.
            for load_iter in range_constexpr(2):
                token4_dim_group_i32 = _raw(ArithValue(token4_fragment_lane_i32) + fx.Int32(load_iter * 4))
                token4_rope_dim_i32 = _raw(ArithValue(token4_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                token4_decoded = _load_rope8(
                    _raw(token4_physical_i32),
                    token4_rope_dim_i32,
                    kv_rope_rsrc,
                )
                token4_decoded = _mask_bf16x8(
                    token4_decoded,
                    token4_valid,
                )
                token4_lds_offset_i32 = _raw(
                    ArithValue(token4_token_i32) * fx.Int32(DIM_QK)
                    + fx.Int32(DIM_NOPE)
                    + ArithValue(token4_rope_dim_i32)
                )
                _lds_store_bf16x8(
                    token4_lds_offset_i32,
                    token4_decoded,
                )
        else:
            # Resolve the flat page-size-one CSR once per logical token.
            # Invalid tail rows alias the first valid row and are zeroed before
            # LDS store.
            if ArithValue(tid_i32) < fx.Int32(block_n):
                logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(tid_i32))
                token_valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
                safe_logical_i32 = _select(
                    token_valid,
                    logical_i32,
                    tile_i32,
                )
                physical_i32 = buffer_ops.buffer_load(
                    kv_indices_rsrc,
                    safe_logical_i32,
                    vec_width=1,
                    dtype=T.i32,
                )
                fx.memref_store(physical_i32, phys_lds, tid_i32)
            gpu.barrier()

            # 448 NoPE bytes = 28 aligned 16-value groups per token.  Matching
            # four/eight waves with 64/128 rows keeps this at exactly seven
            # dwordx4 loads per workitem.  Each load feeds two vec8 LDS stores.
            for load_iter in range_constexpr(7):
                group_i32 = _raw(ArithValue(tid_i32) + fx.Int32(load_iter * num_threads))
                token_i32 = _raw(
                    ArithValue(group_i32).with_signedness(False) // fx.Int32(DIM_NOPE // (2 * MFMA_INPUT_VALUES))
                )
                dim_group_i32 = _raw(ArithValue(group_i32) % fx.Int32(DIM_NOPE // (2 * MFMA_INPUT_VALUES)))
                dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                physical_i32 = fx.memref_load(phys_lds, token_i32)
                decoded_fragments = _decode_nope16(
                    _raw(physical_i32),
                    dim_i32,
                    kv_packed_rsrc,
                )
                token_valid = _raw(ArithValue(tile_i32) + ArithValue(token_i32) < ArithValue(split_end_i32))
                for fragment_idx in range_constexpr(2):
                    decoded = _mask_bf16x8(
                        decoded_fragments[fragment_idx],
                        token_valid,
                    )
                    lds_offset_i32 = _raw(
                        ArithValue(token_i32) * fx.Int32(DIM_QK)
                        + ArithValue(dim_i32)
                        + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                    )
                    _lds_store_bf16x8(lds_offset_i32, decoded)

            # The separate RoPE64 buffer contributes eight vector groups per
            # row, exactly two loads per workitem for either configured tile.
            for load_iter in range_constexpr(2):
                group_i32 = _raw(ArithValue(tid_i32) + fx.Int32(load_iter * num_threads))
                token_i32 = _raw(
                    ArithValue(group_i32).with_signedness(False) // fx.Int32(DIM_ROPE // MFMA_INPUT_VALUES)
                )
                dim_group_i32 = _raw(ArithValue(group_i32) % fx.Int32(DIM_ROPE // MFMA_INPUT_VALUES))
                rope_dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                physical_i32 = fx.memref_load(phys_lds, token_i32)
                decoded = _load_rope8(
                    _raw(physical_i32),
                    rope_dim_i32,
                    kv_rope_rsrc,
                )
                token_valid = _raw(ArithValue(tile_i32) + ArithValue(token_i32) < ArithValue(split_end_i32))
                decoded = _mask_bf16x8(decoded, token_valid)
                lds_offset_i32 = _raw(
                    ArithValue(token_i32) * fx.Int32(DIM_QK) + fx.Int32(DIM_NOPE) + ArithValue(rope_dim_i32)
                )
                _lds_store_bf16x8(lds_offset_i32, decoded)
        if const_expr(progressive_vgpr_prefetch):
            # The token4 first fill has exact wave-local QK ownership.  Later
            # rows were published by the preceding iteration's final barrier.
            if ArithValue(tile_i32) == ArithValue(split_start_i32):
                llvm.inline_asm(
                    None,
                    [],
                    "s_waitcnt vmcnt(0) lgkmcnt(0)",
                    "~{memory}",
                    has_side_effects=True,
                )
        elif const_expr(progressive_raw_g2l):
            if ArithValue(tile_i32) == ArithValue(split_start_i32):
                llvm.inline_asm(
                    None,
                    [],
                    "s_waitcnt vmcnt(0) lgkmcnt(0)",
                    "~{memory}",
                    has_side_effects=True,
                )
        elif const_expr(wave_local_qk):
            # token4 assigns wave w every byte of rows [16*w, 16*w+15],
            # exactly the rows consumed by that wave's following QK chain.
            # Drain this wave's dependent VMEM/LDS fill without synchronizing
            # unrelated waves.  The retained max/P barriers protect later
            # cross-wave PV reads, and the retained end barrier protects the
            # next tile from overwriting any current-tile reads.
            llvm.inline_asm(
                None,
                [],
                "s_waitcnt vmcnt(0) lgkmcnt(0)",
                "~{memory}",
                has_side_effects=True,
            )
        elif const_expr(not qh64_raw_pipeline):
            gpu.barrier()

        if const_expr(padded_qh64):
            token_row0_i32 = head_i32
            token_row1_i32 = _raw(fx.Int32(TOKENS_PER_WAVE) + ArithValue(head_i32))
        else:
            token_row0_i32 = _raw(ArithValue(wave_i32) * fx.Int32(TOKENS_PER_WAVE) + ArithValue(head_i32))
            token_row1_i32 = token_row0_i32

        # The qh64 path deliberately exposes two independent 16-token QK
        # chains in every wave.  The legacy routes retain their single chain.
        score_acc0 = zero_f32x4
        score_acc1 = zero_f32x4
        for k_step in range_constexpr(DIM_QK // MFMA_K):
            k_dim_i32 = _raw(ArithValue(lane_group_i32) * fx.Int32(MFMA_INPUT_VALUES) + fx.Int32(k_step * MFMA_K))
            k_offset0_i32 = _raw(ArithValue(token_row0_i32) * fx.Int32(kv_lds_stride) + ArithValue(k_dim_i32))
            if const_expr(qh64_raw_pipeline):
                k_fragment0 = _m11_decoded_load_bf16x8(
                    m11_current_pong_base_i32,
                    k_offset0_i32,
                )
            else:
                k_fragment0 = _lds_load_bf16x8(k_offset0_i32)
            if const_expr(padded_qh64):
                qh64_q_offset_i32 = _raw(ArithValue(head_i32) * fx.Int32(DIM_QK) + ArithValue(k_dim_i32))
                qh64_q_fragment = _mask_bf16x8(
                    _q_lds_load_bf16x8(qh64_q_offset_i32),
                    real_query,
                )
                score_acc0 = _mfma_bf16(
                    k_fragment0,
                    qh64_q_fragment,
                    score_acc0,
                )
                k_offset1_i32 = _raw(ArithValue(token_row1_i32) * fx.Int32(DIM_QK) + ArithValue(k_dim_i32))
                if const_expr(qh64_raw_pipeline):
                    k_fragment1 = _m11_decoded_load_bf16x8(
                        m11_current_pong_base_i32,
                        k_offset1_i32,
                    )
                else:
                    k_fragment1 = _lds_load_bf16x8(k_offset1_i32)
                score_acc1 = _mfma_bf16(
                    k_fragment1,
                    qh64_q_fragment,
                    score_acc1,
                )
            else:
                score_acc0 = _mfma_bf16(
                    k_fragment0,
                    q_fragments[k_step],
                    score_acc0,
                )

        scores = []
        score_count = 2 * MFMA_OUTPUT_VALUES if padded_qh64 else MFMA_OUTPUT_VALUES
        if const_expr(padded_qh64):
            for token_half in range_constexpr(2):
                half_acc = score_acc0 if token_half == 0 else score_acc1
                for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                    token_in_tile_i32 = _raw(
                        fx.Int32(token_half * TOKENS_PER_WAVE)
                        + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                        + fx.Int32(value_idx)
                    )
                    logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_in_tile_i32))
                    valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
                    scaled = arith.mulf(
                        _raw(half_acc[value_idx]),
                        _raw(softmax_score_scale),
                        fastmath=fm_fast,
                    )
                    scores.append(_select(valid, scaled, c_neg_inf))
        else:
            for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                token_in_tile_i32 = _raw(
                    ArithValue(wave_i32) * fx.Int32(TOKENS_PER_WAVE)
                    + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                    + fx.Int32(value_idx)
                )
                logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_in_tile_i32))
                valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
                scaled = arith.mulf(
                    _raw(score_acc0[value_idx]),
                    _raw(softmax_score_scale),
                    fastmath=fm_fast,
                )
                scores.append(_select(valid, scaled, c_neg_inf))

        local_max = scores[0]
        for value_idx in range_constexpr(1, score_count):
            local_max = arith.maximumf(
                local_max,
                scores[value_idx],
                fastmath=fm_no_inf,
            )
        for shuffle_offset in (16, 32):
            peer = fx.Float32(local_max).shuffle_xor(
                fx.Int32(shuffle_offset),
                fx.Int32(WARP_SIZE),
            )
            local_max = arith.maximumf(
                local_max,
                _raw(peer),
                fastmath=fm_no_inf,
            )

        if const_expr(padded_qh64):
            tile_max = local_max
        else:
            if ArithValue(lane_group_i32) == fx.Int32(0):
                max_offset_i32 = _raw(ArithValue(wave_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
                fx.memref_store(local_max, red_lds, max_offset_i32)
            gpu.barrier()

            tile_max = c_neg_inf.ir_value()
            for source_wave in range_constexpr(num_warps):
                value = fx.memref_load(
                    red_lds,
                    _raw(ArithValue(head_i32) + fx.Int32(source_wave * NUM_HEADS)),
                )
                tile_max = arith.maximumf(
                    tile_max,
                    _raw(value),
                    fastmath=fm_no_inf,
                )

        row_max_old = fx.Float32(state[0])
        row_sum_old = fx.Float32(state[1])
        row_max_new = fx.Float32(
            arith.maximumf(
                _raw(row_max_old),
                tile_max,
                fastmath=fm_no_inf,
            )
        )
        old_scale = fx.Float32(
            _softmax_exp_difference(
                arith.subf(
                    _raw(row_max_old),
                    _raw(row_max_new),
                    fastmath=fm_no_inf,
                )
            )
        )

        probabilities = []
        local_sum = c_zero.ir_value()
        for value_idx in range_constexpr(score_count):
            probability = _softmax_exp_difference(
                arith.subf(
                    scores[value_idx],
                    _raw(row_max_new),
                    fastmath=fm_no_inf,
                )
            )
            probabilities.append(probability)
            local_sum = arith.addf(
                local_sum,
                probability,
                fastmath=fm_no_inf,
            )
        for shuffle_offset in (16, 32):
            peer = fx.Float32(local_sum).shuffle_xor(
                fx.Int32(shuffle_offset),
                fx.Int32(WARP_SIZE),
            )
            local_sum = arith.addf(
                local_sum,
                _raw(peer),
                fastmath=fm_no_inf,
            )

        if const_expr(padded_qh64):
            tile_sum = local_sum
            p_bf16 = Vec.from_elements(
                probabilities,
                fx.Float32,
            ).to(fx.BFloat16)
        else:
            if ArithValue(lane_group_i32) == fx.Int32(0):
                sum_offset_i32 = _raw(
                    fx.Int32(num_warps * NUM_HEADS) + ArithValue(wave_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32)
                )
                fx.memref_store(local_sum, red_lds, sum_offset_i32)

            p_base_i32 = _raw(
                (ArithValue(wave_i32) * fx.Int32(WARP_SIZE) + ArithValue(lane_i32)) * fx.Int32(MFMA_OUTPUT_VALUES)
            )
            p_bf16 = Vec.from_elements(
                probabilities,
                fx.Float32,
            ).to(fx.BFloat16)
            for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                fx.memref_store(
                    _raw(p_bf16[value_idx]),
                    p_lds,
                    _raw(ArithValue(p_base_i32) + fx.Int32(value_idx)),
                )
            gpu.barrier()

            tile_sum = c_zero.ir_value()
            for source_wave in range_constexpr(num_warps):
                value = fx.memref_load(
                    red_lds,
                    _raw(fx.Int32(num_warps * NUM_HEADS) + fx.Int32(source_wave * NUM_HEADS) + ArithValue(head_i32)),
                )
                tile_sum = arith.addf(
                    tile_sum,
                    _raw(value),
                    fastmath=fm_no_inf,
                )
        row_sum_new = fx.Float32(
            arith.addf(
                arith.mulf(
                    _raw(row_sum_old),
                    _raw(old_scale),
                    fastmath=fm_no_inf,
                ),
                tile_sum,
                fastmath=fm_no_inf,
            )
        )

        # Keep all independent PV accumulator chains visible together.  The
        # probability fragment depends only on the source-wave pair, so load
        # it once and reuse it across the output-dimension tiles of this wave.
        pv_accumulators = []
        for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
            accumulator = Vec(
                state[2 + dim_tile],
                (MFMA_OUTPUT_VALUES,),
                fx.Float32,
            )
            pv_accumulators.append(accumulator * old_scale)

        # Each PV MFMA contracts two adjacent 16-token halves.  The qh64
        # specialization has exactly one B32 pair and reuses its VGPR P
        # fragment across all 32 independent output accumulators.  Legacy
        # routes retain their LDS P source pairs and dimension ownership.
        pv_source_pairs = 1 if padded_qh64 else num_warps // 2
        if const_expr(lookahead_token4_prefetch):
            # Issue the complete next B128 tile once, then keep its raw values
            # live across all four source pairs / sixteen current-tile PV
            # MFMAs.  The runtime-uniform branch suppresses all final-tile
            # speculative traffic.
            m10_has_next_tile = _raw(ArithValue(tile_i32) + fx.Int32(block_n) < ArithValue(split_end_i32))
            m10_packed_groups = [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(7)]
            if const_expr(opaque_payload_ladder):
                m10_scale_bytes = [fx.Int32(0) for _ in range_constexpr(7)]
                m10_rope_groups = [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(2)]
            else:
                m10_scale_bytes = [fx.Int8(0) for _ in range_constexpr(7)]
                m10_rope_groups = [zero_bf16x8 for _ in range_constexpr(2)]
            (
                m10_packed_0,
                m10_packed_1,
                m10_packed_2,
                m10_packed_3,
                m10_packed_4,
                m10_packed_5,
                m10_packed_6,
            ) = m10_packed_groups
            (
                m10_scale_0,
                m10_scale_1,
                m10_scale_2,
                m10_scale_3,
                m10_scale_4,
                m10_scale_5,
                m10_scale_6,
            ) = m10_scale_bytes
            (
                m10_rope_0,
                m10_rope_1,
            ) = m10_rope_groups

            if m10_has_next_tile:
                m10_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
                m10_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
                m10_next_tile_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n))
                m10_next_logical_i32 = _raw(ArithValue(m10_next_tile_i32) + ArithValue(m10_token_i32))
                m10_next_valid = _raw(ArithValue(m10_next_logical_i32) < ArithValue(split_end_i32))
                m10_safe_logical_i32 = _select(
                    m10_next_valid,
                    m10_next_logical_i32,
                    split_start_i32,
                )
                m10_physical_lane_i32 = fx.Int32(0).ir_value()
                if ArithValue(m10_fragment_lane_i32) == fx.Int32(0):
                    m10_physical_lane_i32 = buffer_ops.buffer_load(
                        kv_indices_rsrc,
                        m10_safe_logical_i32,
                        vec_width=1,
                        dtype=T.i32,
                    )
                m10_physical_i32 = dpp_utils.update_dpp_i32(
                    m10_physical_lane_i32,
                    m10_physical_lane_i32,
                    0,
                )

                m10_loaded_packed_groups = []
                m10_loaded_scale_bytes = []
                for m10_load_iter in range_constexpr(7):
                    m10_dim_group_i32 = _raw(ArithValue(m10_fragment_lane_i32) + fx.Int32(m10_load_iter * 4))
                    m10_dim_i32 = _raw(ArithValue(m10_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                    m10_dword_offset_i32 = _raw(
                        ArithValue(m10_physical_i32) * fx.Int32(PACKED_DWORDS)
                        + ArithValue(m10_dim_i32).with_signedness(False) // fx.Int32(4)
                    )
                    if const_expr(opaque_payload_ladder):
                        m10_loaded_packed_groups.append(
                            _m15_opaque_load_i32x4(
                                kv_packed_rsrc,
                                _raw(ArithValue(m10_dword_offset_i32) * fx.Int32(4)),
                            )
                        )
                    else:
                        m10_loaded_packed_groups.append(
                            Vec(
                                buffer_ops.buffer_load(
                                    kv_packed_rsrc,
                                    m10_dword_offset_i32,
                                    vec_width=4,
                                    dtype=T.i32,
                                ),
                                (4,),
                                fx.Int32,
                            )
                        )
                    m10_scale_offset_i32 = _raw(
                        ArithValue(m10_physical_i32) * fx.Int32(DIM_PACKED)
                        + fx.Int32(NOPE_SCALE_OFFSET)
                        + (ArithValue(m10_dim_i32).with_signedness(False) // fx.Int32(MFMA_K))
                    )
                    if const_expr(opaque_payload_ladder):
                        m10_loaded_scale_bytes.append(
                            _m15_opaque_load_scale_i32(
                                kv_packed_rsrc,
                                m10_scale_offset_i32,
                            )
                        )
                    else:
                        m10_loaded_scale_bytes.append(
                            buffer_ops.buffer_load(
                                kv_packed_rsrc,
                                m10_scale_offset_i32,
                                vec_width=1,
                                dtype=T.i8,
                            )
                        )

                m10_loaded_rope_groups = []
                for m10_load_iter in range_constexpr(2):
                    m10_dim_group_i32 = _raw(ArithValue(m10_fragment_lane_i32) + fx.Int32(m10_load_iter * 4))
                    m10_rope_dim_i32 = _raw(ArithValue(m10_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                    if const_expr(opaque_payload_ladder):
                        m10_rope_byte_offset_i32 = _raw(
                            (ArithValue(m10_physical_i32) * fx.Int32(DIM_ROPE) + ArithValue(m10_rope_dim_i32))
                            * fx.Int32(2)
                        )
                        m10_loaded_rope_groups.append(
                            _m15_opaque_load_i32x4(
                                kv_rope_rsrc,
                                m10_rope_byte_offset_i32,
                            )
                        )
                    else:
                        m10_loaded_rope_groups.append(
                            _load_rope8(
                                _raw(m10_physical_i32),
                                m10_rope_dim_i32,
                                kv_rope_rsrc,
                            )
                        )

                (
                    m10_packed_0,
                    m10_packed_1,
                    m10_packed_2,
                    m10_packed_3,
                    m10_packed_4,
                    m10_packed_5,
                    m10_packed_6,
                ) = m10_loaded_packed_groups
                (
                    m10_scale_0,
                    m10_scale_1,
                    m10_scale_2,
                    m10_scale_3,
                    m10_scale_4,
                    m10_scale_5,
                    m10_scale_6,
                ) = m10_loaded_scale_bytes
                (
                    m10_rope_0,
                    m10_rope_1,
                ) = m10_loaded_rope_groups

            m10_packed_groups = [
                m10_packed_0,
                m10_packed_1,
                m10_packed_2,
                m10_packed_3,
                m10_packed_4,
                m10_packed_5,
                m10_packed_6,
            ]
            m10_scale_bytes = [
                m10_scale_0,
                m10_scale_1,
                m10_scale_2,
                m10_scale_3,
                m10_scale_4,
                m10_scale_5,
                m10_scale_6,
            ]
            m10_rope_groups = [
                m10_rope_0,
                m10_rope_1,
            ]
        if const_expr(progressive_vgpr_prefetch):
            m6_has_next_tile = _raw(ArithValue(tile_i32) + fx.Int32(block_n) < ArithValue(split_end_i32))
        if const_expr(progressive_raw_g2l):
            m6r_has_next_tile = _raw(ArithValue(tile_i32) + fx.Int32(block_n) < ArithValue(split_end_i32))
        for source_pair in range_constexpr(pv_source_pairs):
            source0 = source_pair * 2
            source1 = source0 + 1
            if const_expr(padded_qh64):
                p_fragment = p_bf16
            else:
                p_values = []
                for source_wave in (source0, source1):
                    source_p_base_i32 = _raw(
                        (fx.Int32(source_wave * WARP_SIZE) + ArithValue(lane_i32)) * fx.Int32(MFMA_OUTPUT_VALUES)
                    )
                    for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                        p_value = fx.memref_load(
                            p_lds,
                            _raw(ArithValue(source_p_base_i32) + fx.Int32(value_idx)),
                        )
                        p_values.append(_raw(p_value))
                p_fragment = Vec.from_elements(p_values, fx.BFloat16)

            if const_expr(progressive_vgpr_prefetch):
                # Lane f of each 16-lane subgroup carries two adjacent 16-B
                # packed fragments for token j.  The first 14 lanes also keep
                # the matching E8M0 scale byte, while lanes 0..7 carry the
                # eight RoPE vec8 fragments.  These values remain in VGPRs
                # across this pair's current-tile PV chain.
                m6_fragment_i32 = _raw(ArithValue(lane_i32) % fx.Int32(16))
                m6_token_in_wave_i32 = _raw(ArithValue(lane_i32).with_signedness(False) // fx.Int32(16))
                m6_next_token_i32 = _raw(
                    fx.Int32(source_pair * 2 * TOKENS_PER_WAVE)
                    + ArithValue(wave_i32) * fx.Int32(4)
                    + ArithValue(m6_token_in_wave_i32)
                )
                m6_next_logical_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n) + ArithValue(m6_next_token_i32))
                m6_next_valid = _raw(ArithValue(m6_next_logical_i32) < ArithValue(split_end_i32))
                m6_safe_logical_i32 = _select(
                    m6_next_valid,
                    m6_next_logical_i32,
                    split_start_i32,
                )

                m6_packed_even = Vec.filled(4, 0, fx.Int32)
                m6_packed_odd = Vec.filled(4, 0, fx.Int32)
                m6_scale_byte_i8 = fx.Int8(0)
                m6_rope = zero_bf16x8
                if m6_has_next_tile:
                    m6_physical_lane_i32 = fx.Int32(0).ir_value()
                    if ArithValue(m6_fragment_i32) == fx.Int32(0):
                        m6_physical_lane_i32 = buffer_ops.buffer_load(
                            kv_indices_rsrc,
                            m6_safe_logical_i32,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    m6_leader_lane_i32 = _raw(ArithValue(lane_i32) - ArithValue(m6_fragment_i32))
                    m6_physical_i32 = rocdl.ds_bpermute(
                        T.i32,
                        _raw(ArithValue(m6_leader_lane_i32) * fx.Int32(4)),
                        _raw(m6_physical_lane_i32),
                    )
                    m6_packed_dword_i32 = _raw(
                        ArithValue(m6_physical_i32) * fx.Int32(PACKED_DWORDS)
                        + ArithValue(m6_fragment_i32) * fx.Int32(8)
                    )
                    m6_packed_even = Vec(
                        buffer_ops.buffer_load(
                            kv_packed_rsrc,
                            m6_packed_dword_i32,
                            vec_width=4,
                            dtype=T.i32,
                        ),
                        (4,),
                        fx.Int32,
                    )
                    m6_packed_odd = Vec(
                        buffer_ops.buffer_load(
                            kv_packed_rsrc,
                            _raw(ArithValue(m6_packed_dword_i32) + fx.Int32(4)),
                            vec_width=4,
                            dtype=T.i32,
                        ),
                        (4,),
                        fx.Int32,
                    )
                    if ArithValue(m6_fragment_i32) < fx.Int32(14):
                        m6_scale_byte_i8 = buffer_ops.buffer_load(
                            kv_packed_rsrc,
                            _raw(
                                ArithValue(m6_physical_i32) * fx.Int32(DIM_PACKED)
                                + fx.Int32(NOPE_SCALE_OFFSET)
                                + ArithValue(m6_fragment_i32)
                            ),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    if ArithValue(m6_fragment_i32) < fx.Int32(8):
                        m6_rope = _load_rope8(
                            _raw(m6_physical_i32),
                            _raw(ArithValue(m6_fragment_i32) * fx.Int32(MFMA_INPUT_VALUES)),
                            kv_rope_rsrc,
                        )

            if const_expr(progressive_raw_g2l):
                # Stage the same next-pair ownership map directly into one
                # wave-private raw LDS slot.  The opaque G2L instructions do
                # not expose an LDS alias to LLVM, allowing their VMEM latency
                # to overlap the current pair's eight tr16 reads/four MFMAs.
                m6r_fragment_i32 = _raw(ArithValue(lane_i32) % fx.Int32(16))
                m6r_token_in_wave_i32 = _raw(ArithValue(lane_i32).with_signedness(False) // fx.Int32(16))
                m6r_next_token_i32 = _raw(
                    fx.Int32(source_pair * 2 * TOKENS_PER_WAVE)
                    + ArithValue(wave_i32) * fx.Int32(4)
                    + ArithValue(m6r_token_in_wave_i32)
                )
                m6r_next_logical_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n) + ArithValue(m6r_next_token_i32))
                m6r_next_valid = _raw(ArithValue(m6r_next_logical_i32) < ArithValue(split_end_i32))
                m6r_safe_logical_i32 = _select(
                    m6r_next_valid,
                    m6r_next_logical_i32,
                    split_start_i32,
                )

                if m6r_has_next_tile:
                    m6r_physical_lane_i32 = fx.Int32(0).ir_value()
                    if ArithValue(m6r_fragment_i32) == fx.Int32(0):
                        m6r_physical_lane_i32 = buffer_ops.buffer_load(
                            kv_indices_rsrc,
                            m6r_safe_logical_i32,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    m6r_leader_lane_i32 = _raw(ArithValue(lane_i32) - ArithValue(m6r_fragment_i32))
                    m6r_physical_i32 = rocdl.ds_bpermute(
                        T.i32,
                        _raw(ArithValue(m6r_leader_lane_i32) * fx.Int32(4)),
                        _raw(m6r_physical_lane_i32),
                    )
                    for m6r_half in range_constexpr(2):
                        _m6_raw_buffer_load_lds(
                            kv_packed_rsrc,
                            shared.raw_packed,
                            _raw(ArithValue(wave_i32) * fx.Int32(2048) + fx.Int32(m6r_half * 1024)),
                            _raw(
                                ArithValue(m6r_physical_i32) * fx.Int32(DIM_PACKED)
                                + ArithValue(m6r_fragment_i32) * fx.Int32(4 * MFMA_INPUT_VALUES)
                                + fx.Int32(m6r_half * 16)
                            ),
                        )
                    if ArithValue(m6r_fragment_i32) < fx.Int32(8):
                        _m6_raw_buffer_load_lds(
                            kv_rope_rsrc,
                            shared.raw_rope,
                            _raw(ArithValue(wave_i32) * fx.Int32(1024)),
                            _raw(
                                ArithValue(m6r_physical_i32) * fx.Int32(DIM_ROPE * 2)
                                + ArithValue(m6r_fragment_i32) * fx.Int32(2 * MFMA_INPUT_VALUES)
                            ),
                        )

            # M5 exposes all four independent output-dimension fragments
            # before consuming any of them.  This gives LLVM eight native
            # transpose reads to schedule across the same four PV MFMAs while
            # preserving the original dim_tile accumulation order.
            wide_pv_fragments = []
            if const_expr(wide_pv_qh64):
                if const_expr(not padded_qh64):
                    for wide_dim_tile in range_constexpr(output_mfma_tiles_per_wave):
                        tr_source_token_i32 = _raw(
                            fx.Int32(source0 * TOKENS_PER_WAVE)
                            + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                            + (ArithValue(head_i32).with_signedness(False) // fx.Int32(MFMA_OUTPUT_VALUES))
                        )
                        tr_source_dim_i32 = _raw(
                            ArithValue(wave_i32) * fx.Int32(output_dims_per_wave)
                            + fx.Int32(wide_dim_tile * 16)
                            + (ArithValue(head_i32) % fx.Int32(MFMA_OUTPUT_VALUES)) * fx.Int32(MFMA_OUTPUT_VALUES)
                        )
                        tr_base_i32 = _raw(
                            ArithValue(tr_source_token_i32) * fx.Int32(kv_lds_stride) + ArithValue(tr_source_dim_i32)
                        )
                        v_lo = _qh64_ds_read_tr16_bf16x4(tr_base_i32)
                        v_hi = _qh64_ds_read_tr16_bf16x4(
                            _raw(ArithValue(tr_base_i32) + fx.Int32(TOKENS_PER_WAVE * kv_lds_stride))
                        )
                        wide_pv_fragments.append(
                            v_lo.shuffle(
                                v_hi,
                                list(range(MFMA_INPUT_VALUES)),
                            )
                        )

            for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
                if const_expr(padded_qh64):
                    dim_row_i32 = _raw(fx.Int32(dim_tile * 16) + ArithValue(head_i32))
                else:
                    dim_row_i32 = _raw(
                        ArithValue(wave_i32) * fx.Int32(output_dims_per_wave)
                        + fx.Int32(dim_tile * 16)
                        + ArithValue(head_i32)
                    )
                if const_expr(wide_pv_qh64):
                    # gfx950 tr16 maps row-major [token, dim] LDS directly
                    # into MFMA-A lane order.  For output wave w, source
                    # wave s, h=lane%16, and g=lane//16, the source pointer is
                    # token=s*16+g*4+h//4 and
                    # dim=w*64+16*dim_tile+4*(h%4).  The transpose result is
                    # tokens s*16+g*4+[0..3] at dimension
                    # w*64+16*dim_tile+h.
                    if const_expr(padded_qh64):
                        tr_source_token_i32 = _raw(
                            ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                            + (ArithValue(head_i32).with_signedness(False) // fx.Int32(MFMA_OUTPUT_VALUES))
                        )
                        tr_source_dim_i32 = _raw(
                            fx.Int32(dim_tile * 16)
                            + (ArithValue(head_i32) % fx.Int32(MFMA_OUTPUT_VALUES)) * fx.Int32(MFMA_OUTPUT_VALUES)
                        )
                        tr_base_i32 = _raw(
                            ArithValue(tr_source_token_i32) * fx.Int32(DIM_QK) + ArithValue(tr_source_dim_i32)
                        )
                        if const_expr(qh64_raw_pipeline):
                            v_lo = _m11_decoded_ds_read_tr16_bf16x4(
                                m11_current_pong_base_i32,
                                tr_base_i32,
                            )
                            v_hi = _m11_decoded_ds_read_tr16_bf16x4(
                                m11_current_pong_base_i32,
                                _raw(ArithValue(tr_base_i32) + fx.Int32(TOKENS_PER_WAVE * DIM_QK)),
                            )
                        else:
                            v_lo = _qh64_ds_read_tr16_bf16x4(tr_base_i32)
                            v_hi = _qh64_ds_read_tr16_bf16x4(
                                _raw(ArithValue(tr_base_i32) + fx.Int32(TOKENS_PER_WAVE * DIM_QK))
                            )
                        v_fragment = v_lo.shuffle(
                            v_hi,
                            list(range(MFMA_INPUT_VALUES)),
                        )
                    else:
                        v_fragment = wide_pv_fragments[dim_tile]
                else:
                    v_values = []
                    for source_wave in (source0, source1):
                        token_base_i32 = _raw(
                            fx.Int32(source_wave * TOKENS_PER_WAVE)
                            + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                        )
                        for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                            token_i32 = _raw(ArithValue(token_base_i32) + fx.Int32(value_idx))
                            v_value = fx.memref_load(
                                kv_lds,
                                _raw(ArithValue(token_i32) * fx.Int32(DIM_QK) + ArithValue(dim_row_i32)),
                            )
                            v_values.append(_raw(v_value))

                    v_fragment = Vec.from_elements(
                        v_values,
                        fx.BFloat16,
                    )
                pv_accumulators[dim_tile] = _mfma_bf16(
                    v_fragment,
                    p_fragment,
                    pv_accumulators[dim_tile],
                )

            if const_expr(progressive_vgpr_prefetch):
                # Every wave reads all 32 current rows in this source pair.
                # Retire those reads globally before overwriting the same rows
                # with the prefetched next-tile pair.
                gpu.barrier()
                if m6_has_next_tile:
                    llvm.inline_asm(
                        None,
                        [],
                        "s_waitcnt vmcnt(0)",
                        "~{memory}",
                        has_side_effects=True,
                    )
                    m6_destination_token_i32 = m6_next_token_i32
                    if ArithValue(m6_fragment_i32) < fx.Int32(14):
                        m6_decoded_even = _decode_nope16_prefetched(
                            m6_packed_even,
                            m6_scale_byte_i8,
                        )
                        m6_decoded_odd = _decode_nope16_prefetched(
                            m6_packed_odd,
                            m6_scale_byte_i8,
                        )
                        for m6_half in range_constexpr(2):
                            for m6_fragment in range_constexpr(2):
                                m6_decoded = (
                                    m6_decoded_even[m6_fragment] if m6_half == 0 else m6_decoded_odd[m6_fragment]
                                )
                                m6_decoded = _mask_bf16x8(
                                    m6_decoded,
                                    m6_next_valid,
                                )
                                m6_dimension_i32 = _raw(
                                    ArithValue(m6_fragment_i32) * fx.Int32(MFMA_K)
                                    + fx.Int32(m6_half * 2 * MFMA_INPUT_VALUES + m6_fragment * MFMA_INPUT_VALUES)
                                )
                                _lds_store_bf16x8(
                                    _raw(
                                        ArithValue(m6_destination_token_i32) * fx.Int32(DIM_QK)
                                        + ArithValue(m6_dimension_i32)
                                    ),
                                    m6_decoded,
                                )
                    if ArithValue(m6_fragment_i32) < fx.Int32(8):
                        _lds_store_bf16x8(
                            _raw(
                                ArithValue(m6_destination_token_i32) * fx.Int32(DIM_QK)
                                + fx.Int32(DIM_NOPE)
                                + ArithValue(m6_fragment_i32) * fx.Int32(MFMA_INPUT_VALUES)
                            ),
                            _mask_bf16x8(
                                m6_rope,
                                m6_next_valid,
                            ),
                        )

            if const_expr(progressive_raw_g2l):
                gpu.barrier()
                if m6r_has_next_tile:
                    llvm.inline_asm(
                        None,
                        [],
                        "s_waitcnt vmcnt(0)",
                        "~{memory}",
                        has_side_effects=True,
                    )
                    # Recompute the cheap lane/token map here instead of
                    # extending those VGPR live ranges across the PV chain.
                    m6rd_fragment_i32 = _raw(ArithValue(lane_i32) % fx.Int32(16))
                    m6rd_token_in_wave_i32 = _raw(ArithValue(lane_i32).with_signedness(False) // fx.Int32(16))
                    m6rd_next_token_i32 = _raw(
                        fx.Int32(source_pair * 2 * TOKENS_PER_WAVE)
                        + ArithValue(wave_i32) * fx.Int32(4)
                        + ArithValue(m6rd_token_in_wave_i32)
                    )
                    m6rd_next_logical_i32 = _raw(
                        ArithValue(tile_i32) + fx.Int32(block_n) + ArithValue(m6rd_next_token_i32)
                    )
                    m6rd_next_valid = _raw(ArithValue(m6rd_next_logical_i32) < ArithValue(split_end_i32))
                    m6rd_wave_packed_base_i32 = _raw(ArithValue(wave_i32) * fx.Int32(2048))
                    m6rd_lane_byte_i32 = _raw(ArithValue(lane_i32) * fx.Int32(16))
                    if ArithValue(m6rd_fragment_i32) < fx.Int32(14):
                        m6rd_scale_byte_i8 = _m6_raw_lds_load_i8(
                            shared.raw_packed,
                            _raw(
                                ArithValue(m6rd_wave_packed_base_i32)
                                + ArithValue(m6rd_token_in_wave_i32) * fx.Int32(256)
                                + fx.Int32(14 * 16)
                                + ArithValue(m6rd_fragment_i32)
                            ),
                        )
                        for m6r_half in range_constexpr(2):
                            m6rd_packed = _m6_raw_lds_load_i32x4(
                                shared.raw_packed,
                                _raw(
                                    ArithValue(m6rd_wave_packed_base_i32)
                                    + fx.Int32(m6r_half * 1024)
                                    + ArithValue(m6rd_lane_byte_i32)
                                ),
                            )
                            m6rd_decoded_pair = _decode_nope16_prefetched(
                                m6rd_packed,
                                m6rd_scale_byte_i8,
                            )
                            for m6r_fragment in range_constexpr(2):
                                m6rd_decoded = _mask_bf16x8(
                                    m6rd_decoded_pair[m6r_fragment],
                                    m6rd_next_valid,
                                )
                                m6rd_dimension_i32 = _raw(
                                    ArithValue(m6rd_fragment_i32) * fx.Int32(MFMA_K)
                                    + fx.Int32(m6r_half * 2 * MFMA_INPUT_VALUES + m6r_fragment * MFMA_INPUT_VALUES)
                                )
                                _lds_store_bf16x8(
                                    _raw(
                                        ArithValue(m6rd_next_token_i32) * fx.Int32(DIM_QK)
                                        + ArithValue(m6rd_dimension_i32)
                                    ),
                                    m6rd_decoded,
                                )
                    if ArithValue(m6rd_fragment_i32) < fx.Int32(8):
                        m6rd_rope = _m6_raw_lds_load_bf16x8(
                            shared.raw_rope,
                            _raw(ArithValue(wave_i32) * fx.Int32(1024) + ArithValue(m6rd_lane_byte_i32)),
                        )
                        _lds_store_bf16x8(
                            _raw(
                                ArithValue(m6rd_next_token_i32) * fx.Int32(DIM_QK)
                                + fx.Int32(DIM_NOPE)
                                + ArithValue(m6rd_fragment_i32) * fx.Int32(MFMA_INPUT_VALUES)
                            ),
                            _mask_bf16x8(
                                m6rd_rope,
                                m6rd_next_valid,
                            ),
                        )

        next_state = [_raw(row_max_new), _raw(row_sum_new)]
        for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
            next_state.append(_raw(pv_accumulators[dim_tile]))

        if const_expr(qh64_raw_pipeline):
            # Each wave waits only for its own five future G2L requests, then
            # decodes its own eight rows into the opposite pong.  The single
            # unconditional barrier both retires all current-pong readers and
            # publishes every wave's decoded next-pong rows.
            llvm.inline_asm(
                None,
                [],
                "s_waitcnt vmcnt(0)",
                "~{memory}",
                has_side_effects=True,
            )
            m11_next_tile_i32 = _raw(ArithValue(tile_i32) + fx.Int32(QH64_B32_BLOCK_N))
            m11_has_next_tile = _raw(ArithValue(m11_next_tile_i32) < ArithValue(split_end_i32))
            m11_next_ordinal_i32 = _raw(ArithValue(m11_tile_ordinal_i32) + fx.Int32(1))
            m11_next_bank_i32 = _raw(ArithValue(m11_next_ordinal_i32) % fx.Int32(QH64_M11_RAW_BANKS))
            m11_next_pong_i32 = _raw(ArithValue(m11_next_ordinal_i32) % fx.Int32(QH64_M11_DECODED_PONGS))
            m11_next_bank_base_i32 = _raw(ArithValue(m11_next_bank_i32) * fx.Int32(QH64_M11_RAW_BANK_BYTES))
            m11_next_pong_base_i32 = _raw(ArithValue(m11_next_pong_i32) * fx.Int32(QH64_B32_KV_TILE_ELEMS))
            if m11_has_next_tile:
                _m11_decode_raw_tile(
                    m11_next_tile_i32,
                    m11_next_bank_base_i32,
                    m11_next_pong_base_i32,
                )
            llvm.inline_asm(
                None,
                [],
                "s_waitcnt lgkmcnt(0)",
                "~{memory}",
                has_side_effects=True,
            )
            gpu.barrier()
        else:
            # All waves must finish reading the current KV/P workspaces before
            # a fast wave starts filling the next legacy tile.
            gpu.barrier()
        if const_expr(lookahead_token4_prefetch):
            # The existing barrier retires every current-tile LDS reader.
            # Consume the prefetched SSA values only now; their dependencies
            # provide progressive VMEM waits without an eager vmcnt(0).
            if m10_has_next_tile:
                m10_store_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
                m10_store_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
                m10_store_next_tile_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n))
                m10_store_logical_i32 = _raw(ArithValue(m10_store_next_tile_i32) + ArithValue(m10_store_token_i32))
                m10_store_valid = _raw(ArithValue(m10_store_logical_i32) < ArithValue(split_end_i32))
                for m10_store_iter in range_constexpr(7):
                    m10_store_dim_group_i32 = _raw(
                        ArithValue(m10_store_fragment_lane_i32) + fx.Int32(m10_store_iter * 4)
                    )
                    m10_store_dim_i32 = _raw(ArithValue(m10_store_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                    if const_expr(opaque_payload_ladder):
                        # Two oldest requests are the packed/scale pair for
                        # this group; leave every younger pair outstanding.
                        _m15_wait_vmcnt(14 - 2 * m10_store_iter)
                    m10_decoded_fragments = _decode_nope16_prefetched(
                        m10_packed_groups[m10_store_iter],
                        m10_scale_bytes[m10_store_iter],
                        opaque_payload_ladder,
                    )
                    for m10_fragment_idx in range_constexpr(2):
                        m10_decoded = _mask_bf16x8(
                            m10_decoded_fragments[m10_fragment_idx],
                            m10_store_valid,
                        )
                        m10_store_lds_offset_i32 = _raw(
                            ArithValue(m10_store_token_i32) * fx.Int32(kv_lds_stride)
                            + ArithValue(m10_store_dim_i32)
                            + fx.Int32(m10_fragment_idx * MFMA_INPUT_VALUES)
                        )
                        _lds_store_bf16x8(
                            m10_store_lds_offset_i32,
                            m10_decoded,
                        )

                for m10_store_iter in range_constexpr(2):
                    m10_store_dim_group_i32 = _raw(
                        ArithValue(m10_store_fragment_lane_i32) + fx.Int32(m10_store_iter * 4)
                    )
                    m10_store_rope_dim_i32 = _raw(ArithValue(m10_store_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                    m10_store_lds_offset_i32 = _raw(
                        ArithValue(m10_store_token_i32) * fx.Int32(kv_lds_stride)
                        + fx.Int32(DIM_NOPE)
                        + ArithValue(m10_store_rope_dim_i32)
                    )
                    m10_store_rope_group = m10_rope_groups[m10_store_iter]
                    if const_expr(opaque_payload_ladder):
                        # The final two queue entries are rope0/rope1.
                        _m15_wait_vmcnt(1 - m10_store_iter)
                        m10_store_rope_group = m10_store_rope_group.bitcast(fx.BFloat16)
                    _lds_store_bf16x8(
                        m10_store_lds_offset_i32,
                        _mask_bf16x8(
                            m10_store_rope_group,
                            m10_store_valid,
                        ),
                    )
        results = yield next_state

    if is_valid_split:
        row_max_tokens = fx.Float32(results[0])
        row_sum_tokens = fx.Float32(results[1])
        sink_softmax_domain = sink_lane
        if const_expr(log2_domain_softmax):
            # The sink arrives in natural-log units.  Convert it once before
            # max/rescale so every subsequent difference is already log2.
            sink_softmax_domain = fx.Float32(
                arith.mulf(
                    _raw(sink_lane),
                    _raw(fx.Float32(LOG2E)),
                    fastmath=fm_fast,
                )
            )
        row_max_with_sink = arith.maximumf(
            _raw(row_max_tokens),
            _raw(sink_softmax_domain),
            fastmath=fm_no_inf,
        )
        row_max_final = fx.Float32(
            _select(
                is_last_valid_split,
                row_max_with_sink,
                row_max_tokens,
            )
        )
        token_scale = fx.Float32(
            _softmax_exp_difference(
                arith.subf(
                    _raw(row_max_tokens),
                    _raw(row_max_final),
                    fastmath=fm_no_inf,
                )
            )
        )
        sink_scale = _softmax_exp_difference(
            arith.subf(
                _raw(sink_softmax_domain),
                _raw(row_max_final),
                fastmath=fm_no_inf,
            )
        )
        sink_contribution = _select(
            is_last_valid_split,
            sink_scale,
            c_zero,
        )
        row_sum_final = fx.Float32(
            arith.addf(
                arith.mulf(
                    _raw(row_sum_tokens),
                    _raw(token_scale),
                    fastmath=fm_no_inf,
                ),
                sink_contribution,
                fastmath=fm_no_inf,
            )
        )
        inv_sum = fx.Float32(rocdl.rcp(T.f32, _raw(row_sum_final)))
        is_direct_output = _raw(ArithValue(_raw(num_kv_splits.ir_value())) == fx.Int32(1))
        output_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
        partial_record_i32 = _raw(
            (ArithValue(q_row_i32) * ArithValue(num_kv_splits) + ArithValue(split_i32)) * fx.Int32(NUM_HEADS)
            + ArithValue(head_i32)
        )

        for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
            accumulator = Vec(
                results[2 + dim_tile],
                (MFMA_OUTPUT_VALUES,),
                fx.Float32,
            )
            normalized = accumulator * token_scale * inv_sum
            if const_expr(padded_qh64):
                dim_offset_i32 = _raw(
                    fx.Int32(dim_tile * 16) + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                )
            else:
                dim_offset_i32 = _raw(
                    ArithValue(wave_i32) * fx.Int32(output_dims_per_wave)
                    + fx.Int32(dim_tile * 16)
                    + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                )

            def _store_output_tile(
                normalized_value,
                dim_offset_value_i32,
            ):
                if is_direct_output:
                    buffer_ops.buffer_store(
                        normalized_value.to(fx.BFloat16),
                        output_rsrc,
                        _raw(ArithValue(output_record_i32) * fx.Int32(V_HEAD_DIM) + ArithValue(dim_offset_value_i32)),
                    )
                else:
                    buffer_ops.buffer_store(
                        normalized_value,
                        logits_rsrc,
                        _raw(ArithValue(partial_record_i32) * fx.Int32(V_HEAD_DIM) + ArithValue(dim_offset_value_i32)),
                    )
                    if dim_tile == 0:
                        if ArithValue(wave_i32) == fx.Int32(0):
                            if ArithValue(lane_group_i32) == fx.Int32(0):
                                log2_sum = fmath.log2(
                                    row_sum_final,
                                    fastmath=fm_no_inf,
                                )
                                if const_expr(log2_domain_softmax):
                                    # M12-L keeps both terms in log2 until one
                                    # final conversion to the reducer's
                                    # natural-log LSE contract.
                                    lse_log2 = row_max_final + log2_sum
                                    lse_value = lse_log2 * fx.Float32(INV_LOG2E)
                                else:
                                    lse_value = row_max_final + log2_sum * fx.Float32(INV_LOG2E)
                                buffer_ops.buffer_store(
                                    lse_value,
                                    attn_lse_rsrc,
                                    partial_record_i32,
                                )

            if const_expr(padded_qh64):
                # Only the 16 real wave-0 rows are externally visible.
                if ArithValue(wave_i32) == fx.Int32(0):
                    _store_output_tile(normalized, dim_offset_i32)
            else:
                _store_output_tile(normalized, dim_offset_i32)


_kn_mla_v4_sparse_decode_mfma_n128_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_n128_split4_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_n128_split4_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_qh64_b32_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_qh64_b32_m1_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m1_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)
_kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline_source = types.FunctionType(
    _kn_mla_v4_sparse_decode_mfma_impl.__code__,
    _kn_mla_v4_sparse_decode_mfma_impl.__globals__,
    _kn_mla_v4_sparse_decode_mfma_impl.__name__,
    _kn_mla_v4_sparse_decode_mfma_impl.__defaults__,
    _kn_mla_v4_sparse_decode_mfma_impl.__closure__,
)
functools.update_wrapper(
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline_source,
    _kn_mla_v4_sparse_decode_mfma_impl,
)


class _ForgeUnregisteredKernel:
    """Fail-closed placeholder for a heavy kernel omitted by selected registration."""

    __slots__ = (
        "registered_symbol",
        "registered_variant",
        "requested_symbol",
        "requested_variant",
    )
    __forge_unregistered_kernel__ = True

    def __init__(
        self,
        requested_variant,
        requested_symbol,
        registered_variant,
        registered_symbol,
    ):
        self.requested_variant = requested_variant
        self.requested_symbol = requested_symbol
        self.registered_variant = registered_variant
        self.registered_symbol = registered_symbol

    def __call__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "attempted to use a Forge heavy kernel omitted by selected registration: "
            f"requested_variant={self.requested_variant!r}, "
            f"requested_symbol={self.requested_symbol!r}, "
            f"registered_variant={self.registered_variant!r}, "
            f"registered_symbol={self.registered_symbol!r}"
        )

    def __repr__(self):
        return (
            f"<{type(self).__name__} requested_variant={self.requested_variant!r} "
            f"requested_symbol={self.requested_symbol!r} "
            f"registered_variant={self.registered_variant!r} "
            f"registered_symbol={self.registered_symbol!r}>"
        )


_forge_registered_heavy_count = 0


def _register_forge_heavy_kernel(registration_tag, source, symbol, block):
    global _forge_registered_heavy_count

    if not _FORGE_SELECTED_REGISTRATION or registration_tag == FORGE_VARIANT:
        _forge_registered_heavy_count += 1
        return flyc.kernel(
            source,
            name=symbol,
            known_block_size=block,
        )

    _, registered_symbol, _ = _FORGE_CANONICAL_REGISTRATIONS[FORGE_VARIANT]
    return _ForgeUnregisteredKernel(
        registration_tag,
        symbol,
        FORGE_VARIANT,
        registered_symbol,
    )


kn_mla_v4_sparse_decode_mfma = _register_forge_heavy_kernel(
    "default",
    _kn_mla_v4_sparse_decode_mfma_impl,
    "kn_mla_v4_sparse_decode_mfma",
    [NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128 = _register_forge_heavy_kernel(
    "n128",
    _kn_mla_v4_sparse_decode_mfma_n128_source,
    "kn_mla_v4_sparse_decode_mfma_n128",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4 = _register_forge_heavy_kernel(
    "token4",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv = _register_forge_heavy_kernel(
    "m5",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk = _register_forge_heavy_kernel(
    "m5b",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode = _register_forge_heavy_kernel(
    "m7",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill = _register_forge_heavy_kernel(
    "m9",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead = _register_forge_heavy_kernel(
    "m10",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax = _register_forge_heavy_kernel(
    "m12l",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder = _register_forge_heavy_kernel(
    "m15",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride = _register_forge_heavy_kernel(
    "m16s536",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr = _register_forge_heavy_kernel(
    "m6-progressive-vgpr",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l = _register_forge_heavy_kernel(
    "m6-raw-g2l",
    _kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l_source,
    "kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_n128_split4 = _register_forge_heavy_kernel(
    "n128-split4",
    _kn_mla_v4_sparse_decode_mfma_n128_split4_source,
    "kn_mla_v4_sparse_decode_mfma_n128_split4",
    [N256_SINGLE_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_qh64_b32 = _register_forge_heavy_kernel(
    "qh64-b32",
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_source,
    "kn_mla_v4_sparse_decode_mfma_qh64_b32",
    [QH64_B32_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_qh64_b32_m1 = _register_forge_heavy_kernel(
    "qh64-b32-m1",
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m1_source,
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m1",
    [QH64_B32_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode = _register_forge_heavy_kernel(
    "m8",
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode_source,
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode",
    [QH64_B32_NUM_THREADS, 1, 1],
)
kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline = _register_forge_heavy_kernel(
    "m11",
    _kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline_source,
    "kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline",
    [QH64_B32_NUM_THREADS, 1, 1],
)

_forge_kernel_global, _forge_kernel_symbol, _forge_kernel_block = _FORGE_CANONICAL_REGISTRATIONS[FORGE_VARIANT]
_forge_expected_registration_count = 1 if _FORGE_SELECTED_REGISTRATION else len(FORGE_HEAVY_KERNEL_GLOBALS)
if _forge_registered_heavy_count != _forge_expected_registration_count:
    raise RuntimeError(
        "Forge heavy-registration count mismatch: "
        f"mode={'selected' if _FORGE_SELECTED_REGISTRATION else 'all'}, "
        f"expected={_forge_expected_registration_count}, "
        f"registered={_forge_registered_heavy_count}"
    )
if _FORGE_SELECTED_REGISTRATION and getattr(
    globals()[_forge_kernel_global],
    "__forge_unregistered_kernel__",
    False,
):
    raise RuntimeError(
        f"selected Forge kernel {_forge_kernel_global!r} was not registered for FORGE_VARIANT={FORGE_VARIANT!r}"
    )

FORGE_REGISTRATION_IDENTITY = types.MappingProxyType(
    {
        "schema": FORGE_REGISTRATION_SCHEMA,
        "mode": "selected" if _FORGE_SELECTED_REGISTRATION else "all",
        "variant": FORGE_VARIANT,
        "kernel_global": _forge_kernel_global,
        "symbol": _forge_kernel_symbol,
        "block": _forge_kernel_block,
        "registered_heavy_count": _forge_registered_heavy_count,
    }
)


@flyc.jit
def launch_mla_v4_sparse_decode(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    grid_splits: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
    use_n128: fx.Constexpr[bool] = False,
    use_n128_split4: fx.Constexpr[bool] = False,
    use_n128_token4: fx.Constexpr[bool] = False,
):
    """Launch the direct ``(sequence, split)`` stage-1 grid."""
    if const_expr(use_n128_token4):
        kn_mla_v4_sparse_decode_mfma_n128_token4(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            softmax_scale,
            N256_SINGLE_BLOCK_N,
            N256_SINGLE_NUM_WARPS,
            False,
            True,
            False,
            False,
            value_attrs={
                "passthrough": [
                    [
                        "denormal-fp-math-f32",
                        "preserve-sign,preserve-sign",
                    ],
                    ["no-nans-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(num_sequences, grid_splits, 1),
            block=(N256_SINGLE_NUM_THREADS, 1, 1),
            stream=stream,
        )
    elif const_expr(use_n128_split4):
        kn_mla_v4_sparse_decode_mfma_n128_split4(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            softmax_scale,
            N256_SINGLE_BLOCK_N,
            N256_SINGLE_NUM_WARPS,
            True,
            False,
            False,
            False,
            value_attrs={
                "passthrough": [
                    [
                        "denormal-fp-math-f32",
                        "preserve-sign,preserve-sign",
                    ],
                    ["no-nans-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(num_sequences, grid_splits, 1),
            block=(N256_SINGLE_NUM_THREADS, 1, 1),
            stream=stream,
        )
    elif const_expr(use_n128):
        kn_mla_v4_sparse_decode_mfma_n128(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            softmax_scale,
            N256_SINGLE_BLOCK_N,
            N256_SINGLE_NUM_WARPS,
            False,
            False,
            False,
            False,
            value_attrs={
                "passthrough": [
                    [
                        "denormal-fp-math-f32",
                        "preserve-sign,preserve-sign",
                    ],
                    ["no-nans-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(num_sequences, grid_splits, 1),
            block=(N256_SINGLE_NUM_THREADS, 1, 1),
            stream=stream,
        )
    else:
        kn_mla_v4_sparse_decode_mfma(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            softmax_scale,
            BLOCK_N,
            NUM_WARPS,
            False,
            False,
            False,
            False,
            value_attrs={
                "passthrough": [
                    [
                        "denormal-fp-math-f32",
                        "preserve-sign,preserve-sign",
                    ],
                    ["no-nans-fp-math", "true"],
                ],
            },
        ).launch(
            grid=(num_sequences, grid_splits, 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )


# Keep enough allocator headroom for the 32-value Q slice and the 32-value
# online output accumulator while targeting at least two resident workgroups.
launch_mla_v4_sparse_decode.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m5_wide_pv_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 token4 M5 with native wide PV LDS reads."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m5_wide_pv(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m5_wide_pv_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m5b_wave_local_qk_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 token4 M5b with wave-local fill-to-QK sync."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m5b_wave_local_qk(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m5b_wave_local_qk_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m7_native_decode_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M7 with native scaled FP8-to-BF16 decode."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m7_native_decode(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m7_native_decode_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m9_batched_fill_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M9 with fully batched token4 KV loads."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m9_batched_fill(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m9_batched_fill_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m10_lookahead_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M10 with one-tile VGPR lookahead across PV."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m10_lookahead(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m10_lookahead_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m12l_log2_softmax_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M12-L: M10 topology with log2 softmax state."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m12l_log2_softmax(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m12l_log2_softmax_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m15_opaque_payload_ladder_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M15 with an opaque age-counted payload queue."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m15_opaque_payload_ladder(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m15_opaque_payload_ladder_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M16 with a 536-BF16 decoded-KV LDS row stride."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m16s536_lds_stride(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        M16S536_KV_LDS_STRIDE,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m6_progressive_vgpr_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M6 with pair-progressive VGPR KV prefetch."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m6_progressive_vgpr(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m6_progressive_vgpr_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_n128_token4_m6_raw_g2l_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 M6 with pair-progressive raw buffer-to-LDS."""
    kn_mla_v4_sparse_decode_mfma_n128_token4_m6_raw_g2l(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        N256_SINGLE_BLOCK_N,
        N256_SINGLE_NUM_WARPS,
        False,
        True,
        False,
        True,
        True,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(N256_SINGLE_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_n128_token4_m6_raw_g2l_diagnostic.compile_hints = {
    "maxnreg": 128,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_qh64_b32_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch the exact-N256 diagnostic qh64/B32 M0 symbol."""
    kn_mla_v4_sparse_decode_mfma_qh64_b32(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        QH64_B32_BLOCK_N,
        QH64_B32_NUM_WARPS,
        False,
        False,
        True,
        False,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(QH64_B32_NUM_THREADS, 1, 1),
        stream=stream,
    )


# M0 intentionally carries 32 independent vec4 FP32 output accumulators.
# Keep it isolated from the accepted launcher's maxnreg=128 contract.
launch_mla_v4_sparse_decode_qh64_b32_diagnostic.compile_hints = {
    "maxnreg": 256,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_qh64_b32_m1_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 qh64/B32 M1 with native LDS transpose reads."""
    kn_mla_v4_sparse_decode_mfma_qh64_b32_m1(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        QH64_B32_BLOCK_N,
        QH64_B32_NUM_WARPS,
        False,
        False,
        True,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(QH64_B32_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_qh64_b32_m1_diagnostic.compile_hints = {
    "maxnreg": 256,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_qh64_b32_m8_native_decode_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch qh64/B32 M8 with wide PV and native scaled FP8 decode."""
    kn_mla_v4_sparse_decode_mfma_qh64_b32_m8_native_decode(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        QH64_B32_BLOCK_N,
        QH64_B32_NUM_WARPS,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(QH64_B32_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_qh64_b32_m8_native_decode_diagnostic.compile_hints = {
    "maxnreg": 256,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@flyc.jit
def launch_mla_v4_sparse_decode_qh64_b32_m11_raw_pipeline_diagnostic(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    qo_indptr: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    split_indptr: fx.Tensor,
    output: fx.Tensor,
    logits: fx.Tensor,
    attn_lse: fx.Tensor,
    num_kv_splits: fx.Int32,
    softmax_scale: fx.Float32,
    num_sequences: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    """Launch exact-N256 qh64/B32 M11 with raw4/decoded2 pipelining."""
    kn_mla_v4_sparse_decode_mfma_qh64_b32_m11_raw_pipeline(
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        output,
        logits,
        attn_lse,
        num_kv_splits,
        softmax_scale,
        QH64_B32_BLOCK_N,
        QH64_B32_NUM_WARPS,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        value_attrs={
            "passthrough": [
                [
                    "denormal-fp-math-f32",
                    "preserve-sign,preserve-sign",
                ],
                ["no-nans-fp-math", "true"],
            ],
        },
    ).launch(
        grid=(num_sequences, 1, 1),
        block=(QH64_B32_NUM_THREADS, 1, 1),
        stream=stream,
    )


launch_mla_v4_sparse_decode_qh64_b32_m11_raw_pipeline_diagnostic.compile_hints = {
    "maxnreg": 256,
    "fast_fp_math": False,
    "unsafe_fp_math": False,
}


@triton.jit
def _reduce_v4_sparse_splits(
    logits,
    attn_lse,
    output,
    kv_indptr,
    split_indptr,
    num_kv_splits: tl.constexpr,
    num_heads: tl.constexpr,
    value_dim: tl.constexpr,
    block_n: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """One-warp LSE merge, matching Aiter stage2 with empty-row support."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, BLOCK_DV)
    dim_mask = dims < value_dim

    kv_start = tl.load(kv_indptr + seq)
    kv_end = tl.load(kv_indptr + seq + 1)
    kv_len = kv_end - kv_start
    configured_splits = tl.load(split_indptr + seq + 1) - tl.load(split_indptr + seq)
    valid_from_kv = tl.maximum(1, tl.cdiv(kv_len, block_n))
    valid_splits = tl.minimum(
        num_kv_splits,
        tl.minimum(configured_splits, valid_from_kv),
    )

    running_max = -float("inf")
    running_sum = 0.0
    accumulator = tl.zeros((BLOCK_DV,), dtype=tl.float32)

    for split in range(0, valid_splits):
        partial_record = (seq * num_kv_splits + split) * num_heads + head
        partial = tl.load(
            logits + partial_record * value_dim + dims,
            mask=dim_mask,
            other=0.0,
        )
        partial_lse = tl.load(attn_lse + partial_record)
        new_max = tl.maximum(running_max, partial_lse)
        old_scale = tl.exp(running_max - new_max)
        partial_scale = tl.exp(partial_lse - new_max)
        accumulator = accumulator * old_scale + partial * partial_scale
        running_sum = running_sum * old_scale + partial_scale
        running_max = new_max

    final = accumulator / running_sum
    output_record = seq * num_heads + head
    tl.store(
        output + output_record * value_dim + dims,
        final,
        mask=dim_mask,
    )


def _check_tensor(name, tensor, *, dtype, shape=None):
    if tensor.dtype != dtype:
        raise ValueError(f"{name}: expected dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name}: expected contiguous storage, got stride={tensor.stride()}")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name}: expected shape {tuple(shape)}, got {tuple(tensor.shape)}")


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
    """KernelForge-compatible DeepSeek-V4 sparse decode entry point.

    All tensors are caller-owned.  The function writes ``output`` in place and
    includes the split merge before returning.
    """
    if not isinstance(num_kv_splits, int) or num_kv_splits <= 0:
        raise ValueError(f"num_kv_splits must be a positive int, got {num_kv_splits!r}")

    device = q_packed.device
    if device.type != "cuda":
        raise ValueError(f"q_packed must be on an AMD GPU, got {device}")
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    if not arch.startswith("gfx950"):
        raise NotImplementedError(f"DeepSeek-V4 sparse decode currently requires gfx950, got {arch}")

    if q_packed.ndim != 3:
        raise ValueError("q_packed must be [num_sequences, 16, 512]")
    num_sequences = q_packed.size(0)
    _check_tensor(
        "q_packed",
        q_packed,
        dtype=torch.float8_e4m3fn,
        shape=(num_sequences, NUM_HEADS, DIM_PACKED),
    )
    _check_tensor(
        "q_rope",
        q_rope,
        dtype=torch.bfloat16,
        shape=(num_sequences, NUM_HEADS, DIM_ROPE),
    )

    if kv_packed.ndim != 4:
        raise ValueError("kv_packed must be [num_physical_rows, 1, 1, 512]")
    num_physical_rows = kv_packed.size(0)
    _check_tensor(
        "kv_packed",
        kv_packed,
        dtype=torch.float8_e4m3fn,
        shape=(num_physical_rows, PAGE_SIZE, NUM_KV_HEADS, DIM_PACKED),
    )
    _check_tensor(
        "kv_rope",
        kv_rope,
        dtype=torch.bfloat16,
        shape=(num_physical_rows, PAGE_SIZE, NUM_KV_HEADS, DIM_ROPE),
    )

    _check_tensor(
        "output",
        output,
        dtype=torch.bfloat16,
        shape=(num_sequences, NUM_HEADS, V_HEAD_DIM),
    )
    _check_tensor(
        "logits",
        logits,
        dtype=torch.float32,
        shape=(
            num_sequences,
            num_kv_splits,
            NUM_HEADS,
            V_HEAD_DIM,
        ),
    )
    _check_tensor(
        "attn_lse",
        attn_lse,
        dtype=torch.float32,
        shape=(num_sequences, num_kv_splits, NUM_HEADS, 1),
    )
    _check_tensor(
        "qo_indptr",
        qo_indptr,
        dtype=torch.int32,
        shape=(num_sequences + 1,),
    )
    _check_tensor(
        "kv_indptr",
        kv_indptr,
        dtype=torch.int32,
        shape=(num_sequences + 1,),
    )
    _check_tensor(
        "split_indptr",
        split_indptr,
        dtype=torch.int32,
        shape=(num_sequences + 1,),
    )
    _check_tensor(
        "kv_indices",
        kv_indices,
        dtype=torch.int32,
    )
    _check_tensor(
        "sink",
        sink,
        dtype=torch.float32,
        shape=(NUM_HEADS,),
    )

    tensors = (
        q_rope,
        kv_packed,
        kv_rope,
        output,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sink,
        split_indptr,
        logits,
        attn_lse,
    )
    for tensor in tensors:
        if tensor.device != device:
            raise ValueError(f"all tensors must be on {device}; got {tensor.device}")

    # Token-stationary fill is the exact-N256 release default.  Explicit
    # ``serial`` remains a legacy, non-release diagnostic control only: an
    # isolated post-promotion run failed its numerical gate despite byte-for-
    # byte identical ISA to the pre-token accepted snapshot.  The split4
    # experiment remains gated by the exact uniform-L1152 logical row count.
    n256_shape = num_sequences == 256 and num_kv_splits == 1
    qh64_diagnostic_value = (
        os.environ.get(
            "FLYDSL_V4_N256_QH64_B32_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not qh64_diagnostic_value:
        qh64_diagnostic_value = "0"
    if qh64_diagnostic_value not in {"0", "1"}:
        raise ValueError(f"FLYDSL_V4_N256_QH64_B32_DIAGNOSTIC must be '0' or '1'; got {qh64_diagnostic_value!r}")
    use_qh64_b32_diagnostic = qh64_diagnostic_value == "1"
    qh64_m1_diagnostic_value = (
        os.environ.get(
            "FLYDSL_V4_N256_QH64_B32_M1_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not qh64_m1_diagnostic_value:
        qh64_m1_diagnostic_value = "0"
    if qh64_m1_diagnostic_value not in {"0", "1"}:
        raise ValueError(f"FLYDSL_V4_N256_QH64_B32_M1_DIAGNOSTIC must be '0' or '1'; got {qh64_m1_diagnostic_value!r}")
    use_qh64_b32_m1_diagnostic = qh64_m1_diagnostic_value == "1"
    qh64_m8_native_decode_value = (
        os.environ.get(
            "FLYDSL_V4_N256_QH64_B32_M8_NATIVE_DECODE_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not qh64_m8_native_decode_value:
        qh64_m8_native_decode_value = "0"
    if qh64_m8_native_decode_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_QH64_B32_M8_NATIVE_DECODE_DIAGNOSTIC "
            f"must be '0' or '1'; got {qh64_m8_native_decode_value!r}"
        )
    use_qh64_b32_m8_native_decode = qh64_m8_native_decode_value == "1"
    qh64_m11_raw_pipeline_value = (
        os.environ.get(
            "FLYDSL_V4_N256_QH64_B32_M11_RAW_PIPELINE_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not qh64_m11_raw_pipeline_value:
        qh64_m11_raw_pipeline_value = "0"
    if qh64_m11_raw_pipeline_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_QH64_B32_M11_RAW_PIPELINE_DIAGNOSTIC "
            f"must be '0' or '1'; got {qh64_m11_raw_pipeline_value!r}"
        )
    use_qh64_b32_m11_raw_pipeline = qh64_m11_raw_pipeline_value == "1"
    token4_m5_wide_pv_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M5_WIDE_PV_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m5_wide_pv_value:
        token4_m5_wide_pv_value = "0"
    if token4_m5_wide_pv_value not in {"0", "1"}:
        raise ValueError(
            f"FLYDSL_V4_N256_TOKEN4_M5_WIDE_PV_DIAGNOSTIC must be '0' or '1'; got {token4_m5_wide_pv_value!r}"
        )
    use_token4_m5_wide_pv = token4_m5_wide_pv_value == "1"
    token4_m5b_wave_local_qk_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M5B_WAVE_LOCAL_QK_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m5b_wave_local_qk_value:
        token4_m5b_wave_local_qk_value = "0"
    if token4_m5b_wave_local_qk_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M5B_WAVE_LOCAL_QK_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m5b_wave_local_qk_value!r}"
        )
    use_token4_m5b_wave_local_qk = token4_m5b_wave_local_qk_value == "1"
    token4_m7_native_decode_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M7_NATIVE_DECODE_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m7_native_decode_value:
        token4_m7_native_decode_value = "0"
    if token4_m7_native_decode_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M7_NATIVE_DECODE_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m7_native_decode_value!r}"
        )
    use_token4_m7_native_decode = token4_m7_native_decode_value == "1"
    token4_m9_batched_fill_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M9_BATCHED_FILL_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m9_batched_fill_value:
        token4_m9_batched_fill_value = "0"
    if token4_m9_batched_fill_value not in {"0", "1"}:
        raise ValueError(
            f"FLYDSL_V4_N256_TOKEN4_M9_BATCHED_FILL_DIAGNOSTIC must be '0' or '1'; got {token4_m9_batched_fill_value!r}"
        )
    use_token4_m9_batched_fill = token4_m9_batched_fill_value == "1"
    token4_m10_lookahead_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M10_LOOKAHEAD_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m10_lookahead_value:
        token4_m10_lookahead_value = "0"
    if token4_m10_lookahead_value not in {"0", "1"}:
        raise ValueError(
            f"FLYDSL_V4_N256_TOKEN4_M10_LOOKAHEAD_DIAGNOSTIC must be '0' or '1'; got {token4_m10_lookahead_value!r}"
        )
    use_token4_m10_lookahead = token4_m10_lookahead_value == "1"
    token4_m12l_log2_softmax_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M12L_LOG2_SOFTMAX_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m12l_log2_softmax_value:
        token4_m12l_log2_softmax_value = "0"
    if token4_m12l_log2_softmax_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M12L_LOG2_SOFTMAX_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m12l_log2_softmax_value!r}"
        )
    use_token4_m12l_log2_softmax = token4_m12l_log2_softmax_value == "1"
    token4_m15_opaque_payload_ladder_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M15_OPAQUE_PAYLOAD_LADDER_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m15_opaque_payload_ladder_value:
        token4_m15_opaque_payload_ladder_value = "0"
    if token4_m15_opaque_payload_ladder_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M15_OPAQUE_PAYLOAD_LADDER_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m15_opaque_payload_ladder_value!r}"
        )
    use_token4_m15_opaque_payload_ladder = token4_m15_opaque_payload_ladder_value == "1"
    token4_m16s536_lds_stride_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M16S536_LDS_STRIDE_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m16s536_lds_stride_value:
        token4_m16s536_lds_stride_value = "0"
    if token4_m16s536_lds_stride_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M16S536_LDS_STRIDE_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m16s536_lds_stride_value!r}"
        )
    use_token4_m16s536_lds_stride = token4_m16s536_lds_stride_value == "1"
    token4_m6_progressive_vgpr_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M6_PROGRESSIVE_VGPR_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m6_progressive_vgpr_value:
        token4_m6_progressive_vgpr_value = "0"
    if token4_m6_progressive_vgpr_value not in {"0", "1"}:
        raise ValueError(
            "FLYDSL_V4_N256_TOKEN4_M6_PROGRESSIVE_VGPR_DIAGNOSTIC "
            f"must be '0' or '1'; got {token4_m6_progressive_vgpr_value!r}"
        )
    use_token4_m6_progressive_vgpr = token4_m6_progressive_vgpr_value == "1"
    token4_m6_raw_g2l_value = (
        os.environ.get(
            "FLYDSL_V4_N256_TOKEN4_M6_RAW_G2L_DIAGNOSTIC",
            "0",
        )
        .strip()
        .lower()
    )
    if not token4_m6_raw_g2l_value:
        token4_m6_raw_g2l_value = "0"
    if token4_m6_raw_g2l_value not in {"0", "1"}:
        raise ValueError(
            f"FLYDSL_V4_N256_TOKEN4_M6_RAW_G2L_DIAGNOSTIC must be '0' or '1'; got {token4_m6_raw_g2l_value!r}"
        )
    use_token4_m6_raw_g2l = token4_m6_raw_g2l_value == "1"
    enabled_exact_n256_diagnostics = sum(
        (
            use_qh64_b32_diagnostic,
            use_qh64_b32_m1_diagnostic,
            use_qh64_b32_m8_native_decode,
            use_qh64_b32_m11_raw_pipeline,
            use_token4_m5_wide_pv,
            use_token4_m5b_wave_local_qk,
            use_token4_m7_native_decode,
            use_token4_m9_batched_fill,
            use_token4_m10_lookahead,
            use_token4_m12l_log2_softmax,
            use_token4_m15_opaque_payload_ladder,
            use_token4_m16s536_lds_stride,
            use_token4_m6_progressive_vgpr,
            use_token4_m6_raw_g2l,
        )
    )
    if enabled_exact_n256_diagnostics > 1:
        raise ValueError("exact-N256 serving diagnostics are mutually exclusive")
    exact_n256_l1152_split1 = n256_shape and kv_indices.numel() == 256 * 1152
    if enabled_exact_n256_diagnostics and not exact_n256_l1152_split1:
        raise ValueError("serving diagnostics are restricted to exact N256/L1152/split1")
    n64_split4_shape = num_sequences == 64 and num_kv_splits == 4 and kv_indices.numel() == 64 * 1152
    n256_fill_variant = (
        os.environ.get(
            "FLYDSL_V4_N256_FILL_VARIANT",
            "token4",
        )
        .strip()
        .lower()
    )
    if not n256_fill_variant:
        n256_fill_variant = "token4"
    if n256_fill_variant not in {"serial", "token4"}:
        raise ValueError(f"FLYDSL_V4_N256_FILL_VARIANT must be 'serial' or 'token4'; got {n256_fill_variant!r}")
    if (
        use_token4_m5_wide_pv
        or use_token4_m5b_wave_local_qk
        or use_token4_m7_native_decode
        or use_token4_m9_batched_fill
        or use_token4_m10_lookahead
        or use_token4_m12l_log2_softmax
        or use_token4_m15_opaque_payload_ladder
        or use_token4_m16s536_lds_stride
        or use_token4_m6_progressive_vgpr
        or use_token4_m6_raw_g2l
    ) and n256_fill_variant != "token4":
        raise ValueError("token4 M5/M6/M7/M9/M10/M12-L/M15/M16 diagnostics require FLYDSL_V4_N256_FILL_VARIANT=token4")
    use_n128 = n256_shape and n256_fill_variant == "serial"
    use_n128_token4 = n256_shape and n256_fill_variant == "token4"
    use_n128_split4 = False
    if n64_split4_shape:
        n64_split4_variant = (
            os.environ.get(
                "FLYDSL_V4_N64_SPLIT4_VARIANT",
                "n128",
            )
            .strip()
            .lower()
        )
        if n64_split4_variant not in {"n128", "b64"}:
            raise ValueError(f"FLYDSL_V4_N64_SPLIT4_VARIANT must be 'n128' or 'b64'; got {n64_split4_variant!r}")
        use_n128_split4 = n64_split4_variant == "n128"
    if use_qh64_b32_m11_raw_pipeline:
        launch_mla_v4_sparse_decode_qh64_b32_m11_raw_pipeline_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m16s536_lds_stride:
        launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m15_opaque_payload_ladder:
        launch_mla_v4_sparse_decode_n128_token4_m15_opaque_payload_ladder_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m12l_log2_softmax:
        launch_mla_v4_sparse_decode_n128_token4_m12l_log2_softmax_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m10_lookahead:
        launch_mla_v4_sparse_decode_n128_token4_m10_lookahead_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m9_batched_fill:
        launch_mla_v4_sparse_decode_n128_token4_m9_batched_fill_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m7_native_decode:
        launch_mla_v4_sparse_decode_n128_token4_m7_native_decode_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m6_raw_g2l:
        launch_mla_v4_sparse_decode_n128_token4_m6_raw_g2l_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m6_progressive_vgpr:
        launch_mla_v4_sparse_decode_n128_token4_m6_progressive_vgpr_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m5b_wave_local_qk:
        launch_mla_v4_sparse_decode_n128_token4_m5b_wave_local_qk_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_token4_m5_wide_pv:
        launch_mla_v4_sparse_decode_n128_token4_m5_wide_pv_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_qh64_b32_m8_native_decode:
        launch_mla_v4_sparse_decode_qh64_b32_m8_native_decode_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_qh64_b32_m1_diagnostic:
        launch_mla_v4_sparse_decode_qh64_b32_m1_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    elif use_qh64_b32_diagnostic:
        launch_mla_v4_sparse_decode_qh64_b32_diagnostic(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            stream=torch.cuda.current_stream(device),
        )
    else:
        launch_mla_v4_sparse_decode(
            q_packed,
            q_rope,
            kv_packed,
            kv_rope,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sink,
            split_indptr,
            output,
            logits,
            attn_lse,
            num_kv_splits,
            float(softmax_scale),
            num_sequences=num_sequences,
            grid_splits=num_kv_splits,
            use_n128=use_n128,
            use_n128_split4=use_n128_split4,
            use_n128_token4=use_n128_token4,
            stream=torch.cuda.current_stream(device),
        )

    # Diagnostic-only stream-ordering fence.  This is intentionally off by
    # default and must never be enabled for reported performance numbers.
    if os.environ.get("FLYDSL_V4_DIAGNOSTIC_SYNC_STAGE1") == "1":
        torch.cuda.synchronize(device)

    if num_kv_splits > 1:
        reducer_block_n = N256_SINGLE_BLOCK_N if use_n128_split4 else BLOCK_N
        _reduce_v4_sparse_splits[(num_sequences, NUM_HEADS)](
            logits,
            attn_lse,
            output,
            kv_indptr,
            split_indptr,
            num_kv_splits=num_kv_splits,
            num_heads=NUM_HEADS,
            value_dim=V_HEAD_DIM,
            block_n=reducer_block_n,
            BLOCK_DV=triton.next_power_of_2(V_HEAD_DIM),
            num_warps=1,
            num_stages=2,
            waves_per_eu=4,
        )


__all__ = [
    "BLOCK_N",
    "FORGE_EXPECTED_VARIANT_ENV",
    "FORGE_HEAVY_KERNEL_GLOBALS",
    "FORGE_REGISTRATION_CONTRACT",
    "FORGE_REGISTRATION_IDENTITY",
    "FORGE_REGISTRATION_SCHEMA",
    "FORGE_SELECTED_REGISTRATION_ENV",
    "FORGE_VARIANT",
    "NUM_HEADS",
    "NUM_THREADS",
    "OCCUPANCY",
    "flydsl_mla_v4_sparse_decode",
    "launch_mla_v4_sparse_decode",
    "launch_mla_v4_sparse_decode_n128_token4_m5_wide_pv_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m5b_wave_local_qk_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m7_native_decode_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m9_batched_fill_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m10_lookahead_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m12l_log2_softmax_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m15_opaque_payload_ladder_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m16s536_lds_stride_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m6_progressive_vgpr_diagnostic",
    "launch_mla_v4_sparse_decode_n128_token4_m6_raw_g2l_diagnostic",
    "launch_mla_v4_sparse_decode_qh64_b32_diagnostic",
    "launch_mla_v4_sparse_decode_qh64_b32_m1_diagnostic",
    "launch_mla_v4_sparse_decode_qh64_b32_m8_native_decode_diagnostic",
    "launch_mla_v4_sparse_decode_qh64_b32_m11_raw_pipeline_diagnostic",
]
