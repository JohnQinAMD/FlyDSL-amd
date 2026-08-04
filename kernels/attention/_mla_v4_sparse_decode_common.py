# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared traits for the DSV4 M16 sparse-decode kernel."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue

# ---------------------------------------------------------------------------
# DeepSeek-V4 sparse MLA shape
# ---------------------------------------------------------------------------

NUM_HEADS: int = 16
NUM_KV_HEADS: int = 1
DIM_NOPE: int = 448
DIM_ROPE: int = 64
DIM_QK: int = DIM_NOPE + DIM_ROPE
DIM_PACKED: int = 512
V_HEAD_DIM: int = 512
PAGE_SIZE: int = 1
WARP_SIZE: int = 64
NUM_WARPS: int = 8
NUM_THREADS: int = 512
TOKENS_PER_WAVE: int = 16
BLOCK_N: int = 128
OCCUPANCY: int = 1
PACKED_DWORDS: int = DIM_PACKED // 4
NOPE_SCALE_OFFSET: int = DIM_NOPE
MFMA_K: int = 32
MFMA_INPUT_VALUES: int = 8
MFMA_OUTPUT_VALUES: int = 4


# ---------------------------------------------------------------------------
# M16/S536 tile layout
# ---------------------------------------------------------------------------

# The promoted shape keeps one decoded BF16 KV tile in LDS and uses the same
# rows for QK and PV. 536 = 512 payload columns + 24 BF16 pad columns; the pad
# de-phases ds_read_tr16_b64 PV reads enough to avoid the worst bank conflicts.
M16S536_KV_LDS_STRIDE: int = 536
M16S536_DYNAMIC_LDS_BYTES: int = 142_848
NOPE_DWORDX4_LOADS: int = DIM_NOPE // (4 * 2 * MFMA_INPUT_VALUES)
ROPE_VEC8_LOADS: int = DIM_ROPE // (4 * MFMA_INPUT_VALUES)
LOG2E: float = 1.4426950408889634
assert DIM_QK == V_HEAD_DIM == 512
assert NUM_THREADS == NUM_WARPS * WARP_SIZE
assert BLOCK_N == NUM_WARPS * TOKENS_PER_WAVE
assert DIM_QK % MFMA_K == 0
assert BLOCK_N == 128
assert NUM_THREADS == 512
assert NOPE_DWORDX4_LOADS == 7
assert ROPE_VEC8_LOADS == 2
assert M16S536_KV_LDS_STRIDE % MFMA_INPUT_VALUES == 0
assert (
    BLOCK_N * M16S536_KV_LDS_STRIDE * 2
    + NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES * 2
    + 2 * NUM_WARPS * NUM_HEADS * 4
    + BLOCK_N * 4
    == M16S536_DYNAMIC_LDS_BYTES
)


# ---------------------------------------------------------------------------
# LDS storage
# ---------------------------------------------------------------------------


@fx.struct
class SparseDecodeSharedN128M16S536:
    """N128 workspace with a 24-BF16 bank-breaking decoded-KV row pad."""

    kv: fx.Array[fx.BFloat16, BLOCK_N * M16S536_KV_LDS_STRIDE, 16]
    probability: fx.Array[fx.BFloat16, NUM_WARPS * WARP_SIZE * MFMA_OUTPUT_VALUES, 16]
    reduction: fx.Array[fx.Float32, 2 * NUM_WARPS * NUM_HEADS, 16]
    physical_rows: fx.Array[fx.Int32, BLOCK_N, 16]


# ---------------------------------------------------------------------------
# Tiny DSL helpers
# ---------------------------------------------------------------------------


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
    return _raw(ArithValue(_raw(cond)).select(_raw(true_value), _raw(false_value)))


def _fast_exp(value):
    """Natural exponential through the gfx950 base-2 instruction."""
    scaled = arith.mulf(_raw(value), _raw(fx.Float32(LOG2E)), fastmath=arith.FastMathFlags.fast)
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
