# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Promoted DSV4 M16 S536 sparse-decode FlyDSL kernel body."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from kernels.attention._mla_v4_sparse_decode_common import (
    BLOCK_N,
    DIM_NOPE,
    DIM_PACKED,
    DIM_QK,
    DIM_ROPE,
    M16S536_KV_LDS_STRIDE,
    MFMA_INPUT_VALUES,
    MFMA_K,
    MFMA_OUTPUT_VALUES,
    NOPE_DWORDX4_LOADS,
    NOPE_SCALE_OFFSET,
    NUM_HEADS,
    NUM_WARPS,
    PACKED_DWORDS,
    ROPE_VEC8_LOADS,
    TOKENS_PER_WAVE,
    V_HEAD_DIM,
    WARP_SIZE,
    SparseDecodeSharedN128M16S536,
    _e8m0_to_f32,
    _fast_exp,
    _i32,
    _idx,
    _select,
)
from kernels.common import dpp_utils


def _kn_mla_v4_sparse_decode_mfma_impl(
    q_packed: fx.Tensor,
    q_rope: fx.Tensor,
    kv_packed: fx.Tensor,
    kv_rope: fx.Tensor,
    kv_indptr: fx.Tensor,
    kv_indices: fx.Tensor,
    sink: fx.Tensor,
    output: fx.Tensor,
    softmax_scale: fx.Float32,
):
    """Promoted M16 S536 sparse-decode stage-1 kernel."""
    block_n = BLOCK_N
    num_warps = NUM_WARPS
    kv_lds_stride = M16S536_KV_LDS_STRIDE
    p_tile_elems = num_warps * WARP_SIZE * MFMA_OUTPUT_VALUES
    output_dims_per_wave = V_HEAD_DIM // num_warps
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

    def _softmax_exp_difference(value):
        """Exponentiate a difference expressed in the selected max domain."""
        return _fast_exp(value)

    # ---- thread map and buffer resources ----------------------------------

    seq_idx = gpu.block_id("x")
    tid = gpu.thread_id("x")
    seq_i32 = _i32(seq_idx)
    tid_i32 = _i32(tid)
    wave_i32 = _raw(ArithValue(tid_i32) // fx.Int32(WARP_SIZE))
    lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(WARP_SIZE))
    head_i32 = _raw(ArithValue(lane_i32) % fx.Int32(NUM_HEADS))
    lane_group_i32 = _raw(ArithValue(lane_i32) // fx.Int32(NUM_HEADS))

    q_packed_rsrc = buffer_ops.create_buffer_resource(q_packed)
    q_rope_rsrc = buffer_ops.create_buffer_resource(q_rope)
    kv_packed_rsrc = buffer_ops.create_buffer_resource(kv_packed)
    kv_rope_rsrc = buffer_ops.create_buffer_resource(kv_rope)
    kv_indptr_rsrc = buffer_ops.create_buffer_resource(kv_indptr)
    kv_indices_rsrc = buffer_ops.create_buffer_resource(kv_indices)
    sink_rsrc = buffer_ops.create_buffer_resource(sink)
    output_rsrc = buffer_ops.create_buffer_resource(output)

    # ---- LDS views and vector helper types -------------------------------

    shared = fx.SharedAllocator(static=False).allocate(SparseDecodeSharedN128M16S536).peek()
    p_lds = shared.probability.view(fx.make_layout(p_tile_elems, 1))
    red_lds = shared.reduction.view(fx.make_layout(2 * num_warps * NUM_HEADS, 1))

    class Vec8Bf16:
        ir_type = Vec.make_type(MFMA_INPUT_VALUES, fx.BFloat16)

    class Vec4Bf16:
        ir_type = Vec.make_type(MFMA_OUTPUT_VALUES, fx.BFloat16)

    class Vec2Bf16:
        ir_type = Vec.make_type(2, fx.BFloat16)

    # ---- low-level LDS and conversion helpers ---------------------------

    def _native_scaled_fp8_pair(word_i32, scale_f32, word_sel: bool):
        """Decode two scaled OCP-FP8 values directly to packed BF16."""
        word_sel_i1 = arith.constant(word_sel, type=ir.IntegerType.get_signless(1))
        raw = llvm.call_intrinsic(
            Vec2Bf16.ir_type,
            "llvm.amdgcn.cvt.scalef32.pk.bf16.fp8",
            [_raw(word_i32), _raw(scale_f32), _raw(word_sel_i1)],
            [],
            [],
        )
        return Vec(raw, (2,), fx.BFloat16)

    def _lds_ptr_at(array, byte_offset):
        base = fx.Int64(fx.ptrtoint(array.ptr))
        address = base + fx.Int64(byte_offset)
        return buffer_ops.create_llvm_ptr(address, address_space=3)

    def _lds_store_bf16x8(elem_offset_i32, value):
        llvm.StoreOp(
            _raw(value),
            _lds_ptr_at(shared.kv, _raw(ArithValue(elem_offset_i32) * fx.Int32(2))),
            alignment=16,
        )

    def _lds_load_bf16x8(elem_offset_i32):
        raw = llvm.LoadOp(
            Vec8Bf16.ir_type, _lds_ptr_at(shared.kv, _raw(ArithValue(elem_offset_i32) * fx.Int32(2))), alignment=16
        ).result
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _qh64_ds_read_tr16_bf16x4(elem_offset_i32):
        """Transpose-read four V values from the existing row-major KV tile."""
        raw = rocdl.ds_read_tr16_b64(
            Vec4Bf16.ir_type, _lds_ptr_at(shared.kv, _raw(ArithValue(elem_offset_i32) * fx.Int32(2)))
        ).result
        return Vec(raw, (MFMA_OUTPUT_VALUES,), fx.BFloat16)

    # ---- packed NoPE / RoPE fragment loaders ---------------------------

    def _decode_nope8(record_i32, dim_i32, packed_rsrc):
        """Decode one naturally aligned eight-value NoPE fragment."""
        dword_offset_i32 = _raw(
            ArithValue(record_i32) * fx.Int32(PACKED_DWORDS) + ArithValue(dim_i32).with_signedness(False) // fx.Int32(4)
        )
        raw_words = buffer_ops.buffer_load(packed_rsrc, dword_offset_i32, vec_width=2, dtype=T.i32)
        words = Vec(raw_words, (2,), fx.Int32)
        scale_byte_i8 = buffer_ops.buffer_load(
            packed_rsrc,
            _raw(
                ArithValue(record_i32) * fx.Int32(DIM_PACKED)
                + fx.Int32(NOPE_SCALE_OFFSET)
                + ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K)
            ),
            vec_width=1,
            dtype=T.i8,
        )
        scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
        decoded_words = []
        for word_idx in range_constexpr(2):
            low = _native_scaled_fp8_pair(words[word_idx], scale_f32, False)
            high = _native_scaled_fp8_pair(words[word_idx], scale_f32, True)
            decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
        return decoded_words[0].shuffle(decoded_words[1], list(range(MFMA_INPUT_VALUES)))

    def _decode_nope16_prefetched(words, scale_byte_i8):
        """Decode one prefetched NoPE dwordx4 already resident in VGPRs."""
        scale_i32 = _raw(ArithValue(scale_byte_i8).extui(T.i32))
        scale_f32 = fx.Float32(_e8m0_to_f32(scale_i32))
        decoded_words = []
        for word_idx in range_constexpr(4):
            low = _native_scaled_fp8_pair(words[word_idx], scale_f32, False)
            high = _native_scaled_fp8_pair(words[word_idx], scale_f32, True)
            decoded_words.append(low.shuffle(high, [0, 1, 2, 3]))
        return (
            decoded_words[0].shuffle(decoded_words[1], list(range(MFMA_INPUT_VALUES))),
            decoded_words[2].shuffle(decoded_words[3], list(range(MFMA_INPUT_VALUES))),
        )

    def _load_rope8(record_i32, dim_i32, rope_rsrc):
        raw = buffer_ops.buffer_load(
            rope_rsrc,
            _raw(ArithValue(record_i32) * fx.Int32(DIM_ROPE) + ArithValue(dim_i32)),
            vec_width=MFMA_INPUT_VALUES,
            dtype=T.bf16,
        )
        return Vec(raw, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _mask_bf16x8(value, valid):
        selected = arith.select(valid, _raw(value), _raw(zero_bf16x8))
        return Vec(selected, (MFMA_INPUT_VALUES,), fx.BFloat16)

    def _mfma_bf16(a, b, accumulator):
        raw = rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(MFMA_OUTPUT_VALUES, fx.Float32), [_raw(a), _raw(b), _raw(accumulator), 0, 0, 0]
        )
        rocdl.sched_mfma(1)
        return Vec(raw, (MFMA_OUTPUT_VALUES,), fx.Float32)

    # ---- prologue: sparse row and resident Q fragments -------------------

    # Runtime row mapping: one CTA owns one sequence row and all 16 query heads.
    q_row_i32 = seq_i32
    kv_start_i32 = buffer_ops.buffer_load(kv_indptr_rsrc, seq_i32, vec_width=1, is_scalar=True)
    kv_end_i32 = buffer_ops.buffer_load(
        kv_indptr_rsrc, _raw(ArithValue(seq_i32) + fx.Int32(1)), vec_width=1, is_scalar=True
    )
    split_start_i32 = kv_start_i32
    split_end_i32 = kv_end_i32
    q_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))

    # Q is kept resident in VGPRs for the whole sparse row.
    q_fragments = []
    for k_step in range_constexpr(DIM_QK // MFMA_K):
        dim_i32 = _raw(ArithValue(lane_group_i32) * fx.Int32(MFMA_INPUT_VALUES) + fx.Int32(k_step * MFMA_K))
        if k_step < DIM_NOPE // MFMA_K:
            q_fragments.append(_decode_nope8(q_record_i32, dim_i32, q_packed_rsrc))
        else:
            q_fragments.append(_load_rope8(q_record_i32, _raw(ArithValue(dim_i32) - fx.Int32(DIM_NOPE)), q_rope_rsrc))

    sink_loaded = buffer_ops.buffer_load(sink_rsrc, head_i32, vec_width=1, dtype=T.f32)
    sink_lane = sink_loaded

    # ---- M9 fill: decode current sparse tile into padded LDS -------------

    def _m9_fill_batched_tile(tile_i32):
        """Issue the complete token4 tile before native decode and LDS store."""
        token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
        fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
        logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_i32))
        valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
        safe_logical_i32 = _select(valid, logical_i32, tile_i32)
        physical_lane_i32 = fx.Int32(0).ir_value()
        if ArithValue(fragment_lane_i32) == fx.Int32(0):
            physical_lane_i32 = buffer_ops.buffer_load(kv_indices_rsrc, safe_logical_i32, vec_width=1, dtype=T.i32)
        physical_i32 = dpp_utils.update_dpp_i32(physical_lane_i32, physical_lane_i32, 0)

        packed_groups = []
        scale_bytes = []
        # Four lanes cooperate on one token; lane 0 loads the sparse row id and
        # DPP broadcasts it to the three fragment lanes.
        for load_iter in range_constexpr(NOPE_DWORDX4_LOADS):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
            dword_offset_i32 = _raw(
                ArithValue(physical_i32) * fx.Int32(PACKED_DWORDS)
                + ArithValue(dim_i32).with_signedness(False) // fx.Int32(4)
            )
            packed_groups.append(
                Vec(buffer_ops.buffer_load(kv_packed_rsrc, dword_offset_i32, vec_width=4, dtype=T.i32), (4,), fx.Int32)
            )
            scale_bytes.append(
                buffer_ops.buffer_load(
                    kv_packed_rsrc,
                    _raw(
                        ArithValue(physical_i32) * fx.Int32(DIM_PACKED)
                        + fx.Int32(NOPE_SCALE_OFFSET)
                        + ArithValue(dim_i32).with_signedness(False) // fx.Int32(MFMA_K)
                    ),
                    vec_width=1,
                    dtype=T.i8,
                )
            )

        rope_groups = []
        for load_iter in range_constexpr(ROPE_VEC8_LOADS):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            rope_dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
            rope_groups.append(_load_rope8(_raw(physical_i32), rope_dim_i32, kv_rope_rsrc))

        for load_iter in range_constexpr(NOPE_DWORDX4_LOADS):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
            decoded_fragments = _decode_nope16_prefetched(packed_groups[load_iter], scale_bytes[load_iter])
            for fragment_idx in range_constexpr(2):
                decoded = _mask_bf16x8(decoded_fragments[fragment_idx], valid)
                lds_offset_i32 = _raw(
                    ArithValue(token_i32) * fx.Int32(kv_lds_stride)
                    + ArithValue(dim_i32)
                    + fx.Int32(fragment_idx * MFMA_INPUT_VALUES)
                )
                _lds_store_bf16x8(lds_offset_i32, decoded)

        for load_iter in range_constexpr(ROPE_VEC8_LOADS):
            dim_group_i32 = _raw(ArithValue(fragment_lane_i32) + fx.Int32(load_iter * 4))
            rope_dim_i32 = _raw(ArithValue(dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
            lds_offset_i32 = _raw(
                ArithValue(token_i32) * fx.Int32(kv_lds_stride) + fx.Int32(DIM_NOPE) + ArithValue(rope_dim_i32)
            )
            _lds_store_bf16x8(lds_offset_i32, _mask_bf16x8(rope_groups[load_iter], valid))

    # ---- main loop: QK, online softmax, PV, M10 lookahead ----------------

    # Online softmax state: running max, running sum, and PV accumulators.
    row_max_init = _raw(c_neg_inf)
    row_sum_init = _raw(c_zero)
    state_init = [row_max_init, row_sum_init] + [_raw(zero_f32x4) for _ in range_constexpr(output_mfma_tiles_per_wave)]
    for tile_pos, state in range(_idx(split_start_i32), _idx(split_end_i32), _idx(block_n), init=state_init):
        tile_i32 = _i32(tile_pos)
        if ArithValue(tile_i32) == ArithValue(split_start_i32):
            _m9_fill_batched_tile(tile_i32)

        llvm.inline_asm(None, [], "s_waitcnt vmcnt(0) lgkmcnt(0)", "~{memory}", has_side_effects=True)

        # QK: each wave covers 16 tokens x 16 heads.
        token_row0_i32 = _raw(ArithValue(wave_i32) * fx.Int32(TOKENS_PER_WAVE) + ArithValue(head_i32))
        score_acc0 = zero_f32x4
        for k_step in range_constexpr(DIM_QK // MFMA_K):
            k_dim_i32 = _raw(ArithValue(lane_group_i32) * fx.Int32(MFMA_INPUT_VALUES) + fx.Int32(k_step * MFMA_K))
            k_offset0_i32 = _raw(ArithValue(token_row0_i32) * fx.Int32(kv_lds_stride) + ArithValue(k_dim_i32))
            k_fragment0 = _lds_load_bf16x8(k_offset0_i32)
            score_acc0 = _mfma_bf16(k_fragment0, q_fragments[k_step], score_acc0)

        scores = []
        score_count = MFMA_OUTPUT_VALUES
        for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
            token_in_tile_i32 = _raw(
                ArithValue(wave_i32) * fx.Int32(TOKENS_PER_WAVE)
                + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                + fx.Int32(value_idx)
            )
            logical_i32 = _raw(ArithValue(tile_i32) + ArithValue(token_in_tile_i32))
            valid = _raw(ArithValue(logical_i32) < ArithValue(split_end_i32))
            scaled = arith.mulf(_raw(score_acc0[value_idx]), _raw(softmax_score_scale), fastmath=fm_fast)
            scores.append(_select(valid, scaled, c_neg_inf))

        local_max = scores[0]
        for value_idx in range_constexpr(1, score_count):
            local_max = arith.maximumf(local_max, scores[value_idx], fastmath=fm_no_inf)
        for shuffle_offset in (16, 32):
            peer = fx.Float32(local_max).shuffle_xor(fx.Int32(shuffle_offset), fx.Int32(WARP_SIZE))
            local_max = arith.maximumf(local_max, _raw(peer), fastmath=fm_no_inf)
        if ArithValue(lane_group_i32) == fx.Int32(0):
            max_offset_i32 = _raw(ArithValue(wave_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))
            fx.memref_store(local_max, red_lds, max_offset_i32)

        gpu.barrier()

        tile_max = c_neg_inf.ir_value()
        for source_wave in range_constexpr(num_warps):
            value = fx.memref_load(red_lds, _raw(ArithValue(head_i32) + fx.Int32(source_wave * NUM_HEADS)))
            tile_max = arith.maximumf(tile_max, _raw(value), fastmath=fm_no_inf)
        row_max_old = fx.Float32(state[0])
        row_sum_old = fx.Float32(state[1])
        row_max_new = fx.Float32(arith.maximumf(_raw(row_max_old), tile_max, fastmath=fm_no_inf))

        old_scale = fx.Float32(
            _softmax_exp_difference(arith.subf(_raw(row_max_old), _raw(row_max_new), fastmath=fm_no_inf))
        )
        probabilities = []
        local_sum = c_zero.ir_value()
        for value_idx in range_constexpr(score_count):
            probability = _softmax_exp_difference(arith.subf(scores[value_idx], _raw(row_max_new), fastmath=fm_no_inf))
            probabilities.append(probability)
            local_sum = arith.addf(local_sum, probability, fastmath=fm_no_inf)
        for shuffle_offset in (16, 32):
            peer = fx.Float32(local_sum).shuffle_xor(fx.Int32(shuffle_offset), fx.Int32(WARP_SIZE))
            local_sum = arith.addf(local_sum, _raw(peer), fastmath=fm_no_inf)
        if ArithValue(lane_group_i32) == fx.Int32(0):
            sum_offset_i32 = _raw(
                fx.Int32(num_warps * NUM_HEADS) + ArithValue(wave_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32)
            )
            fx.memref_store(local_sum, red_lds, sum_offset_i32)

        p_base_i32 = _raw(
            (ArithValue(wave_i32) * fx.Int32(WARP_SIZE) + ArithValue(lane_i32)) * fx.Int32(MFMA_OUTPUT_VALUES)
        )
        p_bf16 = Vec.from_elements(probabilities, fx.Float32).to(fx.BFloat16)
        for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
            fx.memref_store(_raw(p_bf16[value_idx]), p_lds, _raw(ArithValue(p_base_i32) + fx.Int32(value_idx)))

        gpu.barrier()

        tile_sum = c_zero.ir_value()
        for source_wave in range_constexpr(num_warps):
            value = fx.memref_load(
                red_lds,
                _raw(fx.Int32(num_warps * NUM_HEADS) + fx.Int32(source_wave * NUM_HEADS) + ArithValue(head_i32)),
            )
            tile_sum = arith.addf(tile_sum, _raw(value), fastmath=fm_no_inf)

        row_sum_new = fx.Float32(
            arith.addf(arith.mulf(_raw(row_sum_old), _raw(old_scale), fastmath=fm_no_inf), tile_sum, fastmath=fm_no_inf)
        )
        # Rescale the carried PV state before adding this tile.
        pv_accumulators = []
        for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
            accumulator = Vec(state[2 + dim_tile], (MFMA_OUTPUT_VALUES,), fx.Float32)
            pv_accumulators.append(accumulator * old_scale)

        pv_source_pairs = num_warps // 2
        # M10 lookahead: prefetch next tile into VGPRs while current tile runs PV.
        m10_has_next_tile = _raw(ArithValue(tile_i32) + fx.Int32(block_n) < ArithValue(split_end_i32))
        # Keep individual names across the DSL if; these become the if-result SSA values.
        m10_packed_groups = [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(NOPE_DWORDX4_LOADS)]
        m10_scale_bytes = [fx.Int8(0) for _ in range_constexpr(NOPE_DWORDX4_LOADS)]
        m10_rope_groups = [zero_bf16x8 for _ in range_constexpr(ROPE_VEC8_LOADS)]
        m10_packed_0, m10_packed_1, m10_packed_2, m10_packed_3, m10_packed_4, m10_packed_5, m10_packed_6 = (
            m10_packed_groups
        )
        m10_scale_0, m10_scale_1, m10_scale_2, m10_scale_3, m10_scale_4, m10_scale_5, m10_scale_6 = m10_scale_bytes
        m10_rope_0, m10_rope_1 = m10_rope_groups

        if m10_has_next_tile:
            m10_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
            m10_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
            m10_next_tile_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n))
            m10_next_logical_i32 = _raw(ArithValue(m10_next_tile_i32) + ArithValue(m10_token_i32))
            m10_next_valid = _raw(ArithValue(m10_next_logical_i32) < ArithValue(split_end_i32))
            m10_safe_logical_i32 = _select(m10_next_valid, m10_next_logical_i32, split_start_i32)
            m10_physical_lane_i32 = fx.Int32(0).ir_value()
            if ArithValue(m10_fragment_lane_i32) == fx.Int32(0):
                m10_physical_lane_i32 = buffer_ops.buffer_load(
                    kv_indices_rsrc, m10_safe_logical_i32, vec_width=1, dtype=T.i32
                )
            m10_physical_i32 = dpp_utils.update_dpp_i32(m10_physical_lane_i32, m10_physical_lane_i32, 0)
            m10_loaded_packed_groups = []
            m10_loaded_scale_bytes = []
            for m10_load_iter in range_constexpr(NOPE_DWORDX4_LOADS):
                m10_dim_group_i32 = _raw(ArithValue(m10_fragment_lane_i32) + fx.Int32(m10_load_iter * 4))
                m10_dim_i32 = _raw(ArithValue(m10_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                m10_dword_offset_i32 = _raw(
                    ArithValue(m10_physical_i32) * fx.Int32(PACKED_DWORDS)
                    + ArithValue(m10_dim_i32).with_signedness(False) // fx.Int32(4)
                )
                m10_loaded_packed_groups.append(
                    Vec(
                        buffer_ops.buffer_load(kv_packed_rsrc, m10_dword_offset_i32, vec_width=4, dtype=T.i32),
                        (4,),
                        fx.Int32,
                    )
                )
                m10_scale_offset_i32 = _raw(
                    ArithValue(m10_physical_i32) * fx.Int32(DIM_PACKED)
                    + fx.Int32(NOPE_SCALE_OFFSET)
                    + ArithValue(m10_dim_i32).with_signedness(False) // fx.Int32(MFMA_K)
                )
                m10_loaded_scale_bytes.append(
                    buffer_ops.buffer_load(kv_packed_rsrc, m10_scale_offset_i32, vec_width=1, dtype=T.i8)
                )

            m10_loaded_rope_groups = []
            for m10_load_iter in range_constexpr(ROPE_VEC8_LOADS):
                m10_dim_group_i32 = _raw(ArithValue(m10_fragment_lane_i32) + fx.Int32(m10_load_iter * 4))
                m10_rope_dim_i32 = _raw(ArithValue(m10_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                m10_loaded_rope_groups.append(_load_rope8(_raw(m10_physical_i32), m10_rope_dim_i32, kv_rope_rsrc))
            m10_packed_0, m10_packed_1, m10_packed_2, m10_packed_3, m10_packed_4, m10_packed_5, m10_packed_6 = (
                m10_loaded_packed_groups
            )
            m10_scale_0, m10_scale_1, m10_scale_2, m10_scale_3, m10_scale_4, m10_scale_5, m10_scale_6 = (
                m10_loaded_scale_bytes
            )
            m10_rope_0, m10_rope_1 = m10_loaded_rope_groups

        m10_packed_groups = [
            m10_packed_0,
            m10_packed_1,
            m10_packed_2,
            m10_packed_3,
            m10_packed_4,
            m10_packed_5,
            m10_packed_6,
        ]
        m10_scale_bytes = [m10_scale_0, m10_scale_1, m10_scale_2, m10_scale_3, m10_scale_4, m10_scale_5, m10_scale_6]
        m10_rope_groups = [m10_rope_0, m10_rope_1]

        # PV consumes row-major KV through ds_read_tr16_b64; no separate V transpose.
        for source_pair in range_constexpr(pv_source_pairs):
            source0 = source_pair * 2
            source1 = source0 + 1
            p_values = []
            for source_wave in (source0, source1):
                source_p_base_i32 = _raw(
                    (fx.Int32(source_wave * WARP_SIZE) + ArithValue(lane_i32)) * fx.Int32(MFMA_OUTPUT_VALUES)
                )
                for value_idx in range_constexpr(MFMA_OUTPUT_VALUES):
                    p_value = fx.memref_load(p_lds, _raw(ArithValue(source_p_base_i32) + fx.Int32(value_idx)))
                    p_values.append(_raw(p_value))
            p_fragment = Vec.from_elements(p_values, fx.BFloat16)

            wide_pv_fragments = []
            for wide_dim_tile in range_constexpr(output_mfma_tiles_per_wave):
                tr_source_token_i32 = _raw(
                    fx.Int32(source0 * TOKENS_PER_WAVE)
                    + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
                    + ArithValue(head_i32).with_signedness(False) // fx.Int32(MFMA_OUTPUT_VALUES)
                )
                tr_source_dim_i32 = _raw(
                    ArithValue(wave_i32) * fx.Int32(output_dims_per_wave)
                    + fx.Int32(wide_dim_tile * 16)
                    + ArithValue(head_i32) % fx.Int32(MFMA_OUTPUT_VALUES) * fx.Int32(MFMA_OUTPUT_VALUES)
                )
                tr_base_i32 = _raw(
                    ArithValue(tr_source_token_i32) * fx.Int32(kv_lds_stride) + ArithValue(tr_source_dim_i32)
                )
                v_lo = _qh64_ds_read_tr16_bf16x4(tr_base_i32)
                v_hi = _qh64_ds_read_tr16_bf16x4(
                    _raw(ArithValue(tr_base_i32) + fx.Int32(TOKENS_PER_WAVE * kv_lds_stride))
                )
                wide_pv_fragments.append(v_lo.shuffle(v_hi, list(range(MFMA_INPUT_VALUES))))

            for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
                pv_accumulators[dim_tile] = _mfma_bf16(
                    wide_pv_fragments[dim_tile], p_fragment, pv_accumulators[dim_tile]
                )

        next_state = [_raw(row_max_new), _raw(row_sum_new)]
        for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
            next_state.append(_raw(pv_accumulators[dim_tile]))

        gpu.barrier()
        if m10_has_next_tile:
            # Publish the prefetched M10 tile after PV has consumed the current LDS tile.
            m10_store_token_i32 = _raw(ArithValue(tid_i32).with_signedness(False) // fx.Int32(4))
            m10_store_fragment_lane_i32 = _raw(ArithValue(tid_i32) % fx.Int32(4))
            m10_store_next_tile_i32 = _raw(ArithValue(tile_i32) + fx.Int32(block_n))
            m10_store_logical_i32 = _raw(ArithValue(m10_store_next_tile_i32) + ArithValue(m10_store_token_i32))
            m10_store_valid = _raw(ArithValue(m10_store_logical_i32) < ArithValue(split_end_i32))
            for m10_store_iter in range_constexpr(NOPE_DWORDX4_LOADS):
                m10_store_dim_group_i32 = _raw(ArithValue(m10_store_fragment_lane_i32) + fx.Int32(m10_store_iter * 4))
                m10_store_dim_i32 = _raw(ArithValue(m10_store_dim_group_i32) * fx.Int32(2 * MFMA_INPUT_VALUES))
                m10_decoded_fragments = _decode_nope16_prefetched(
                    m10_packed_groups[m10_store_iter], m10_scale_bytes[m10_store_iter]
                )
                for m10_fragment_idx in range_constexpr(2):
                    m10_decoded = _mask_bf16x8(m10_decoded_fragments[m10_fragment_idx], m10_store_valid)
                    m10_store_lds_offset_i32 = _raw(
                        ArithValue(m10_store_token_i32) * fx.Int32(kv_lds_stride)
                        + ArithValue(m10_store_dim_i32)
                        + fx.Int32(m10_fragment_idx * MFMA_INPUT_VALUES)
                    )
                    _lds_store_bf16x8(m10_store_lds_offset_i32, m10_decoded)
            for m10_store_iter in range_constexpr(ROPE_VEC8_LOADS):
                m10_store_dim_group_i32 = _raw(ArithValue(m10_store_fragment_lane_i32) + fx.Int32(m10_store_iter * 4))
                m10_store_rope_dim_i32 = _raw(ArithValue(m10_store_dim_group_i32) * fx.Int32(MFMA_INPUT_VALUES))
                m10_store_lds_offset_i32 = _raw(
                    ArithValue(m10_store_token_i32) * fx.Int32(kv_lds_stride)
                    + fx.Int32(DIM_NOPE)
                    + ArithValue(m10_store_rope_dim_i32)
                )
                m10_store_rope_group = m10_rope_groups[m10_store_iter]
                _lds_store_bf16x8(m10_store_lds_offset_i32, _mask_bf16x8(m10_store_rope_group, m10_store_valid))
        results = yield next_state

    # ---- epilogue: sink normalization and output store -------------------

    # Sink-aware normalization and final bf16 store.
    row_max_tokens = fx.Float32(results[0])
    row_sum_tokens = fx.Float32(results[1])
    row_max_final = fx.Float32(arith.maximumf(_raw(row_max_tokens), _raw(sink_lane), fastmath=fm_no_inf))
    token_scale = fx.Float32(
        _softmax_exp_difference(arith.subf(_raw(row_max_tokens), _raw(row_max_final), fastmath=fm_no_inf))
    )
    sink_scale = _softmax_exp_difference(arith.subf(_raw(sink_lane), _raw(row_max_final), fastmath=fm_no_inf))
    row_sum_final = fx.Float32(
        arith.addf(
            arith.mulf(_raw(row_sum_tokens), _raw(token_scale), fastmath=fm_no_inf), sink_scale, fastmath=fm_no_inf
        )
    )

    inv_sum = fx.Float32(rocdl.rcp(T.f32, _raw(row_sum_final)))
    output_record_i32 = _raw(ArithValue(q_row_i32) * fx.Int32(NUM_HEADS) + ArithValue(head_i32))

    for dim_tile in range_constexpr(output_mfma_tiles_per_wave):
        accumulator = Vec(results[2 + dim_tile], (MFMA_OUTPUT_VALUES,), fx.Float32)
        normalized = accumulator * token_scale * inv_sum
        dim_offset_i32 = _raw(
            ArithValue(wave_i32) * fx.Int32(output_dims_per_wave)
            + fx.Int32(dim_tile * 16)
            + ArithValue(lane_group_i32) * fx.Int32(MFMA_OUTPUT_VALUES)
        )
        buffer_ops.buffer_store(
            normalized.to(fx.BFloat16),
            output_rsrc,
            _raw(ArithValue(output_record_i32) * fx.Int32(V_HEAD_DIM) + ArithValue(dim_offset_i32)),
        )
