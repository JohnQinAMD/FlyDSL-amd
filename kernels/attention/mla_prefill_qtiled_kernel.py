# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL absorbed-MLA PREFILL device kernel (query-tiled, causal, gfx950).

Design: one workgroup per (query token, head-group of BLOCK_H heads).  All
BLOCK_H heads of a workgroup share the SAME KV latent stream, so every KV
element is fetched from HBM ONCE and reused across BLOCK_H heads (an HBM-traffic
cut of BLOCK_H vs. the one-head-per-workgroup baseline).  Because each head is
handled by its own scalar lane-reduction, the online-softmax running max / sum
and the O rescale factor stay per-head SCALARS — no MFMA-fragment row
redistribution — which keeps the kernel simple and provably correct.

Absorbed MLA / MQA 128:1: all heads attend one shared 576-wide latent per KV
token.  Score(h,q,k) = q[576]·kv[576]*sm_scale ; Out = softmax_k·kv[:512].
CAUSAL full prefill (batch=1): logical query token j attends logical KV
positions 0..j; logical KV position i lives in physical page
kv_page_indices[kv_indptr[b] + i] (page size 1).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

QK_HEAD_DIM = 576
V_HEAD_DIM = 512
NUM_QO_HEADS = 128

BLOCK_THREADS = 256
WARP_SIZE = 64
RED_SLOTS = BLOCK_THREADS // WARP_SIZE  # 4
LOG2E = 1.4426950408889634

# LDS score-buffer capacity (per head): supports sequences up to MAX_KV tokens.
MAX_KV = 1024
QK_VEC = 8
QK_STEPS = QK_HEAD_DIM // QK_VEC  # 72

# Heads processed per workgroup (KV reuse factor).  H=1 measured fastest:
# the kernel is scalar-compute/latency-bound, so more workgroups (H=1) wins
# over KV reuse (larger H serializes more work per workgroup).
BLOCK_H = 1


def _make_shared(block_h):
    @fx.struct
    class SharedStorage:
        q_lds: fx.Array[fx.Float32, block_h * QK_HEAD_DIM, 16]
        p_lds: fx.Array[fx.Float32, block_h * MAX_KV, 16]
        s_red: fx.Array[fx.Float32, RED_SLOTS, 16]

    return SharedStorage


def build_mla_prefill_qtiled_module(is_causal: bool = True, block_h: int = BLOCK_H):
    CAUSAL = bool(is_causal)
    H = int(block_h)
    assert NUM_QO_HEADS % H == 0
    SharedStorage = _make_shared(H)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def mla_prefill_kernel(
        query: fx.Tensor,           # [total_q * 128 * 576] bf16 (flat)
        kv_flat: fx.Tensor,         # [num_page * 576] bf16 (flat)
        kv_page_indices: fx.Tensor,  # [num_page] i32
        kv_base: fx.Int32,          # kv_indptr[b]  (batch=1 -> 0)
        out: fx.Tensor,             # [total_q * 128 * 512] bf16 (flat)
        seq_len: fx.Int32,          # total_q (batch=1)
        sm_scale: fx.Float32,
    ):
        row = fx.block_idx.x   # query token j
        hg = fx.block_idx.y    # head group
        tid = fx.thread_idx.x
        head_base = hg * fx.Int32(H)

        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = fx.arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_lds = lds.q_lds.view(fx.make_layout(H * QK_HEAD_DIM, 1))
        p_lds = lds.p_lds.view(fx.make_layout(H * MAX_KV, 1))
        s_red = lds.s_red.view(fx.make_layout(RED_SLOTS, 1))

        q_rsrc = buffer_ops.create_buffer_resource(query)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_flat)
        kvidx_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
        out_rsrc = buffer_ops.create_buffer_resource(out)

        if const_expr(CAUSAL):
            n_len = row + fx.Int32(1)
        else:
            n_len = seq_len
        n_len_idx = fx.Index(n_len)

        def wave_reduce(x, mode):
            w = x
            for sh in range_constexpr(6):
                off = WARP_SIZE // (2 << sh)
                peer = w.shuffle_xor(off, WARP_SIZE)
                if const_expr(mode == "max"):
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce(val, mode):
            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            neutral = c_neg_inf if mode == "max" else c_zero_f
            w = wave_reduce(val, mode)
            if lane == 0:
                fx.memref_store(w, s_red, wave)
            gpu.barrier()
            in_range = lane < RED_SLOTS
            lane_safe = in_range.select(lane, fx.Int32(0))
            v = fx.memref_load(s_red, lane_safe)
            ww = in_range.select(v, neutral)
            ww = wave_reduce(ww, mode)
            gpu.barrier()
            return ww

        # ---- load Q for the H heads into LDS as f32 ----
        # H*576 elements, strided by heads (query row-major [row, head, dim]).
        for h in range_constexpr(H):
            q_off_base = (row * NUM_QO_HEADS + head_base + fx.Int32(h)) * QK_HEAD_DIM
            for j in range_constexpr((QK_HEAD_DIM + BLOCK_THREADS - 1) // BLOCK_THREADS):
                d = tid + fx.Int32(j * BLOCK_THREADS)
                if d < fx.Int32(QK_HEAD_DIM):
                    qv = buffer_ops.buffer_load(q_rsrc, q_off_base + d, vec_width=1, dtype=fx.BFloat16)
                    fx.memref_store(fx.Float32(fx.BFloat16(qv)), q_lds, fx.Int32(h * QK_HEAD_DIM) + d)
        gpu.barrier()

        # ---- score: each thread computes H dot products per owned KV position ----
        # KV is swept in runtime tiles of BLOCK_THREADS (thread t owns kv =
        # tile*BLOCK_THREADS + t).  n_tiles is bounded to the causal range, so no
        # thread ever computes an address for a kv position past the sequence.
        c_bt = fx.Int32(BLOCK_THREADS)
        n_tiles = (n_len + fx.Int32(BLOCK_THREADS - 1)) // c_bt
        n_tiles_idx = fx.Index(n_tiles)

        # The DSL's scf.for carry collapses a length-1 list to a bare scalar and
        # mis-wires; pad the carried state to width >= 2 (the proven iterable
        # pattern used by the output loop below).  First H slots are the real
        # per-head reduction; extra slots are inert.
        PAD = max(2 - H, 0)

        smax_init = [c_neg_inf for _ in range_constexpr(H)] + [c_zero_f for _ in range_constexpr(PAD)]
        for _t, st in range(fx.Index(0), n_tiles_idx, fx.Index(1), init=smax_init):
            tmax = list(st)
            kv = tid + fx.Int32(_t) * c_bt
            valid = kv < n_len
            kv_safe = valid.select(kv, fx.Int32(0))
            phys = fx.Int32(buffer_ops.buffer_load(kvidx_rsrc, kv_base + kv_safe, vec_width=1, dtype=fx.Int32))
            base = phys * fx.Int32(QK_HEAD_DIM)
            acc = [c_zero_f for _ in range_constexpr(H)]
            for t in range_constexpr(QK_STEPS):
                kvv = fx.Vector(
                    buffer_ops.buffer_load(kv_rsrc, base + fx.Int32(t * QK_VEC), vec_width=QK_VEC, dtype=fx.BFloat16)
                ).to(fx.Float32)
                for i in range_constexpr(QK_VEC):
                    kvi = kvv[i]
                    d_idx = t * QK_VEC + i
                    for h in range_constexpr(H):
                        qd = fx.memref_load(q_lds, fx.Int32(h * QK_HEAD_DIM + d_idx))
                        acc[h] = acc[h] + qd * kvi
            nxt = []
            for h in range_constexpr(H):
                s = valid.select(acc[h] * sm_scale, c_neg_inf)
                fx.memref_store(s, p_lds, fx.Int32(h * MAX_KV) + kv)
                nxt.append(tmax[h].maximumf(valid.select(s, c_neg_inf)))
            nxt = nxt + [c_zero_f for _ in range_constexpr(PAD)]
            smax_res = yield nxt
        thread_max = list(smax_res)[:H]
        gpu.barrier()

        # ---- per-head max/sum reductions ----
        m = [block_reduce(thread_max[h], "max") for h in range_constexpr(H)]

        ssum_init = [c_zero_f for _ in range_constexpr(H + PAD)]
        for _t, st in range(fx.Index(0), n_tiles_idx, fx.Index(1), init=ssum_init):
            tsum = list(st)
            kv = tid + fx.Int32(_t) * c_bt
            valid = kv < n_len
            nxt = []
            for h in range_constexpr(H):
                s = fx.memref_load(p_lds, fx.Int32(h * MAX_KV) + kv)
                p = valid.select(fmath.exp2((s - m[h]) * c_log2e, fastmath=fm_fast), c_zero_f)
                fx.memref_store(p, p_lds, fx.Int32(h * MAX_KV) + kv)
                nxt.append(tsum[h] + p)
            nxt = nxt + [c_zero_f for _ in range_constexpr(PAD)]
            ssum_res = yield nxt
        thread_sum = list(ssum_res)[:H]
        gpu.barrier()

        l = [block_reduce(thread_sum[h], "sum") for h in range_constexpr(H)]
        inv_l = [fx.Float32(1.0) / l[h] for h in range_constexpr(H)]

        # ---- output: O[h][d] = sum_k p[h][k]*v[k][d] ; d = tid, tid+256 ----
        d0 = tid
        d1 = tid + fx.Int32(BLOCK_THREADS)
        init = []
        for h in range_constexpr(H):
            init.append(c_zero_f)  # o0[h]
            init.append(c_zero_f)  # o1[h]
        start = fx.Index(0)
        stop = n_len_idx
        step = fx.Index(1)
        for _k, st in range(start, stop, step, init=init):
            o = list(st)
            k_i32 = fx.Int32(_k)
            phys = fx.Int32(buffer_ops.buffer_load(kvidx_rsrc, kv_base + k_i32, vec_width=1, dtype=fx.Int32))
            base = phys * fx.Int32(QK_HEAD_DIM)
            v0 = fx.Float32(fx.BFloat16(buffer_ops.buffer_load(kv_rsrc, base + d0, vec_width=1, dtype=fx.BFloat16)))
            v1 = fx.Float32(fx.BFloat16(buffer_ops.buffer_load(kv_rsrc, base + d1, vec_width=1, dtype=fx.BFloat16)))
            for h in range_constexpr(H):
                pk = fx.memref_load(p_lds, fx.Int32(h * MAX_KV) + k_i32)
                o[2 * h] = o[2 * h] + pk * v0
                o[2 * h + 1] = o[2 * h + 1] + pk * v1
            res = yield list(o)

        for h in range_constexpr(H):
            out_base = (row * NUM_QO_HEADS + head_base + fx.Int32(h)) * V_HEAD_DIM
            buffer_ops.buffer_store(fx.BFloat16(res[2 * h] * inv_l[h]), out_rsrc, out_base + d0)
            buffer_ops.buffer_store(fx.BFloat16(res[2 * h + 1] * inv_l[h]), out_rsrc, out_base + d1)

    @flyc.jit
    def launch_mla_prefill(
        query: fx.Tensor,
        kv_flat: fx.Tensor,
        kv_page_indices: fx.Tensor,
        kv_base: fx.Int32,
        out: fx.Tensor,
        total_q: fx.Int32,
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        mla_prefill_kernel(
            query, kv_flat, kv_page_indices, kv_base, out, seq_len, sm_scale,
            value_attrs={"rocdl.waves_per_eu": 1},
        ).launch(
            grid=(total_q, NUM_QO_HEADS // H, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_mla_prefill
