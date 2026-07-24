# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL absorbed-MLA PREFILL — MFMA query-tiled flash-attention (gfx950, bf16).

One workgroup owns BLOCK_M query rows = BLOCK_M heads of ONE token (so the causal
KV length is uniform across the tile).  Paged KV latent tiles of BLOCK_N are
streamed into LDS ONCE and reused across all BLOCK_M heads.  Two chained
matrix-engine GEMMs through the high-level tiled-MMA layout API:
    S = Q[BM,576]·Kᵀ[576,BN]      (mfma_f32_16x16x16_bf16, contract 576)
    O[BM,512] += P[BM,BN]·V[BN,512]  (contract BN)

Fragment-layout bridging through LDS (all verified in isolation, rel<3e-3):
  * gemm1 C fragment (S) is stored to LDS f32 [BM,BN] (identity layout) and the
    online softmax runs one-thread-per-row (head), reading S row-major and writing
    P (bf16) row-major to LDS.
  * V is staged into a CONTIGUOUS transposed LDS buffer [V_HEAD_DIM, BN] so gemm2's
    B operand is a clean read.
  * The running O accumulator lives in LDS f32 [BM,V_HEAD_DIM]; it is rescaled
    per-head by ``corr`` (linear pass) then accumulated by a gemm2 fragment
    round-trip (load C from o_lds, gemm, store C back).  tiled_copy_C store/load to
    LDS is an identity round-trip, so this is exact.
The epilogue normalizes O by the running denominator ``l`` and writes bf16 out
with a simple linear buffer_store (no fragment relayout on the way out).
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath

QK_HEAD_DIM = 576
V_HEAD_DIM = 512
NUM_QO_HEADS = 128

# Tile config (coupled: BM x BN x gemm2-contract). Overridable via env for
# sweeping; default 32/32 is the correctness-banked baseline.
BLOCK_M = int(os.environ.get("FLYDSL_MLA_BM", "32"))
# BN=32 keeps k_lds (bn*578*2 = 37 KB) + o_lds scratch small enough for occ 2
# (LDS 74.6 KB <= 80, VGPR 244 <= 256); with the K-pad de-conflict it edges out
# BN=96/occ1 (1.266 vs 1.293 ms) by ~2x-ing the sync-bound bucket via occupancy.
BLOCK_N = int(os.environ.get("FLYDSL_MLA_BN", "32"))
BLOCK_THREADS = 256
LOG2E = 1.4426950408889634

# k_lds row-stride PAD (bf16 elems). Row stride 576 = 288 words and 288 % 32 == 0,
# so a gemm1 contract-column read (fixed k, varying n) hits the SAME LDS bank for
# every n -> up to 32-way bank conflict (measured 0.59 ms / 30% of the kernel at
# S=1024). Padding by 2 makes the word-stride odd (289) so consecutive KV rows land
# in different banks -> conflict-free; also de-conflicts gemm2's vt_lds V-read.
# KPAD=2 measured 1.95 -> 1.29 ms. (KPAD>=8 overflows the 160 KB LDS at BN=96.)
K_PAD = int(os.environ.get("FLYDSL_MLA_KPAD", "2"))
K_STRIDE = QK_HEAD_DIM + K_PAD

# Register-resident O: carry the gemm2 C-accumulator fragment across the KV loop
# instead of the per-tile o_lds store+reload round-trip (measured 0.43 ms / 21% at
# S=1024).  Default OFF (o_lds path == the correctness-banked baseline); set
# FLYDSL_MLA_REGO=1 to select the single-pass register-resident path.
_REGO = os.environ.get("FLYDSL_MLA_REGO", "1") not in ("0", "false", "False")
# v-dim chunking for the register-resident O accumulator.  Also sizes the tiny
# corr/inv-l broadcast LDS buffer ([BM, V_HEAD_DIM//REGO_PVC] f32), so keep it >=4
# (PVC=1 -> 64 KB broadcast, does not fit alongside the 108 KB K tile).
_REGO_PVC = int(os.environ.get("FLYDSL_MLA_REGO_PVC", "4"))

# ---- occ-1 latency-hiding knobs (framework-level scheduling + register-allocation
# control) ----
# The REGO path is latency/sync-bound at ~13% of the bf16 MAF roofline: it passes NO
# scheduling or RA hints to the compiler, so at occ-1 (BM=64) the LLVM machine
# scheduler cannot interleave the KV-gather VMEM latency with MFMA/VALU, and the RA
# targets the default 2-wave occupancy (spilling 1 VGPR at 411 live).  This mirrors
# the dense dualwave-SWP kernel's hint set (flash_attn_gfx950.py):
#   * value_attrs rocdl.waves_per_eu -> RA occupancy TARGET.  At BM=64 setting it to 1
#     hands the allocator the full ~512-VGPR budget so the occ-1 mega-tile is 0-spill.
#   * passthrough DAZ/no-nans/unsafe-fp-math -> frees the FP scheduler.
#   * llvm_options enable-post-misched -> a second post-RA machine-scheduler pass that
#     hoists independent VMEM loads above the MFMA/VALU that consume earlier tiles.
# Default ON; set FLYDSL_MLA_SCHED=0 to restore the exact no-hint baseline.
_SCHED = os.environ.get("FLYDSL_MLA_SCHED", "1") not in ("0", "false", "False")
# waves_per_eu target: 0 = auto (occ-1 for the BM>=64 mega-tile, occ-2 for BM<=32).
_WPE = int(os.environ.get("FLYDSL_MLA_WPE", "0"))
# enable-post-misched: "auto" (default) = on for occ-2 (BM<64), off for the occ-1
# mega-tile (BM>=64).  Measured: the extra post-RA reschedule helps occ-2 (BM=32
# 0.893 on / 0.909 off) but hurts occ-1 (BM=64 0.888 off / 0.893 on) because at 412
# live VGPR it reorders against the tight occ-1 budget.  "0"/"1" force it.
_POST_MISCHED_ENV = os.environ.get("FLYDSL_MLA_POSTMISCHED", "auto")


def _post_misched(bm):
    if _POST_MISCHED_ENV == "auto":
        return bm < 64
    return _POST_MISCHED_ENV not in ("0", "false", "False")


def _sched_value_attrs(bm):
    if not _SCHED:
        return None
    wpe = _WPE or (1 if bm >= 64 else 2)
    return {
        "rocdl.waves_per_eu": wpe,
        "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
        "passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }


def _sched_compile_hints(bm):
    if not _SCHED:
        return None
    return {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": _post_misched(bm),
            "lsr-drop-solution": True,
        },
    }


def _make_shared(bm, bn):
    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[fx.BFloat16, bn * K_STRIDE, 16]      # K tile [BN,576] padded row-stride (gemm1 B); V=first 512 cols
        s_lds: fx.Array[fx.Float32, bm * bn, 16]             # raw scores from gemm1
        p_lds: fx.Array[fx.BFloat16, bm * bn, 16]            # softmax probs -> gemm2 A
        o_lds: fx.Array[fx.BFloat16, bm * V_HEAD_DIM, 16]    # running O accumulator (bf16: LDS<80KB -> occ2)
        m_lds: fx.Array[fx.Float32, bm, 16]
        l_lds: fx.Array[fx.Float32, bm, 16]
        corr_lds: fx.Array[fx.Float32, bm, 16]
        corr_bf_lds: fx.Array[fx.BFloat16, bm, 16]  # per-head rescale corr (bf16) for in-register broadcast
        phys_lds: fx.Array[fx.Int32, bn, 16]

    return SharedStorage


def _build_rego(is_causal: bool = True, bm: int = BLOCK_M, bn: int = BLOCK_N):
    CAUSAL = bool(is_causal)
    BM = int(bm)
    BN = int(bn)
    HG = NUM_QO_HEADS // BM
    SS = _make_shared(BM, BN)
    GATHER_PER_THR = (BN * QK_HEAD_DIM + BLOCK_THREADS - 1) // BLOCK_THREADS
    GATHER_VEC = 8
    assert QK_HEAD_DIM % GATHER_VEC == 0 and V_HEAD_DIM % GATHER_VEC == 0
    QK_VEC_PER_ROW = QK_HEAD_DIM // GATHER_VEC
    GATHER_VEC_GROUPS = (BN * QK_HEAD_DIM) // GATHER_VEC
    GATHER_VEC_ITERS = (GATHER_VEC_GROUPS + BLOCK_THREADS - 1) // BLOCK_THREADS
    O_ELEMS = BM * V_HEAD_DIM
    O_PER_THR = O_ELEMS // BLOCK_THREADS
    PV_CHUNKS = int(os.environ.get("FLYDSL_MLA_PVC", "4"))
    PV_CHUNK_N = V_HEAD_DIM // PV_CHUNKS  # 128

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def mfma_kernel(
        query2d: fx.Tensor,   # [total_q*128, 576] bf16
        kv_flat: fx.Tensor,   # [num_page*576] bf16
        kv_page_indices: fx.Tensor,  # [num_page] i32
        kv_base: fx.Int32,
        out2d: fx.Tensor,     # [total_q*128, 512] bf16
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
    ):
        row = fx.block_idx.x   # token j
        hg = fx.block_idx.y    # head group
        tid = fx.thread_idx.x

        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = fx.arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SS).peek()
        k_lds = lds.k_lds.view(fx.make_layout((BN, QK_HEAD_DIM), (K_STRIDE, 1)))
        k_lds_1d = lds.k_lds.view(fx.make_layout(BN * K_STRIDE, 1))
        # V operand for gemm2 is the first V_HEAD_DIM cols of the SAME latent held
        # in k_lds: read it as a transposed view [d, kv] (element (d,kv) at kv*576+d)
        # instead of staging a separate contiguous copy (saves V_HEAD_DIM*BN*2 B LDS).
        vt_lds = lds.k_lds.view(fx.make_layout((V_HEAD_DIM, BN), (1, K_STRIDE)))
        s_lds = lds.s_lds.view(fx.make_layout((BM, BN), (BN, 1)))
        # gemm1's tiled C-store scatters the fragment transposed (physical
        # location holds C[n,m]); write through a column-major view so the
        # row-major read view above then yields the correct C[m,n].
        s_lds_wr = lds.s_lds.view(fx.make_layout((BM, BN), (1, BM)))
        s_lds_1d = lds.s_lds.view(fx.make_layout(BM * BN, 1))
        p_lds = lds.p_lds.view(fx.make_layout((BM, BN), (BN, 1)))
        p_lds_1d = lds.p_lds.view(fx.make_layout(BM * BN, 1))
        o_lds_2d = lds.o_lds.view(fx.make_layout((BM, V_HEAD_DIM), (V_HEAD_DIM, 1)))
        o_lds_1d = lds.o_lds.view(fx.make_layout(O_ELEMS, 1))
        m_lds = lds.m_lds.view(fx.make_layout(BM, 1))
        l_lds = lds.l_lds.view(fx.make_layout(BM, 1))
        corr_lds = lds.corr_lds.view(fx.make_layout(BM, 1))
        corr_bf_lds = lds.corr_bf_lds.view(fx.make_layout(BM, 1))
        # broadcast (v-dim stride 0) view of the bf16 per-head corr, shaped like an O
        # PV-chunk tile so the gemm2 C-copy reads corr[q-row] into the O-fragment layout.
        corr_bcast = lds.corr_bf_lds.view(fx.make_layout((BM, PV_CHUNK_N), (1, 0)))
        phys_lds = lds.phys_lds.view(fx.make_layout(BN, 1))

        kvidx_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_flat)
        out_rsrc = buffer_ops.create_buffer_resource(out2d)

        qb = fx.rocdl.make_buffer_tensor(query2d)
        mtile = row * fx.Int32(HG) + hg
        bA = fx.slice(fx.zipped_divide(qb, (BM, QK_HEAD_DIM)), (None, mtile))
        # shape token [BM,BN] for gemm1 C fragment (only the layout shape is used)
        bS = fx.slice(fx.zipped_divide(qb, (BM, BN)), (None, fx.Int32(0)))

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr = tiled.thr_slice(tid)
        ca16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        uni16 = fx.make_copy_atom(fx.UniversalCopy(16), fx.BFloat16)
        uni32 = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)

        # ---- Q resident in registers (gemm1 A operand), loaded once ----
        thrA = fx.make_tiled_copy_A(ca16, tiled).get_slice(tid)
        fA = thr.make_fragment_A(bA)
        fx.copy(ca16, thrA.partition_S(bA), thrA.retile(fA), pred=None)

        # ---- init running state (O accumulator is register-resident, see below) ----
        if tid < fx.Int32(BM):
            fx.memref_store(c_neg_inf, m_lds, tid)
            fx.memref_store(c_zero_f, l_lds, tid)
        gpu.barrier()

        if const_expr(CAUSAL):
            n_len = row + fx.Int32(1)
        else:
            n_len = seq_len
        num_tiles = (fx.Index(n_len) + fx.Index(BN - 1)) // fx.Index(BN)

        thrB = fx.make_tiled_copy_B(uni16, tiled).get_slice(tid)
        thrCst = fx.make_tiled_copy_C(uni32, tiled).get_slice(tid)     # store S -> s_lds
        thrA2 = fx.make_tiled_copy_A(uni16, tiled).get_slice(tid)      # p_lds -> fP
        thrB2 = fx.make_tiled_copy_B(uni16, tiled).get_slice(tid)      # vt_lds -> fV
        thrOc = fx.make_tiled_copy_C(uni16, tiled).get_slice(tid)      # o_lds(bf16) <-> fO(f32)

        # ---- register-resident O accumulator (carried through scf.for; NO per-tile
        # LDS round-trip). One f32 MMA-C vector per PV chunk, initialized to zero. ----
        o_div_i = fx.zipped_divide(o_lds_2d, (BM, PV_CHUNK_N))
        vO_init = []
        for c in range_constexpr(PV_CHUNKS):
            fO0 = thr.make_fragment_C(fx.slice(o_div_i, (None, fx.Int32(c))))
            fO0.fill(0)
            vO_init.append(fx.Vector(fO0.load()).ir_value())

        loop_results = vO_init
        for t_idx, loop_args in range(fx.Index(0), num_tiles, fx.Index(1), init=vO_init):
            vO_chunks = [loop_args[c] for c in range_constexpr(PV_CHUNKS)]
            t_i32 = fx.Int32(t_idx)
            kv_tile0 = t_i32 * fx.Int32(BN)

            # ---- resolve paged KV rows for this tile ----
            if tid < fx.Int32(BN):
                pos = kv_base + kv_tile0 + tid
                valid_row = (kv_tile0 + tid) < n_len
                pos_safe = valid_row.select(pos, kv_base)
                ph = fx.Int32(buffer_ops.buffer_load(kvidx_rsrc, pos_safe, vec_width=1, dtype=fx.Int32))
                fx.memref_store(ph, phys_lds, tid)
            gpu.barrier()

            # ---- gather KV tile into LDS (K row-major + V transposed) ----
            # Vectorized: 8 contiguous head-dims per load (QK_HEAD_DIM % 8 == 0),
            # so each group stays within one KV row and one V/rope region.
            for i in range_constexpr(GATHER_VEC_ITERS):
                g = tid + fx.Int32(i * BLOCK_THREADS)
                if g < fx.Int32(GATHER_VEC_GROUPS):
                    d0 = (g % fx.Int32(QK_VEC_PER_ROW)) * fx.Int32(GATHER_VEC)
                    n = g // fx.Int32(QK_VEC_PER_ROW)
                    ph = fx.memref_load(phys_lds, n)
                    vv = fx.Vector(
                        buffer_ops.buffer_load(
                            kv_rsrc, ph * fx.Int32(QK_HEAD_DIM) + d0, vec_width=GATHER_VEC, dtype=fx.BFloat16
                        )
                    )
                    base_k = n * fx.Int32(K_STRIDE) + d0
                    for j in range_constexpr(GATHER_VEC):
                        fx.memref_store(fx.BFloat16(vv[j]), k_lds_1d, base_k + fx.Int32(j))
            gpu.barrier()

            # ---- gemm1: S = Q @ K^T ----
            fB = thr.make_fragment_B(k_lds)
            fx.copy(uni16, thrB.partition_S(k_lds), thrB.retile(fB), pred=None)
            fS = thr.make_fragment_C(bS)
            fS.fill(0)
            fx.gemm(mma, fS, fA, fB, fS)
            fx.copy(uni32, thrCst.retile(fS), thrCst.partition_D(s_lds), pred=None)
            gpu.barrier()

            # ---- online softmax on s_lds — PARALLELIZED across TPR threads/row ----
            # Each head-row is reduced by TPR=BLOCK_THREADS//BM threads (all 256
            # threads active, vs 32 before); each owns CPR=BN//TPR kv columns and
            # the per-row max/sum are combined by an intra-wave shuffle_xor reduce
            # over the TPR-aligned lane group.
            TPR = BLOCK_THREADS // BM          # threads per head-row (8)
            CPR = BN // TPR                    # kv columns per thread (4)
            srow = tid // fx.Int32(TPR)        # head row 0..BM-1
            ssub = tid % fx.Int32(TPR)         # sub-lane 0..TPR-1
            m_row = fx.memref_load(m_lds, srow)
            l_row = fx.memref_load(l_lds, srow)
            s_loc = []
            v_loc = []
            local_max = c_neg_inf
            for cc in range_constexpr(CPR):
                ncol = ssub * fx.Int32(CPR) + fx.Int32(cc)
                pos_ok = (kv_tile0 + ncol) < n_len
                scaled = fx.memref_load(s_lds_1d, srow * fx.Int32(BN) + ncol) * sm_scale
                s_loc.append(scaled)
                v_loc.append(pos_ok)
                local_max = local_max.maximumf(pos_ok.select(scaled, c_neg_inf))
            tmax = local_max
            for off in (1, 2, 4, 8, 16):
                if const_expr(off < TPR):
                    tmax = tmax.maximumf(tmax.shuffle_xor(off, 64))
            m_new = m_row.maximumf(tmax)
            corr = fmath.exp2((m_row - m_new) * c_log2e, fastmath=fm_fast)
            local_sum = c_zero_f
            for cc in range_constexpr(CPR):
                ncol = ssub * fx.Int32(CPR) + fx.Int32(cc)
                p = v_loc[cc].select(
                    fmath.exp2((s_loc[cc] - m_new) * c_log2e, fastmath=fm_fast), c_zero_f
                )
                fx.memref_store(fx.BFloat16(p), p_lds_1d, srow * fx.Int32(BN) + ncol)
                local_sum = local_sum + p
            tsum = local_sum
            for off in (1, 2, 4, 8, 16):
                if const_expr(off < TPR):
                    tsum = tsum + tsum.shuffle_xor(off, 64)
            l_new = l_row * corr + tsum
            if ssub == fx.Int32(0):
                fx.memref_store(m_new, m_lds, srow)
                fx.memref_store(l_new, l_lds, srow)
                # bf16 per-head corr for the in-register broadcast (consumed in gemm2)
                fx.memref_store(fx.BFloat16(corr), corr_bf_lds, srow)
            gpu.barrier()

            # ---- gemm2: O(reg) = O(reg) * corr + P @ V, accumulated in registers ----
            fP = thr.make_fragment_A(p_lds)
            fx.copy(uni16, thrA2.partition_S(p_lds), thrA2.retile(fP), pred=None)
            vt_div = fx.zipped_divide(vt_lds, (PV_CHUNK_N, BN))
            o_div = fx.zipped_divide(o_lds_2d, (BM, PV_CHUNK_N))
            # Read per-head corr ONCE, directly in the O-fragment layout, via the SAME
            # thrOc C-copy but from the broadcast (v-stride-0) view of corr_bf_lds:
            # register->q-row map is correct-by-construction (no hand-derived MFMA map).
            # corr depends only on the q-row, so the identical fragment is reused across
            # every PV chunk. This replaces the o_lds broadcast round-trip
            # (~64 ds_write_b16 + readback + 1 gpu.barrier per tile).
            fCorr_bf = fx.make_fragment_like(
                thr.make_fragment_C(fx.slice(o_div, (None, fx.Int32(0)))), fx.BFloat16.ir_type
            )
            fx.copy(uni16, thrOc.partition_S(corr_bcast), thrOc.retile(fCorr_bf), pred=None)
            vO_next = []
            for c in range_constexpr(PV_CHUNKS):
                o_c = fx.slice(o_div, (None, fx.Int32(c)))
                vt_c = fx.slice(vt_div, (None, fx.Int32(c)))
                corr_vec = fx.Vector(fCorr_bf.load()).to(fx.Float32)
                rescaled = fx.Vector(vO_chunks[c]) * corr_vec
                # accumulate this tile's P @ V into the rescaled register accumulator
                fV_c = thr.make_fragment_B(vt_c)
                fx.copy(uni16, thrB2.partition_S(vt_c), thrB2.retile(fV_c), pred=None)
                fO_c = thr.make_fragment_C(o_c)
                fO_c.store(rescaled.ir_value())
                fx.gemm(mma, fO_c, fP, fV_c, fO_c)
                vO_next.append(fx.Vector(fO_c.load()).ir_value())
            gpu.barrier()
            loop_results = yield vO_next

        # ---- epilogue: land register-resident O into o_lds (once), normalize by the
        # running denominator l (f32, exact) and store bf16 to out. ----
        # scf.for unwraps a single loop-carried value; keep a uniform list.
        loop_final = [loop_results] if const_expr(PV_CHUNKS == 1) else loop_results
        vO_final = [loop_final[c] for c in range_constexpr(PV_CHUNKS)]
        o_div_e = fx.zipped_divide(o_lds_2d, (BM, PV_CHUNK_N))
        for c in range_constexpr(PV_CHUNKS):
            o_c = fx.slice(o_div_e, (None, fx.Int32(c)))
            fO_bf = fx.make_fragment_like(thr.make_fragment_C(o_c), fx.BFloat16.ir_type)
            fO_bf.store(fx.Vector(vO_final[c]).to(fx.BFloat16).ir_value())
            fx.copy(uni16, thrOc.retile(fO_bf), thrOc.partition_D(o_c), pred=None)
        gpu.barrier()
        out_tile_base = mtile * fx.Int32(O_ELEMS)
        for i in range_constexpr(O_PER_THR):
            fidx = tid + fx.Int32(i * BLOCK_THREADS)
            h = fidx // fx.Int32(V_HEAD_DIM)
            lval = fx.memref_load(l_lds, h)
            oval = fx.Float32(fx.memref_load(o_lds_1d, fidx))
            buffer_ops.buffer_store(fx.BFloat16(oval / lval), out_rsrc, out_tile_base + fidx)

    _vattrs = _sched_value_attrs(BM)
    _hints = _sched_compile_hints(BM)

    @flyc.jit
    def _launch_mfma_jit(
        query2d: fx.Tensor,
        kv_flat: fx.Tensor,
        kv_page_indices: fx.Tensor,
        kv_base: fx.Int32,
        out2d: fx.Tensor,
        total_q: fx.Int32,
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        mfma_kernel(
            query2d, kv_flat, kv_page_indices, kv_base, out2d, seq_len, sm_scale,
            value_attrs=_vattrs,
        ).launch(
            grid=(total_q, NUM_QO_HEADS // BM, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_mfma(*args, **kwargs):
        if _hints is None:
            return _launch_mfma_jit(*args, **kwargs)
        with CompilationContext.compile_hints(_hints):
            return _launch_mfma_jit(*args, **kwargs)

    return launch_mfma


def _build_olds(is_causal: bool = True, bm: int = BLOCK_M, bn: int = BLOCK_N):
    """o_lds round-trip path (FLYDSL_MLA_REGO=0) — the correctness-banked baseline."""
    CAUSAL = bool(is_causal)
    BM = int(bm)
    BN = int(bn)
    HG = NUM_QO_HEADS // BM
    SS = _make_shared(BM, BN)
    GATHER_PER_THR = (BN * QK_HEAD_DIM + BLOCK_THREADS - 1) // BLOCK_THREADS
    GATHER_VEC = 8
    assert QK_HEAD_DIM % GATHER_VEC == 0 and V_HEAD_DIM % GATHER_VEC == 0
    QK_VEC_PER_ROW = QK_HEAD_DIM // GATHER_VEC
    GATHER_VEC_GROUPS = (BN * QK_HEAD_DIM) // GATHER_VEC
    GATHER_VEC_ITERS = (GATHER_VEC_GROUPS + BLOCK_THREADS - 1) // BLOCK_THREADS
    O_ELEMS = BM * V_HEAD_DIM
    O_PER_THR = O_ELEMS // BLOCK_THREADS
    PV_CHUNKS = int(os.environ.get("FLYDSL_MLA_PVC", "4"))
    PV_CHUNK_N = V_HEAD_DIM // PV_CHUNKS  # 128

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def mfma_kernel(
        query2d: fx.Tensor,   # [total_q*128, 576] bf16
        kv_flat: fx.Tensor,   # [num_page*576] bf16
        kv_page_indices: fx.Tensor,  # [num_page] i32
        kv_base: fx.Int32,
        out2d: fx.Tensor,     # [total_q*128, 512] bf16
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
    ):
        row = fx.block_idx.x   # token j
        hg = fx.block_idx.y    # head group
        tid = fx.thread_idx.x

        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = fx.arith.FastMathFlags.fast

        lds = fx.SharedAllocator().allocate(SS).peek()
        k_lds = lds.k_lds.view(fx.make_layout((BN, QK_HEAD_DIM), (K_STRIDE, 1)))
        k_lds_1d = lds.k_lds.view(fx.make_layout(BN * K_STRIDE, 1))
        # V operand for gemm2 is the first V_HEAD_DIM cols of the SAME latent held
        # in k_lds: read it as a transposed view [d, kv] (element (d,kv) at kv*576+d)
        # instead of staging a separate contiguous copy (saves V_HEAD_DIM*BN*2 B LDS).
        vt_lds = lds.k_lds.view(fx.make_layout((V_HEAD_DIM, BN), (1, K_STRIDE)))
        s_lds = lds.s_lds.view(fx.make_layout((BM, BN), (BN, 1)))
        # gemm1's tiled C-store scatters the fragment transposed (physical
        # location holds C[n,m]); write through a column-major view so the
        # row-major read view above then yields the correct C[m,n].
        s_lds_wr = lds.s_lds.view(fx.make_layout((BM, BN), (1, BM)))
        s_lds_1d = lds.s_lds.view(fx.make_layout(BM * BN, 1))
        p_lds = lds.p_lds.view(fx.make_layout((BM, BN), (BN, 1)))
        p_lds_1d = lds.p_lds.view(fx.make_layout(BM * BN, 1))
        o_lds_2d = lds.o_lds.view(fx.make_layout((BM, V_HEAD_DIM), (V_HEAD_DIM, 1)))
        o_lds_1d = lds.o_lds.view(fx.make_layout(O_ELEMS, 1))
        m_lds = lds.m_lds.view(fx.make_layout(BM, 1))
        l_lds = lds.l_lds.view(fx.make_layout(BM, 1))
        corr_lds = lds.corr_lds.view(fx.make_layout(BM, 1))
        corr_bf_lds = lds.corr_bf_lds.view(fx.make_layout(BM, 1))
        # broadcast (v-dim stride 0) view of the bf16 per-head corr, shaped like an O
        # PV-chunk tile so the gemm2 C-copy reads corr[q-row] into the O-fragment layout.
        corr_bcast = lds.corr_bf_lds.view(fx.make_layout((BM, PV_CHUNK_N), (1, 0)))
        phys_lds = lds.phys_lds.view(fx.make_layout(BN, 1))

        kvidx_rsrc = buffer_ops.create_buffer_resource(kv_page_indices)
        kv_rsrc = buffer_ops.create_buffer_resource(kv_flat)
        out_rsrc = buffer_ops.create_buffer_resource(out2d)

        qb = fx.rocdl.make_buffer_tensor(query2d)
        mtile = row * fx.Int32(HG) + hg
        bA = fx.slice(fx.zipped_divide(qb, (BM, QK_HEAD_DIM)), (None, mtile))
        # shape token [BM,BN] for gemm1 C fragment (only the layout shape is used)
        bS = fx.slice(fx.zipped_divide(qb, (BM, BN)), (None, fx.Int32(0)))

        mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tiled = fx.make_tiled_mma(mma, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr = tiled.thr_slice(tid)
        ca16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.BFloat16)
        uni16 = fx.make_copy_atom(fx.UniversalCopy(16), fx.BFloat16)
        uni32 = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)

        # ---- Q resident in registers (gemm1 A operand), loaded once ----
        thrA = fx.make_tiled_copy_A(ca16, tiled).get_slice(tid)
        fA = thr.make_fragment_A(bA)
        fx.copy(ca16, thrA.partition_S(bA), thrA.retile(fA), pred=None)

        # ---- init running state in LDS ----
        for i in range_constexpr(O_PER_THR):
            fx.memref_store(fx.BFloat16(0.0), o_lds_1d, tid + fx.Int32(i * BLOCK_THREADS))
        if tid < fx.Int32(BM):
            fx.memref_store(c_neg_inf, m_lds, tid)
            fx.memref_store(c_zero_f, l_lds, tid)
        gpu.barrier()

        if const_expr(CAUSAL):
            n_len = row + fx.Int32(1)
        else:
            n_len = seq_len
        num_tiles = (fx.Index(n_len) + fx.Index(BN - 1)) // fx.Index(BN)

        thrB = fx.make_tiled_copy_B(uni16, tiled).get_slice(tid)
        thrCst = fx.make_tiled_copy_C(uni32, tiled).get_slice(tid)     # store S -> s_lds
        thrA2 = fx.make_tiled_copy_A(uni16, tiled).get_slice(tid)      # p_lds -> fP
        thrB2 = fx.make_tiled_copy_B(uni16, tiled).get_slice(tid)      # vt_lds -> fV
        thrOc = fx.make_tiled_copy_C(uni16, tiled).get_slice(tid)      # o_lds(bf16) <-> fO(f32)

        for t_idx, _st in range(fx.Index(0), num_tiles, fx.Index(1), init=[fx.Int32(0)]):
            t_i32 = fx.Int32(t_idx)
            kv_tile0 = t_i32 * fx.Int32(BN)

            # ---- resolve paged KV rows for this tile ----
            if tid < fx.Int32(BN):
                pos = kv_base + kv_tile0 + tid
                valid_row = (kv_tile0 + tid) < n_len
                pos_safe = valid_row.select(pos, kv_base)
                ph = fx.Int32(buffer_ops.buffer_load(kvidx_rsrc, pos_safe, vec_width=1, dtype=fx.Int32))
                fx.memref_store(ph, phys_lds, tid)
            gpu.barrier()

            # ---- gather KV tile into LDS (K row-major + V transposed) ----
            # Vectorized: 8 contiguous head-dims per load (QK_HEAD_DIM % 8 == 0),
            # so each group stays within one KV row and one V/rope region.
            for i in range_constexpr(GATHER_VEC_ITERS):
                g = tid + fx.Int32(i * BLOCK_THREADS)
                if g < fx.Int32(GATHER_VEC_GROUPS):
                    d0 = (g % fx.Int32(QK_VEC_PER_ROW)) * fx.Int32(GATHER_VEC)
                    n = g // fx.Int32(QK_VEC_PER_ROW)
                    ph = fx.memref_load(phys_lds, n)
                    vv = fx.Vector(
                        buffer_ops.buffer_load(
                            kv_rsrc, ph * fx.Int32(QK_HEAD_DIM) + d0, vec_width=GATHER_VEC, dtype=fx.BFloat16
                        )
                    )
                    base_k = n * fx.Int32(K_STRIDE) + d0
                    for j in range_constexpr(GATHER_VEC):
                        fx.memref_store(fx.BFloat16(vv[j]), k_lds_1d, base_k + fx.Int32(j))
            gpu.barrier()

            # ---- gemm1: S = Q @ K^T ----
            fB = thr.make_fragment_B(k_lds)
            fx.copy(uni16, thrB.partition_S(k_lds), thrB.retile(fB), pred=None)
            fS = thr.make_fragment_C(bS)
            fS.fill(0)
            fx.gemm(mma, fS, fA, fB, fS)
            fx.copy(uni32, thrCst.retile(fS), thrCst.partition_D(s_lds), pred=None)
            gpu.barrier()

            # ---- online softmax on s_lds — PARALLELIZED across TPR threads/row ----
            # Each head-row is reduced by TPR=BLOCK_THREADS//BM threads (all 256
            # threads active, vs 32 before); each owns CPR=BN//TPR kv columns and
            # the per-row max/sum are combined by an intra-wave shuffle_xor reduce
            # over the TPR-aligned lane group.
            TPR = BLOCK_THREADS // BM          # threads per head-row (8)
            CPR = BN // TPR                    # kv columns per thread (4)
            srow = tid // fx.Int32(TPR)        # head row 0..BM-1
            ssub = tid % fx.Int32(TPR)         # sub-lane 0..TPR-1
            m_row = fx.memref_load(m_lds, srow)
            l_row = fx.memref_load(l_lds, srow)
            s_loc = []
            v_loc = []
            local_max = c_neg_inf
            for cc in range_constexpr(CPR):
                ncol = ssub * fx.Int32(CPR) + fx.Int32(cc)
                pos_ok = (kv_tile0 + ncol) < n_len
                scaled = fx.memref_load(s_lds_1d, srow * fx.Int32(BN) + ncol) * sm_scale
                s_loc.append(scaled)
                v_loc.append(pos_ok)
                local_max = local_max.maximumf(pos_ok.select(scaled, c_neg_inf))
            tmax = local_max
            for off in (1, 2, 4, 8, 16):
                if const_expr(off < TPR):
                    tmax = tmax.maximumf(tmax.shuffle_xor(off, 64))
            m_new = m_row.maximumf(tmax)
            corr = fmath.exp2((m_row - m_new) * c_log2e, fastmath=fm_fast)
            local_sum = c_zero_f
            for cc in range_constexpr(CPR):
                ncol = ssub * fx.Int32(CPR) + fx.Int32(cc)
                p = v_loc[cc].select(
                    fmath.exp2((s_loc[cc] - m_new) * c_log2e, fastmath=fm_fast), c_zero_f
                )
                fx.memref_store(fx.BFloat16(p), p_lds_1d, srow * fx.Int32(BN) + ncol)
                local_sum = local_sum + p
            tsum = local_sum
            for off in (1, 2, 4, 8, 16):
                if const_expr(off < TPR):
                    tsum = tsum + tsum.shuffle_xor(off, 64)
            l_new = l_row * corr + tsum
            if ssub == fx.Int32(0):
                fx.memref_store(m_new, m_lds, srow)
                fx.memref_store(l_new, l_lds, srow)
                fx.memref_store(corr, corr_lds, srow)
            gpu.barrier()

            # ---- rescale O accumulator by per-head corr ----
            for i in range_constexpr(O_PER_THR):
                fidx = tid + fx.Int32(i * BLOCK_THREADS)
                h = fidx // fx.Int32(V_HEAD_DIM)
                cval = fx.memref_load(corr_lds, h)
                oval = fx.Float32(fx.memref_load(o_lds_1d, fidx))
                fx.memref_store(fx.BFloat16(oval * cval), o_lds_1d, fidx)
            gpu.barrier()

            # ---- gemm2: O += P @ V, chunked over the v-dim to shrink the peak
            # B/C fragments (raises occupancy). fP is shared across chunks. ----
            fP = thr.make_fragment_A(p_lds)
            fx.copy(uni16, thrA2.partition_S(p_lds), thrA2.retile(fP), pred=None)
            vt_div = fx.zipped_divide(vt_lds, (PV_CHUNK_N, BN))
            o_div = fx.zipped_divide(o_lds_2d, (BM, PV_CHUNK_N))
            for c in range_constexpr(PV_CHUNKS):
                vt_c = fx.slice(vt_div, (None, fx.Int32(c)))
                o_c = fx.slice(o_div, (None, fx.Int32(c)))
                fV_c = thr.make_fragment_B(vt_c)
                fx.copy(uni16, thrB2.partition_S(vt_c), thrB2.retile(fV_c), pred=None)
                fO_c = thr.make_fragment_C(o_c)                       # f32 accumulator (from MMA atom)
                fO_bf = fx.make_fragment_like(fO_c, fx.BFloat16.ir_type)
                fx.copy(uni16, thrOc.partition_S(o_c), thrOc.retile(fO_bf), pred=None)  # bf16 LDS -> bf16 reg
                fO_c.store(fx.Vector(fO_bf.load()).to(fx.Float32).ir_value())           # bf16 -> f32
                fx.gemm(mma, fO_c, fP, fV_c, fO_c)
                fO_bf.store(fx.Vector(fO_c.load()).to(fx.BFloat16).ir_value())          # f32 -> bf16
                fx.copy(uni16, thrOc.retile(fO_bf), thrOc.partition_D(o_c), pred=None)  # bf16 reg -> bf16 LDS
            gpu.barrier()
            _res = yield [fx.Int32(0)]

        # ---- epilogue: normalize (O / l) and store bf16 to out ----
        out_tile_base = mtile * fx.Int32(O_ELEMS)
        for i in range_constexpr(O_PER_THR):
            fidx = tid + fx.Int32(i * BLOCK_THREADS)
            h = fidx // fx.Int32(V_HEAD_DIM)
            lval = fx.memref_load(l_lds, h)
            oval = fx.Float32(fx.memref_load(o_lds_1d, fidx))
            buffer_ops.buffer_store(fx.BFloat16(oval / lval), out_rsrc, out_tile_base + fidx)

    @flyc.jit
    def launch_mfma(
        query2d: fx.Tensor,
        kv_flat: fx.Tensor,
        kv_page_indices: fx.Tensor,
        kv_base: fx.Int32,
        out2d: fx.Tensor,
        total_q: fx.Int32,
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        mfma_kernel(
            query2d, kv_flat, kv_page_indices, kv_base, out2d, seq_len, sm_scale
        ).launch(
            grid=(total_q, NUM_QO_HEADS // BM, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_mfma

def build_mla_prefill_mfma_module(is_causal: bool = True, bm: int = BLOCK_M, bn: int = BLOCK_N):
    """Dispatch to the register-resident O path (FLYDSL_MLA_REGO=1, default) or the
    o_lds round-trip fallback (FLYDSL_MLA_REGO=0)."""
    if _REGO:
        return _build_rego(is_causal, bm, bn)
    return _build_olds(is_causal, bm, bn)
