# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: E741, F841

"""DeepSeek-V4 sparse-MLA training backward for gfx950/MI355X.

All backward kernels (delta / dq / interm / gather / fused) + host dispatch, single file.
"""

from __future__ import annotations

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.common import buffer_ops

__all__ = ["sparse_mla_bwd_flydsl"]


def make_bf16_rebased_rsrc(arg, base_elems, num_records_bytes):
    """Create a raw BF16 buffer SRD with a 64-bit rebased base address."""
    base = arith.index_cast(T.i64, buffer_ops.extract_base_index(arg))
    base = base + arith.index_cast(T.i64, base_elems) * fx.Int64(2)
    records = arith.minui(
        arith.index_cast(T.index, num_records_bytes),
        arith.index(0xFFFFFFFF),
    )
    records = _raw(fx.Int64(arith.index_cast(T.i64, records)))
    flags = buffer_ops._get_buffer_flags()
    base_ptr = _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr"), _raw(base)).result
    return rocdl.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"),
        base_ptr,
        buffer_ops._create_i16_constant(0),
        records,
        buffer_ops._create_i32_constant(flags),
    ).result


def make_value_attrs(waves_per_eu, agpr_alloc, flat_work_group_size):
    """Build kernel attributes, optionally pinning the AGPR allocation."""
    attrs = {
        "rocdl.waves_per_eu": waves_per_eu,
        "rocdl.flat_work_group_size": flat_work_group_size,
    }
    if agpr_alloc:
        allocation = f"0,{-agpr_alloc}" if agpr_alloc < 0 else f"{agpr_alloc},{agpr_alloc}"
        attrs["passthrough"] = [
            ["amdgpu-agpr-alloc", allocation],
            ["amdgpu-mfma-vgpr-form", "false"],
        ]
    return attrs


# ---- shared physical constants ----
_LOG2E = 1.4426950408889634
D = 512  # value head_dim
DQK = 576  # QK head_dim (incl. rope)
D_V = D  # alias used by gather / interm sections


def _attach(launch):
    """Attach the standard .compile() wrapper and return the launch closure."""
    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def _cached_run(cache, key, launch, args):
    """Compile ``launch`` for ``args`` once (memoized on ``key`` in ``cache``) and run it."""
    compiled = cache.get(key)
    if compiled is None:
        compiled = launch.compile(*args)
        cache[key] = compiled
    compiled(*args)


def _concat_bf16_v8(a, b):
    """Concat two v4 int16 into a direct v8 bf16 (plain register concat, no shuffle
    crossbar; 16x16x32 MFMA operands must be direct-v8)."""
    va = Vec(a)
    vb = Vec(b)
    return _raw(
        Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], fx.Int16).bitcast(fx.BFloat16)
    )


# ============================================================================
# kernel: delta / dsink
# ============================================================================

DELTA_THREADS = 64
DELTA_EPL = D // DELTA_THREADS  # 8 elems per lane
DSINK_THREADS = 256


def build_delta():
    elem = fx.BFloat16
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_delta_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + DELTA_THREADS * 4  # 64 fp32 partials

    @flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
    def k_fn(O: fx.Tensor, DO: fx.Tensor, DELTA: fx.Tensor, NROWS: fx.Int32):
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(DELTA_THREADS,)).get()
        row = fx.Index(gpu.block_idx.x)
        lane = fx.Index(gpu.thread_idx.x)
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(D * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(D * 2))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(NROWS) * fx.Index(4))
        )

        base = row * fx.Index(D) + lane * fx.Index(DELTA_EPL)
        ov = buffer_ops.buffer_load(o_rsrc, base, vec_width=8, dtype=elem)
        dov = buffer_ops.buffer_load(do_rsrc, base, vec_width=8, dtype=elem)

        partial = fx.Float32(0.0)
        for i in range_constexpr(DELTA_EPL):
            oi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(ov)[i])).result)
            di = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(dov)[i])).result)
            partial = fx.Float32(arith.AddFOp(_raw(partial), _raw(oi * di)).result)

        Vec.from_elements([partial], fx.Float32).store(lds, [lane])
        gpu.barrier()
        # full-wave reduction by lane 0 (delta kernel is tiny; serial 64-sum is fine).
        total = fx.Float32(0.0)
        for j in range_constexpr(DELTA_THREADS):
            total = fx.Float32(
                arith.AddFOp(_raw(total), _raw(Vec.load(Vec.make_type(1, fx.Float32), lds, [fx.Index(j)])[0])).result
            )
        buffer_ops.buffer_store(
            total,
            delta_rsrc,
            row * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(lane), _raw(fx.Index(0))).result),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(O, DO, DELTA, NROWS, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(O, DO, DELTA, NROWS).launch(grid=(fx.Index(NROWS), 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream)

    return _attach(launch)


DSINK_TB = 64  # tokens per pass-1 WG (tb*H must be a multiple of DSINK_THREADS)


def build_dsink_split(T_LEN, H, tb=DSINK_TB):
    """Pass 1 (coalesced) of split d_sink: WG b owns a tb-token slice, reads full rows
    contiguously, each thread accumulates head (tid % H) across its rows, combined via
    LDS into partial[b, h]. OOB tail reads return 0; fp32 accumulate."""
    assert DSINK_THREADS % H == 0 and (tb * H) % DSINK_THREADS == 0
    TPH = DSINK_THREADS // H
    STEPS = (tb * H) // DSINK_THREADS
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dsink_p1_smem")
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + DSINK_THREADS * 4

    @flyc.kernel(known_block_size=[DSINK_THREADS, 1, 1])
    def k_fn(LSE: fx.Tensor, DELTA: fx.Tensor, SINK: fx.Tensor, PARTIAL: fx.Tensor, T: fx.Int32, NBLK: fx.Int32):
        lds = SmemPtr(allocator.get_base(), lds_off, fx.Float32.ir_type, shape=(DSINK_THREADS,)).get()
        b = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        Hn = fx.Index(H)
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        sink_rsrc = buffer_ops.create_buffer_resource(SINK, max_size=False, num_records_bytes=_raw(Hn * fx.Index(4)))
        part_rsrc = buffer_ops.create_buffer_resource(
            PARTIAL, max_size=False, num_records_bytes=_raw(fx.Index(NBLK) * Hn * fx.Index(4))
        )
        c_log2e = fx.Float32(_LOG2E)
        head = tid % Hn
        sink_h = fx.Float32(buffer_ops.buffer_load(sink_rsrc, head, vec_width=1, dtype=fx.Float32))

        base = b * fx.Index(tb) * Hn + tid  # first (contiguous) element this thread reads
        acc = fx.Float32(0.0)
        for s in range_constexpr(STEPS):
            g = base + fx.Index(s * DSINK_THREADS)
            lse_g = fx.Float32(buffer_ops.buffer_load(lse_rsrc, g, vec_width=1, dtype=fx.Float32))
            delta_g = fx.Float32(buffer_ops.buffer_load(delta_rsrc, g, vec_width=1, dtype=fx.Float32))
            e = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((sink_h - lse_g) * c_log2e)))
            acc = fx.Float32(arith.AddFOp(_raw(acc), _raw(e * delta_g)).result)

        Vec.from_elements([acc], fx.Float32).store(lds, [tid])
        gpu.barrier()
        # combine the TPH threads sharing head=tid (tid<H): lds[tid + k*H], k in 0..TPH-1
        psum = fx.Float32(0.0)
        for k in range_constexpr(TPH):
            psum = fx.Float32(
                arith.AddFOp(
                    _raw(psum), _raw(Vec.load(Vec.make_type(1, fx.Float32), lds, [tid + fx.Index(k * H)])[0])
                ).result
            )
        buffer_ops.buffer_store(
            psum,
            part_rsrc,
            (b * Hn + tid) * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.slt, _raw(tid), _raw(Hn)).result),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(LSE, DELTA, SINK, PARTIAL, T, NBLK, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(LSE, DELTA, SINK, PARTIAL, T, NBLK).launch(
            grid=(fx.Index(NBLK), 1, 1), block=(DSINK_THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


def build_dsink_reduce(NBLK, H):
    """Pass 2 of the coalesced split d_sink: d_sink[h] = -sum_b partial[b, h]. One WG of
    DSINK_THREADS; thread tid (< H) sums the NBLK block-partials of head tid, then negates."""
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dsink_p2_smem")

    @flyc.kernel(known_block_size=[DSINK_THREADS, 1, 1])
    def k_fn(PARTIAL: fx.Tensor, DSINK: fx.Tensor):
        tid = fx.Index(gpu.thread_idx.x)
        Hn = fx.Index(H)
        part_rsrc = buffer_ops.create_buffer_resource(
            PARTIAL, max_size=False, num_records_bytes=_raw(fx.Index(NBLK) * Hn * fx.Index(4))
        )
        dsink_rsrc = buffer_ops.create_buffer_resource(DSINK, max_size=False, num_records_bytes=_raw(Hn * fx.Index(4)))
        total = fx.Float32(0.0)
        for b in range_constexpr(NBLK):
            v = fx.Float32(buffer_ops.buffer_load(part_rsrc, fx.Index(b * H) + tid, vec_width=1, dtype=fx.Float32))
            total = fx.Float32(arith.AddFOp(_raw(total), _raw(v)).result)
        neg = fx.Float32(0.0) - total
        buffer_ops.buffer_store(
            neg,
            dsink_rsrc,
            tid * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.slt, _raw(tid), _raw(Hn)).result),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(PARTIAL, DSINK, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(PARTIAL, DSINK).launch(grid=(1, 1, 1), block=(DSINK_THREADS, 1, 1), stream=stream)

    return _attach(launch)


# ============================================================================
# kernel: dq
# ============================================================================

BLOCK_H = 64
HPW = 16
WAVES = BLOCK_H // HPW  # 4
THREADS = WAVES * 64  # 256
TILE_K = 16
TKP = 16
KS = D // 32  # 16 QK MFMA K-steps
DT = D // 16  # 32 PV d-tiles


def build_bwd_dq(
    topk_len,
    scale,
    qk_prefetch=4,
    num_heads=None,
    pv_prefetch=8,
    delta_dot2=False,
):
    # qk_prefetch=4 (QK KV-read prefetch depth): this kernel runs 2-acc QK (_QK4=0 for its pro cr0/cr128
    # shapes), whose 4 chains consume operands slower than the 8-acc path -> the prefetch optimum
    # shifts shallower (qk_prefetch=4 beats 6/3 isolated; pf6 was the 8-acc optimum). pv_prefetch=8 unchanged.
    elem = fx.BFloat16
    # D_LDS: KV LDS row pad. 528 (264 dword %32=8) avoids the QK natural-v8 read's 2-way
    # bank conflict.
    D_LDS = 528
    DTE = DT
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    LDS_ELEMS = TKP * D_LDS
    # KVBUF: KV-LDS double-buffer depth (2 overlaps next-tile LDS store with current-tile
    # LDS reads, no read->store WAR barrier).
    KVBUF = 2
    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_dq_smem")
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + KVBUF * LDS_ELEMS * 2, 16)
    allocator.ptr = allocator._align(mask_off + KVBUF * TILE_K * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_kv = SmemPtr(allocator.get_base(), kv_off, elem.ir_type, shape=(KVBUF * LDS_ELEMS,)).get()
        lds_mask = SmemPtr(allocator.get_base(), mask_off, fx.Float32.ir_type, shape=(KVBUF * TILE_K,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)

        token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)
        head_wave_base = hg * fx.Index(BLOCK_H) + wave * fx.Index(HPW)
        head_A = head_wave_base + lo

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2))
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2))
        )
        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)  # scale folded into exp2 base (saves a softmax mul)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)

        # ---- Q and dO (B operands): head=head_A ----
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)

        def load_q(ks):
            return buffer_ops.buffer_load(
                q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )

        def load_do(ks):
            return buffer_ops.buffer_load(
                do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )

        # Q/dO register-resident (reused across tiles), amortizing the load.
        q_packs = [load_q(ks) for ks in range_constexpr(KS)]
        do_packs = [load_do(ks) for ks in range_constexpr(KS)]

        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        # Clamp lse to finite: a fully-degenerate head (all kv invalid) has lse=-inf, giving
        # NaN P; clamping makes -inf-(-3e38)=-inf -> P=0 (matches triton's P mask).
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)  # loop-invariant: hoist lse*log2e out of tile loop
        # FOLD delta = rowsum(O*dO) into dQ (was a separate BW-bound kernel). dO already resident
        # in do_packs; stream O (1 vec8 at a time), fp32-accumulate the bf16 product, then sum the
        # 4 d-groups (lanes lo,lo+16,lo+32,lo+48) via ds_bpermute.
        o_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        _dpart = fx.Float32(0.0)

        def _dot2_bf16(a, b, acc):
            op = _llvm.InlineAsmOp(
                res=ir.F32Type.get(),
                operands_=[_raw(a), _raw(b), _raw(acc)],
                asm_string="v_dot2_f32_bf16 $0, $1, $2, $0",
                constraints="=v,v,v,0",
                has_side_effects=False,
            )
            return fx.Float32(op.result)

        for _ks in range_constexpr(KS):
            _ov = buffer_ops.buffer_load(
                o_rsrc, o_row + fx.Index(_ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            if const_expr(delta_dot2):
                # Consume adjacent bf16 pairs in their original order through one fp32
                # accumulator. Only the pair-internal rounding differs from the scalar path.
                _op = Vec(_ov).bitcast(fx.Int32)
                _dop = Vec(do_packs[_ks]).bitcast(fx.Int32)
                for _i in range_constexpr(4):
                    _dpart = _dot2_bf16(_op[_i], _dop[_i], _dpart)
            else:
                for _i in range_constexpr(8):
                    _oi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(_ov)[_i])).result)
                    _di = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(do_packs[_ks])[_i])).result)
                    _dpart = fx.Float32(arith.AddFOp(_raw(_dpart), _raw(_oi * _di)).result)
        _part_i = _raw(Vec.from_elements([_dpart], fx.Float32).bitcast(fx.Int32)[0])
        delta_h = fx.Float32(0.0)
        for _g in range_constexpr(4):
            _idx = _raw(fx.Int32(lo + fx.Index(_g * 16)) * fx.Int32(4))
            _gi = rocdl.ds_bpermute(fx.Int32.ir_type, _idx, _part_i)
            _gf = fx.Float32(_raw(Vec.from_elements([fx.Int32(_gi)], fx.Int32).bitcast(fx.Float32)[0]))
            delta_h = fx.Float32(arith.AddFOp(_raw(delta_h), _raw(_gf)).result)
        buffer_ops.buffer_store(
            _raw(delta_h),
            delta_rsrc,
            (token * Hn + head_A) * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(grp), _raw(fx.Index(0))).result),
            offset_is_bytes=True,
        )

        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tbase):
            return fx.Int32(buffer_ops.buffer_load(tk_rsrc, tk_row + tbase + g_row, vec_width=1, dtype=fx.Int32))

        def gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(4)
            ]

        def gather_store(vvs, idx, buf_off, mbuf_off):
            valid = ArithValue(idx >= fx.Int32(0))
            for c in range_constexpr(4):
                Vec(vvs[c]).store(
                    lds_kv, [buf_off + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)]
                )
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mbuf_off + g_row])

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DTE)]
        idxA0 = load_topk(fx.Index(0))
        kv0 = gather_load(idxA0)
        idxB0 = load_topk(fx.Index(TILE_K))
        init = list(dq_acc0) + list(kv0) + [idxA0, idxB0]

        loop_results = init
        for t, iter_args in range(fx.Index(0), fx.Index(NUM_TILES), fx.Index(1), init=init):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DTE)]
            kv_cur = [iter_args[DTE + c] for c in range_constexpr(4)]
            idxA = iter_args[DTE + 4]
            idxB = iter_args[DTE + 5]

            buf_off = (t % fx.Index(KVBUF)) * fx.Index(LDS_ELEMS)
            mbuf_off = (t % fx.Index(KVBUF)) * fx.Index(TILE_K)
            gather_store(kv_cur, idxA, buf_off, mbuf_off)
            # Issue next-tile HBM prefetches (KV data + tile+2 topk index) BEFORE the barrier:
            # they are vmcnt loads independent of the LDS store/barrier, so their issue overlaps
            # the barrier's lgkmcnt drain + s_barrier wait (fills the occ-1 sync bubble).
            kv_next = gather_load(idxB)
            idxB2 = load_topk((t + fx.Index(2)) * fx.Index(TILE_K))
            gpu.barrier()

            mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])

            # QK x2: S (B=Q) and dP (B=dO), sharing the gathered KV bv[ks]. occ-1 (single WG):
            # split each depth-16 MFMA chain into 2 accumulators so the RAW MFMA latency is
            # hidden by ILP (dQ has no neighbour WG to hide the chain).
            acc_s0 = Vec.filled(4, 0.0, fx.Float32)
            acc_s1 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp0 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp1 = Vec.filled(4, 0.0, fx.Float32)
            # QK4: 4 acc/operand (8 chains) vs 2. Gated to large-topk (topk>=512): this kernel
            # only serves pro cr0/cr128 few-tile shapes where the 8-acc's extra VGPR/chains don't
            # amortize -> 2-acc is faster. (Large-R pro cr4 / flash cr4 use pv2x, which keeps 8-acc.)
            _QK4 = 1 if topk_len >= 512 else 0
            acc_s2 = Vec.filled(4, 0.0, fx.Float32)
            acc_s3 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp2 = Vec.filled(4, 0.0, fx.Float32)
            acc_dp3 = Vec.filled(4, 0.0, fx.Float32)

            def _bv(ks, buf_off=buf_off):
                return Vec.load(v8, lds_kv, [buf_off + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)])

            def _q(ks):
                return q_packs[ks]

            def _do(ks):
                return do_packs[ks]

            PF = qk_prefetch  # KV-read prefetch depth (build param, tuned for large-topk pro).
            bvq = [_bv(k) for k in range_constexpr(PF)]
            for ks in range_constexpr(KS):
                if ks + PF < KS:
                    bvq.append(_bv(ks + PF))
                if const_expr(_QK4):
                    r = ks % 4
                    accs = [acc_s0, acc_s1, acc_s2, acc_s3]
                    accd = [acc_dp0, acc_dp1, acc_dp2, acc_dp3]
                    accs[r] = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), accs[r]])
                    accd[r] = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), accd[r]])
                    acc_s0, acc_s1, acc_s2, acc_s3 = accs
                    acc_dp0, acc_dp1, acc_dp2, acc_dp3 = accd
                elif ks % 2 == 0:
                    acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), acc_s0])
                    acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), acc_dp0])
                else:
                    acc_s1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _q(ks), acc_s1])
                    acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), _do(ks), acc_dp1])
            if const_expr(_QK4):
                acc_s = Vec.from_elements(
                    [
                        fx.Float32(_raw(Vec(acc_s0)[i]))
                        + fx.Float32(_raw(Vec(acc_s1)[i]))
                        + fx.Float32(_raw(Vec(acc_s2)[i]))
                        + fx.Float32(_raw(Vec(acc_s3)[i]))
                        for i in range_constexpr(4)
                    ],
                    fx.Float32,
                )
                acc_dp = Vec.from_elements(
                    [
                        fx.Float32(_raw(Vec(acc_dp0)[i]))
                        + fx.Float32(_raw(Vec(acc_dp1)[i]))
                        + fx.Float32(_raw(Vec(acc_dp2)[i]))
                        + fx.Float32(_raw(Vec(acc_dp3)[i]))
                        for i in range_constexpr(4)
                    ],
                    fx.Float32,
                )
            else:
                acc_s = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i])) for i in range_constexpr(4)],
                    fx.Float32,
                )
                acc_dp = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i])) for i in range_constexpr(4)],
                    fx.Float32,
                )

            # P = exp2(acc_s*c_sl + mask4 - lse_l2) (scale*log2e folded into c_sl, mask4 is
            # {0,-inf}). PV A-operand (V ds_read_tr16) address depends only on buf_off+lane, so
            # the first PVPF tr16 reads are hoisted ahead of the exp2 loop (overlap softmax VALU).
            lo_d4 = lo // fx.Index(4)
            lo_m4 = lo % fx.Index(4)
            pv_base = fx.Int64(
                (buf_off + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2)
                + fx.Index(kv_off)
            )

            def _bvv(dt, pv_base=pv_base):
                ptr = buffer_ops.create_llvm_ptr(_raw(pv_base + fx.Int64(dt * 32)), address_space=3)
                return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

            _PVPF = (
                pv_prefetch if (pv_prefetch and ((num_heads is not None and num_heads > 64) or topk_len >= 512)) else 0
            )
            bvv_pf = [_bvv(dt) for dt in range_constexpr(_PVPF)] if const_expr(_PVPF) else None

            # dS = P*(dP - delta)*scale; when masked P=0 -> dS=0 (no explicit dP mask).
            pvals = [None] * 4
            dsvals = [None] * 4
            for i in range_constexpr(4):
                arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                pvals[i] = p
                _dp = fx.Float32(_raw(Vec(acc_dp)[i]))
                dsvals[i] = p * (_dp - delta_h) * c_scale

            # dS as PV B-operand (k=kv=grp*4+i, n=head=lo).
            pB = _raw(
                Vec.from_elements([fx.BFloat16(_raw(dsvals[i])) for i in range_constexpr(4)], elem).bitcast(fx.Int16)
            )

            # dQ += dS @ K (PV): B=dS, A=V(ds_read_tr16), accumulate into dq_acc[dt] over tiles.
            # dq_acc pinned in AGPR via inline-asm MFMA (=a,v,v,0, D=C in-place, no accvgpr
            # shuffle) -> frees 128 arch-VGPR -> occ-2.
            _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

            def _mma_ag(a, b, c, _v4f32_ir=_v4f32_ir):
                op = _llvm.InlineAsmOp(
                    res=_v4f32_ir,
                    operands_=[_raw(a), _raw(b), _raw(c)],
                    asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, $0",
                    constraints="=a,v,v,0",
                    has_side_effects=False,
                )
                return op.result

            def _pv(bvv, c, pB=pB):
                return _mma_ag(bvv, pB, c)

            new_dq = [None] * DT
            if const_expr(_PVPF):
                for dt in range_constexpr(DT):
                    if dt + _PVPF < DT:
                        bvv_pf.append(_bvv(dt + _PVPF))
                    new_dq[dt] = _pv(bvv_pf[dt], dq_acc[dt])
            else:
                for dt in range_constexpr(DT):
                    new_dq[dt] = _pv(_bvv(dt), dq_acc[dt])

            # store dS, P to [T,H,topk] bf16 (rank = tile_start + grp*4 + i, 4 contiguous).
            ts = t * fx.Index(TILE_K)
            sp_base = ((token * Hn + head_A) * fx.Index(topk_len) + ts + grp * fx.Index(4)) * fx.Index(2)
            # pack the two adjacent bf16-pk dwords into one dwordx2 store (halves the dS/P
            # store count; rank = grp*4+i are 4 contiguous, so [pk0,pk1] cover 4 ranks).
            ds_pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[0]), _raw(dsvals[1]))))
            ds_pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[2]), _raw(dsvals[3]))))
            pp_pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[0]), _raw(pvals[1]))))
            pp_pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[2]), _raw(pvals[3]))))
            ds_v2 = Vec.from_elements([ds_pk0, ds_pk1], fx.Int32)
            pp_v2 = Vec.from_elements([pp_pk0, pp_pk1], fx.Int32)
            buffer_ops.buffer_store(_raw(ds_v2), ds_rsrc, sp_base, offset_is_bytes=True)
            buffer_ops.buffer_store(_raw(pp_v2), pp_rsrc, sp_base, offset_is_bytes=True)

            loop_results = yield (list(new_dq) + list(kv_next) + [idxB, idxB2])

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        head_i = head_wave_base + lo
        # dQ[token, head_i, 0:512] bf16.
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (
                token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        # Zero the 64 rope cols (512..575) in-kernel (rope grad is dead), replacing the
        # strided host dq[..., 512:].zero_(). 4 d-tiles x (grp*4+[0..3]) cover 512..575.
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (
                token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        gy = fx.Index(H) // fx.Index(BLOCK_H)
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, DS, PP, O, T, H, NKV).launch(
            grid=(fx.Index(T), gy, 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


def build_bwd_dq_resident_kv_cr0_h128(
    scale,
    qk_prefetch=4,
    pv_prefetch=8,
    prologue_g2l_overlap=False,
):
    """Exact Pro-cr0 path: one H64 workgroup processes both H64 head groups.

    The workgroup stages all R=128 live KV rows once, retains them in padded LDS, and
    runs the ordered-dot2/QK-PF4 dQ arithmetic twice with an audited PV prefetch
    depth. Only one H64 dQ accumulator set is live at a time; dS/P and dQ stores
    use the current head group offset. The exact host-side guard selects this
    builder by default; strict environment overrides retain the accepted PF8
    schedule and accepted two-workgroup implementation for A/B.
    """

    elem = fx.BFloat16
    TOPK_LEN = 128
    NUM_HEADS = 128
    NUM_TILES = TOPK_LEN // TILE_K
    D_LDS = 528
    RESIDENT_LDS_ELEMS = TOPK_LEN * D_LDS
    RESIDENT_LDS_BYTES = RESIDENT_LDS_ELEMS * 2 + TOPK_LEN * 4
    assert RESIDENT_LDS_BYTES == 135680
    assert RESIDENT_LDS_BYTES <= 160 * 1024
    assert qk_prefetch == 4 and pv_prefetch in {2, 8, 10}
    prologue_g2l_overlap = bool(prologue_g2l_overlap)

    smem_symbol = (
        "mla_bwd_dq_resident_kv_cr0_h128_g2l_overlap_smem"
        if prologue_g2l_overlap
        else "mla_bwd_dq_resident_kv_cr0_h128_smem"
    )
    allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=smem_symbol,
    )
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + RESIDENT_LDS_ELEMS * 2, 16)
    allocator.ptr = allocator._align(mask_off + TOPK_LEN * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_kv = SmemPtr(
            allocator.get_base(),
            kv_off,
            elem.ir_type,
            shape=(RESIDENT_LDS_ELEMS,),
        ).get()
        lds_mask = SmemPtr(
            allocator.get_base(),
            mask_off,
            fx.Float32.ir_type,
            shape=(TOPK_LEN,),
        ).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)

        token = fx.Index(gpu.block_idx.x)
        Hn = fx.Index(H)

        q_rsrc = buffer_ops.create_buffer_resource(
            Q,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)),
        )
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV,
            max_size=False,
            num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)),
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)),
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)),
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * fx.Index(TOPK_LEN * 4)),
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)),
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)),
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)),
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(TOPK_LEN * 2)),
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(TOPK_LEN * 2)),
        )
        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        tk_row = token * fx.Index(TOPK_LEN)

        def _dot2_bf16(a, b, acc):
            op = _llvm.InlineAsmOp(
                res=ir.F32Type.get(),
                operands_=[_raw(a), _raw(b), _raw(acc)],
                asm_string="v_dot2_f32_bf16 $0, $1, $2, $0",
                constraints="=v,v,v,0",
                has_side_effects=False,
            )
            return fx.Float32(op.result)

        if const_expr(prologue_g2l_overlap):
            # One lane in each wave owns one of that wave's 32 resident row indices.
            # Resolve this compact TOPK preload before the direct-to-LDS DMA burst;
            # each step broadcasts its source index within the wave.
            idx_lane_rank = (lane // fx.Index(4)) * fx.Index(TILE_K) + wave * fx.Index(4) + lane % fx.Index(4)
            idx_lane = fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + idx_lane_rank,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=_raw(ArithValue(lane) < fx.Index(32)),
                )
            )
            idx_lane_valid = ArithValue(idx_lane >= fx.Int32(0))
            if lane < fx.Index(32):
                mask = fx.Float32(idx_lane_valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([mask], fx.Float32).store(
                    lds_mask,
                    [idx_lane_rank],
                )

            # Hoist the accepted head-group-0 setup. O packs 0:8 form a bounded
            # register prefetch window whose exact ordered dot2 work shadows DMA.
            head_A0 = wave * fx.Index(HPW) + lo
            q_row0 = token * Hn * fx.Index(DQK) + head_A0 * fx.Index(DQK)
            do_row0 = token * Hn * fx.Index(D) + head_A0 * fx.Index(D)
            o_row0 = token * Hn * fx.Index(D) + head_A0 * fx.Index(D)
            q_packs0 = [
                buffer_ops.buffer_load(
                    q_rsrc,
                    q_row0 + fx.Index(ks * 32) + grp * fx.Index(8),
                    vec_width=8,
                    dtype=elem,
                )
                for ks in range_constexpr(KS)
            ]
            do_packs0 = [
                buffer_ops.buffer_load(
                    do_rsrc,
                    do_row0 + fx.Index(ks * 32) + grp * fx.Index(8),
                    vec_width=8,
                    dtype=elem,
                )
                for ks in range_constexpr(KS)
            ]
            lse_h0 = fx.Float32(
                buffer_ops.buffer_load(
                    lse_rsrc,
                    token * Hn + head_A0,
                    vec_width=1,
                    dtype=fx.Float32,
                )
            )
            lse_h0 = fx.Float32(arith.MaxNumFOp(_raw(lse_h0), _raw(fx.Float32(-3.0e38))).result)
            lse_l2_0 = fx.Float32(lse_h0 * c_log2e)
            o_packs0 = [
                buffer_ops.buffer_load(
                    o_rsrc,
                    o_row0 + fx.Index(ks * 32) + grp * fx.Index(8),
                    vec_width=8,
                    dtype=elem,
                )
                for ks in range_constexpr(8)
            ]

            lds_kv_base = buffer_ops.extract_base_index(
                lds_kv,
                address_space=3,
            )
            dma_size = fx.Int32(16)
            dma_zero = fx.Int32(0)
            dma_aux = fx.Int32(1)

            def _dma_row(step):
                tile = step // 4
                row4 = step % 4
                idx = fx.Int32(
                    rocdl.ds_bpermute(
                        fx.Int32.ir_type,
                        _raw(fx.Int32(step * 4)),
                        _raw(idx_lane),
                    )
                )
                valid = ArithValue(idx >= fx.Int32(0))
                src = fx.Index(valid.select(idx, fx.Int32(0)))
                rank = fx.Index(tile * TILE_K + row4) + wave * fx.Index(4)
                lds_addr = lds_kv_base + rank * fx.Index(D_LDS * 2)
                lds_lane0 = rocdl.readfirstlane(
                    fx.Int64.ir_type,
                    fx.Int64(lds_addr),
                )
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_lane0,
                    address_space=3,
                )
                voffset = fx.Int32((src * fx.Index(DQK) + lane * fx.Index(8)) * fx.Index(2))
                rocdl.raw_ptr_buffer_load_lds(
                    kv_rsrc,
                    lds_ptr,
                    dma_size,
                    voffset,
                    dma_zero,
                    dma_zero,
                    dma_aux,
                )

            # Thirty-two 16-byte lane-cooperative row DMAs per wave cover all
            # R=128 resident rows across the four waves.
            for step in range_constexpr(32):
                _dma_row(step)
            rocdl.sched_barrier(0)

            # Keep the first half of the accepted adjacent-pair dot2 chain between
            # the DMA burst and its publication drain.
            dpart0 = fx.Float32(0.0)
            for ks in range_constexpr(8):
                op = Vec(o_packs0[ks]).bitcast(fx.Int32)
                dop = Vec(do_packs0[ks]).bitcast(fx.Int32)
                for i in range_constexpr(4):
                    dpart0 = _dot2_bf16(op[i], dop[i], dpart0)
            rocdl.sched_barrier(0)

            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

            # Stream the remaining O packs only after LDS publication, preserving
            # the exact ks/i accumulation order of the accepted implementation.
            for ks in range_constexpr(8, KS):
                ov0 = buffer_ops.buffer_load(
                    o_rsrc,
                    o_row0 + fx.Index(ks * 32) + grp * fx.Index(8),
                    vec_width=8,
                    dtype=elem,
                )
                op = Vec(ov0).bitcast(fx.Int32)
                dop = Vec(do_packs0[ks]).bitcast(fx.Int32)
                for i in range_constexpr(4):
                    dpart0 = _dot2_bf16(op[i], dop[i], dpart0)
            part_i0 = _raw(Vec.from_elements([dpart0], fx.Float32).bitcast(fx.Int32)[0])
            delta_h0 = fx.Float32(0.0)
            for g in range_constexpr(4):
                idx = _raw(fx.Int32(lo + fx.Index(g * 16)) * fx.Int32(4))
                gi = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part_i0)
                gf = fx.Float32(
                    _raw(
                        Vec.from_elements(
                            [fx.Int32(gi)],
                            fx.Int32,
                        ).bitcast(
                            fx.Float32
                        )[0]
                    )
                )
                delta_h0 = fx.Float32(arith.AddFOp(_raw(delta_h0), _raw(gf)).result)
            buffer_ops.buffer_store(
                _raw(delta_h0),
                delta_rsrc,
                (token * Hn + head_A0) * fx.Index(4),
                mask=_raw(
                    arith.CmpIOp(
                        arith.CmpIPredicate.eq,
                        _raw(grp),
                        _raw(fx.Index(0)),
                    ).result
                ),
                offset_is_bytes=True,
            )
        else:
            # Each 256-thread workgroup covers one 16-row tile at a time: g_row
            # selects the row and g_within selects one 32-column stripe. All eight
            # tiles remain resident for both H64 head passes.
            for tile in range_constexpr(NUM_TILES):
                rank = fx.Index(tile * TILE_K) + g_row
                idx = fx.Int32(
                    buffer_ops.buffer_load(
                        tk_rsrc,
                        tk_row + rank,
                        vec_width=1,
                        dtype=fx.Int32,
                    )
                )
                valid = ArithValue(idx >= fx.Int32(0))
                src = fx.Index(valid.select(idx, fx.Int32(0)))
                row_off = rank * fx.Index(D_LDS)
                for c in range_constexpr(4):
                    kvv = buffer_ops.buffer_load(
                        kv_rsrc,
                        src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                        vec_width=8,
                        dtype=elem,
                    )
                    Vec(kvv).store(
                        lds_kv,
                        [row_off + g_within * fx.Index(32) + fx.Index(c * 8)],
                    )
                if g_within == fx.Index(0):
                    mask = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                    Vec.from_elements([mask], fx.Float32).store(
                        lds_mask,
                        [rank],
                    )

            # All lanes execute the same fixed staging loop. This single barrier
            # drains and publishes every resident row.
            gpu.barrier()

        _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

        def _mma_ag(a, b, c):
            op = _llvm.InlineAsmOp(
                res=_v4f32_ir,
                operands_=[_raw(a), _raw(b), _raw(c)],
                asm_string="v_mfma_f32_16x16x16_bf16 $0, $1, $2, $0",
                constraints="=a,v,v,0",
                has_side_effects=False,
            )
            return op.result

        # Compile-time sequential passes keep one 32xv4 FP32 (128-AGPR) dQ set
        # live per wave. Pass 0 stores heads 0:64 before pass 1 creates a fresh
        # accumulator set for heads 64:128.
        for head_group in range_constexpr(NUM_HEADS // BLOCK_H):
            if const_expr(prologue_g2l_overlap and head_group == 0):
                head_A = head_A0
                q_packs = q_packs0
                do_packs = do_packs0
                lse_l2 = lse_l2_0
                delta_h = delta_h0
            else:
                head_wave_base = fx.Index(head_group * BLOCK_H) + wave * fx.Index(HPW)
                head_A = head_wave_base + lo
                q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
                do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
                o_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)

                q_packs = [
                    buffer_ops.buffer_load(
                        q_rsrc,
                        q_row + fx.Index(ks * 32) + grp * fx.Index(8),
                        vec_width=8,
                        dtype=elem,
                    )
                    for ks in range_constexpr(KS)
                ]
                do_packs = [
                    buffer_ops.buffer_load(
                        do_rsrc,
                        do_row + fx.Index(ks * 32) + grp * fx.Index(8),
                        vec_width=8,
                        dtype=elem,
                    )
                    for ks in range_constexpr(KS)
                ]

                lse_h = fx.Float32(
                    buffer_ops.buffer_load(
                        lse_rsrc,
                        token * Hn + head_A,
                        vec_width=1,
                        dtype=fx.Float32,
                    )
                )
                lse_h = fx.Float32(
                    arith.MaxNumFOp(
                        _raw(lse_h),
                        _raw(fx.Float32(-3.0e38)),
                    ).result
                )
                lse_l2 = fx.Float32(lse_h * c_log2e)

                # Keep the accepted ordered adjacent-pair dot2 accumulation.
                dpart = fx.Float32(0.0)
                for ks in range_constexpr(KS):
                    ov = buffer_ops.buffer_load(
                        o_rsrc,
                        o_row + fx.Index(ks * 32) + grp * fx.Index(8),
                        vec_width=8,
                        dtype=elem,
                    )
                    op = Vec(ov).bitcast(fx.Int32)
                    dop = Vec(do_packs[ks]).bitcast(fx.Int32)
                    for i in range_constexpr(4):
                        dpart = _dot2_bf16(op[i], dop[i], dpart)
                part_i = _raw(Vec.from_elements([dpart], fx.Float32).bitcast(fx.Int32)[0])
                delta_h = fx.Float32(0.0)
                for g in range_constexpr(4):
                    idx = _raw(fx.Int32(lo + fx.Index(g * 16)) * fx.Int32(4))
                    gi = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part_i)
                    gf = fx.Float32(
                        _raw(
                            Vec.from_elements(
                                [fx.Int32(gi)],
                                fx.Int32,
                            ).bitcast(
                                fx.Float32
                            )[0]
                        )
                    )
                    delta_h = fx.Float32(arith.AddFOp(_raw(delta_h), _raw(gf)).result)
                buffer_ops.buffer_store(
                    _raw(delta_h),
                    delta_rsrc,
                    (token * Hn + head_A) * fx.Index(4),
                    mask=_raw(
                        arith.CmpIOp(
                            arith.CmpIPredicate.eq,
                            _raw(grp),
                            _raw(fx.Index(0)),
                        ).result
                    ),
                    offset_is_bytes=True,
                )

            dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DT)]
            loop_results = dq_acc0
            for t, iter_args in range(
                fx.Index(0),
                fx.Index(NUM_TILES),
                fx.Index(1),
                init=dq_acc0,
            ):
                dq_acc = [iter_args[dt] for dt in range_constexpr(DT)]
                buf_off = t * fx.Index(TILE_K * D_LDS)
                mbuf_off = t * fx.Index(TILE_K)
                mask4 = Vec.load(
                    v4f,
                    lds_mask,
                    [mbuf_off + grp * fx.Index(4)],
                )

                def _bv(ks, buf_off=buf_off):
                    return Vec.load(
                        v8,
                        lds_kv,
                        [buf_off + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)],
                    )

                # Exact accepted two-way QK accumulation, with PF4 resident-LDS reads.
                acc_s0 = Vec.filled(4, 0.0, fx.Float32)
                acc_s1 = Vec.filled(4, 0.0, fx.Float32)
                acc_dp0 = Vec.filled(4, 0.0, fx.Float32)
                acc_dp1 = Vec.filled(4, 0.0, fx.Float32)
                bvq = [_bv(k) for k in range_constexpr(qk_prefetch)]
                for ks in range_constexpr(KS):
                    if ks + qk_prefetch < KS:
                        bvq.append(_bv(ks + qk_prefetch))
                    if ks % 2 == 0:
                        acc_s0 = rocdl.mfma_f32_16x16x32_bf16(
                            v4f,
                            [_raw(bvq[ks]), q_packs[ks], acc_s0],
                        )
                        acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(
                            v4f,
                            [_raw(bvq[ks]), do_packs[ks], acc_dp0],
                        )
                    else:
                        acc_s1 = rocdl.mfma_f32_16x16x32_bf16(
                            v4f,
                            [_raw(bvq[ks]), q_packs[ks], acc_s1],
                        )
                        acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(
                            v4f,
                            [_raw(bvq[ks]), do_packs[ks], acc_dp1],
                        )
                acc_s = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i])) for i in range_constexpr(4)],
                    fx.Float32,
                )
                acc_dp = Vec.from_elements(
                    [fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i])) for i in range_constexpr(4)],
                    fx.Float32,
                )

                lo_d4 = lo // fx.Index(4)
                lo_m4 = lo % fx.Index(4)
                pv_base = fx.Int64(
                    (buf_off + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2)
                    + fx.Index(kv_off)
                )

                def _bvv(dt, pv_base=pv_base):
                    ptr = buffer_ops.create_llvm_ptr(
                        _raw(pv_base + fx.Int64(dt * 32)),
                        address_space=3,
                    )
                    return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

                bvv_pf = [_bvv(dt) for dt in range_constexpr(pv_prefetch)]

                pvals = [None] * 4
                dsvals = [None] * 4
                for i in range_constexpr(4):
                    arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                    p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                    pvals[i] = p
                    dp = fx.Float32(_raw(Vec(acc_dp)[i]))
                    dsvals[i] = p * (dp - delta_h) * c_scale

                pB = _raw(
                    Vec.from_elements(
                        [fx.BFloat16(_raw(dsvals[i])) for i in range_constexpr(4)],
                        elem,
                    ).bitcast(fx.Int16)
                )
                new_dq = [None] * DT
                for dt in range_constexpr(DT):
                    if dt + pv_prefetch < DT:
                        bvv_pf.append(_bvv(dt + pv_prefetch))
                    new_dq[dt] = _mma_ag(
                        bvv_pf[dt],
                        pB,
                        dq_acc[dt],
                    )

                ts = t * fx.Index(TILE_K)
                sp_base = ((token * Hn + head_A) * fx.Index(TOPK_LEN) + ts + grp * fx.Index(4)) * fx.Index(2)
                ds_pk0 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(dsvals[0]),
                            _raw(dsvals[1]),
                        )
                    )
                )
                ds_pk1 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(dsvals[2]),
                            _raw(dsvals[3]),
                        )
                    )
                )
                pp_pk0 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(pvals[0]),
                            _raw(pvals[1]),
                        )
                    )
                )
                pp_pk1 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(pvals[2]),
                            _raw(pvals[3]),
                        )
                    )
                )
                buffer_ops.buffer_store(
                    _raw(Vec.from_elements([ds_pk0, ds_pk1], fx.Int32)),
                    ds_rsrc,
                    sp_base,
                    offset_is_bytes=True,
                )
                buffer_ops.buffer_store(
                    _raw(Vec.from_elements([pp_pk0, pp_pk1], fx.Int32)),
                    pp_rsrc,
                    sp_base,
                    offset_is_bytes=True,
                )

                loop_results = yield list(new_dq)

            dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
            for dt in range_constexpr(DT):
                ov = Vec(dq_acc[dt])
                base = (
                    token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)
                ) * fx.Index(2)
                pk0 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(Vec(ov)[0]),
                            _raw(Vec(ov)[1]),
                        )
                    )
                )
                pk1 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(Vec(ov)[2]),
                            _raw(Vec(ov)[3]),
                        )
                    )
                )
                buffer_ops.buffer_store(
                    _raw(Vec.from_elements([pk0, pk1], fx.Int32)),
                    dq_rsrc,
                    base,
                    offset_is_bytes=True,
                )
            zero_v2 = Vec.from_elements(
                [fx.Int32(0), fx.Int32(0)],
                fx.Int32,
            )
            for rt in range_constexpr(4):
                rbase = (
                    token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)
                ) * fx.Index(2)
                buffer_ops.buffer_store(
                    _raw(zero_v2),
                    dq_rsrc,
                    rbase,
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(
            Q,
            KV,
            DO,
            TOPK,
            LSE,
            DELTA,
            DQ,
            DS,
            PP,
            O,
            T,
            H,
            NKV,
        ).launch(
            grid=(fx.Index(T), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return _attach(launch)


def build_bwd_dq_pv2x(
    topk_len,
    scale,
    num_heads=None,
    qk_prefetch=6,
    pv_prefetch=8,
    double_buf=1,
    resident_qdo_g2l=False,
    opaque_g2l_overlap=False,
    tail_mask=False,
):
    if bool(resident_qdo_g2l) and not (int(topk_len) == 1152 and int(num_heads) == 128 and bool(double_buf)):
        raise ValueError("resident_qdo_g2l requires exact Pro/cr4 R1152/H128 " "with four-bank double buffering")
    if bool(opaque_g2l_overlap) and not bool(resident_qdo_g2l):
        raise ValueError("opaque_g2l_overlap requires the exact Pro/cr4 resident-Q/dO " "direct-G2L transport")
    if bool(tail_mask) and not (int(topk_len) == 160 and int(num_heads) == 128 and not bool(resident_qdo_g2l)):
        raise ValueError("tail_mask requires exact Pro/cr128 R160/H128 pv2x")
    elem = fx.BFloat16
    D_LDS = 528
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    LDS_ELEMS = TILE_K * D_LDS
    # double_buf: cross-pair double-buffer (KVBUF=4). pair p writes LDS set (p%2); next pair writes the
    # OTHER set, so the post-PV WAR barrier is unnecessary (1 barrier/pair vs 2) -> -sync-latency.
    KVBUF = 4 if double_buf else 2
    allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=(
            (
                "mla_bwd_dq_pv2x_pro_cr4_opaque_g2l_overlap_smem"
                if opaque_g2l_overlap
                else "mla_bwd_dq_pv2x_pro_cr4_resident_qdo_g2l_smem"
            )
            if resident_qdo_g2l
            else ("mla_bwd_dq_pv2x_pro_cr128_tail_mask_smem" if tail_mask else "mla_bwd_dq_pv2x_smem")
        ),
    )
    kv_off = allocator._align(allocator.ptr, 16)
    mask_off = allocator._align(kv_off + KVBUF * LDS_ELEMS * 2, 16)
    allocator.ptr = allocator._align(mask_off + KVBUF * TILE_K * 4, 16)
    _QK4 = 1 if ((num_heads is not None and num_heads > 64) or topk_len >= 512) else 0

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        i16 = fx.Int16
        lds_kv = SmemPtr(allocator.get_base(), kv_off, elem.ir_type, shape=(KVBUF * LDS_ELEMS,)).get()
        lds_mask = SmemPtr(allocator.get_base(), mask_off, fx.Float32.ir_type, shape=(KVBUF * TILE_K,)).get()
        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)
        head_wave_base = hg * fx.Index(BLOCK_H) + wave * fx.Index(HPW)
        head_A = head_wave_base + lo
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2))
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2))
        )
        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        q_packs = [
            buffer_ops.buffer_load(q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for ks in range_constexpr(KS)
        ]
        do_packs = [
            buffer_ops.buffer_load(do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for ks in range_constexpr(KS)
        ]
        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)

        # Fold delta = rowsum(O*dO) into pv2x. dO is already register-resident; stream one
        # O vec8 at a time before dq_acc exists, so the temporary does not extend into the
        # register-heavy QK/PV loop. The resulting scalar replaces the old DELTA load and is
        # still written once per head for the later d_sink reduction.
        o_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        dpart = fx.Float32(0.0)
        for ks in range_constexpr(KS):
            ov = buffer_ops.buffer_load(o_rsrc, o_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for i in range_constexpr(8):
                oi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(ov)[i])).result)
                doi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(do_packs[ks])[i])).result)
                dpart = fx.Float32(arith.AddFOp(_raw(dpart), _raw(oi * doi)).result)
        part_i = _raw(Vec.from_elements([dpart], fx.Float32).bitcast(fx.Int32)[0])
        delta_h = fx.Float32(0.0)
        for g in range_constexpr(4):
            idx = _raw(fx.Int32(lo + fx.Index(g * 16)) * fx.Int32(4))
            gi = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part_i)
            gf = fx.Float32(_raw(Vec.from_elements([fx.Int32(gi)], fx.Int32).bitcast(fx.Float32)[0]))
            delta_h = fx.Float32(arith.AddFOp(_raw(delta_h), _raw(gf)).result)
        buffer_ops.buffer_store(
            _raw(delta_h),
            delta_rsrc,
            (token * Hn + head_A) * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(grp), _raw(fx.Index(0))).result),
            offset_is_bytes=True,
        )
        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tb):
            return fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + tb + g_row,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )

        def load_topk_masked(tb, enabled):
            return fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + tb + g_row,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=_raw(enabled),
                )
            )

        def gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(4)
            ]

        def gather_load_masked(idx, enabled):
            valid = ArithValue(
                arith.AndIOp(
                    _raw(ArithValue(idx >= fx.Int32(0))),
                    _raw(enabled),
                ).result
            )
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                    mask=_raw(enabled),
                )
                for c in range_constexpr(4)
            ]

        def gather_store(vvs, idx, bo, mo):
            valid = ArithValue(idx >= fx.Int32(0))
            for c in range_constexpr(4):
                Vec(vvs[c]).store(lds_kv, [bo + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)])
            if g_within == fx.Index(0):
                m = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(lds_mask, [mo + g_row])

        def _qkmma(a, b, c):
            return rocdl.mfma_f32_16x16x32_bf16(v4f, [a, b, c])

        def qk_softmax(buf_off, mbuf_off, t):
            # per-16 QK + softmax -> dsvals(4), pvals(4); store dS/P for this tile.
            mask4 = Vec.load(v4f, lds_mask, [mbuf_off + grp * fx.Index(4)])
            accs = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
            accd = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]

            def _bv(ks):
                return Vec.load(v8, lds_kv, [buf_off + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)])

            bvq = [_bv(k) for k in range_constexpr(qk_prefetch)]
            NR = 4 if _QK4 else 2
            for ks in range_constexpr(KS):
                if ks + qk_prefetch < KS:
                    bvq.append(_bv(ks + qk_prefetch))
                r = ks % NR
                accs[r] = _qkmma(_raw(bvq[ks]), q_packs[ks], accs[r])
                accd[r] = _qkmma(_raw(bvq[ks]), do_packs[ks], accd[r])
            acc_s = [sum([fx.Float32(_raw(Vec(accs[j])[i])) for j in range_constexpr(NR)]) for i in range_constexpr(4)]
            acc_dp = [sum([fx.Float32(_raw(Vec(accd[j])[i])) for j in range_constexpr(NR)]) for i in range_constexpr(4)]
            pvals = [None] * 4
            dsvals = [None] * 4
            for i in range_constexpr(4):
                arg = acc_s[i] * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                pvals[i] = p
                dsvals[i] = p * (acc_dp[i] - delta_h) * c_scale
            ts = t * fx.Index(TILE_K)
            sp = ((token * Hn + head_A) * fx.Index(topk_len) + ts + grp * fx.Index(4)) * fx.Index(2)
            ds2 = Vec.from_elements(
                [
                    fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[0]), _raw(dsvals[1])))),
                    fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(dsvals[2]), _raw(dsvals[3])))),
                ],
                fx.Int32,
            )
            pp2 = Vec.from_elements(
                [
                    fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[0]), _raw(pvals[1])))),
                    fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(pvals[2]), _raw(pvals[3])))),
                ],
                fx.Int32,
            )
            buffer_ops.buffer_store(_raw(ds2), ds_rsrc, sp, offset_is_bytes=True)
            buffer_ops.buffer_store(_raw(pp2), pp_rsrc, sp, offset_is_bytes=True)
            return dsvals

        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        def v32(ba, bb, dt):
            pa = buffer_ops.create_llvm_ptr(_raw(ba + fx.Int64(dt * 32)), address_space=3)
            pb = buffer_ops.create_llvm_ptr(_raw(bb + fx.Int64(dt * 32)), address_space=3)
            va = Vec(rocdl.ds_read_tr16_b64(v4, pa).result).bitcast(i16)
            vb = Vec(rocdl.ds_read_tr16_b64(v4, pb).result).bitcast(i16)
            return _raw(Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], i16).bitcast(elem))

        # inline-asm MFMA with D=C in AGPR ("=a,v,v,0"): pins the loop-carried dq_acc in the
        # separate AGPR file in-place (no accvgpr shuffle) -> frees ~128 arch-VGPR -> occ-2.
        _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

        def _mma_agpr(a, b, c):
            op = _llvm.InlineAsmOp(
                res=_v4f32_ir,
                operands_=[_raw(a), _raw(b), _raw(c)],
                asm_string="v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
                constraints="=a,v,v,0",
                has_side_effects=False,
            )
            return op.result

        def _mma_agpr_after_valu(a, b, c):
            # pB is a VALU definition immediately before the first PV MFMA.
            # Inline asm hides the dependency from LLVM, so keep gfx950's
            # required VALU-write -> MFMA-read delay in the same asm block.
            op = _llvm.InlineAsmOp(
                res=_v4f32_ir,
                operands_=[_raw(a), _raw(b), _raw(c)],
                asm_string=("s_nop 1\n" "s_nop 1\n" "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0"),
                constraints="=a,v,v,0",
                has_side_effects=False,
            )
            return op.result

        def _stage_pair_g2l(pair, pair_bank):
            # Eight owner lanes per wave hold that wave's four rows from each
            # tile. Across four waves this loads each pair's 32 TOPK entries
            # exactly once; ds_bpermute broadcasts an owner index to the 64
            # lanes that cooperatively load its 512-element KV row.
            owner_rank = (
                pair * fx.Index(2 * TILE_K)
                + (lane // fx.Index(4)) * fx.Index(TILE_K)
                + wave * fx.Index(4)
                + lane % fx.Index(4)
            )
            idx_lane = fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + owner_rank,
                    vec_width=1,
                    dtype=fx.Int32,
                    mask=_raw(ArithValue(lane) < fx.Index(8)),
                )
            )
            idx_lane_valid = ArithValue(idx_lane >= fx.Int32(0))
            # Publish each row mask from its unique owner before issuing any
            # direct-to-LDS load. Keeping this store outside _dma_row avoids a
            # compiler-inserted vmcnt(0) after every DMA.
            if lane < fx.Index(8):
                m = fx.Float32(idx_lane_valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([m], fx.Float32).store(
                    lds_mask,
                    [
                        pair_bank * fx.Index(2 * TILE_K)
                        + (lane // fx.Index(4)) * fx.Index(TILE_K)
                        + wave * fx.Index(4)
                        + lane % fx.Index(4)
                    ],
                )
            lds_kv_base = buffer_ops.extract_base_index(
                lds_kv,
                address_space=3,
            )
            buf_a = pair_bank * fx.Index(2 * LDS_ELEMS)
            buf_b = buf_a + fx.Index(LDS_ELEMS)
            dma_size = fx.Int32(16)
            dma_zero = fx.Int32(0)
            dma_aux = fx.Int32(1)

            def _dma_row(step):
                tile = step // 4
                row4 = step % 4
                idx = fx.Int32(
                    rocdl.ds_bpermute(
                        fx.Int32.ir_type,
                        _raw(fx.Int32(step * 4)),
                        _raw(idx_lane),
                    )
                )
                valid = ArithValue(idx >= fx.Int32(0))
                src = fx.Index(valid.select(idx, fx.Int32(0)))
                rank = wave * fx.Index(4) + fx.Index(row4)
                bank_off = buf_a if tile == 0 else buf_b
                lds_addr = lds_kv_base + (bank_off + rank * fx.Index(D_LDS)) * fx.Index(2)
                lds_lane0 = rocdl.readfirstlane(
                    fx.Int64.ir_type,
                    fx.Int64(lds_addr),
                )
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_lane0,
                    address_space=3,
                )
                voffset = fx.Int32((src * fx.Index(DQK) + lane * fx.Index(8)) * fx.Index(2))
                if const_expr(opaque_g2l_overlap):
                    # The typed direct-G2L intrinsic conservatively aliases
                    # every following LDS read, so LLVM inserts vmcnt(0)
                    # immediately after staging the next (inactive) bank.
                    # Encode the identical instruction opaquely: the explicit
                    # publish wait/barrier drains it at the next iteration,
                    # while current-bank QK/PV reads may remain in flight.
                    lds_element = fx.Int32(lds_lane0)
                    _llvm.inline_asm(
                        res=None,
                        operands_=[
                            _raw(voffset),
                            _raw(kv_rsrc),
                            _raw(lds_element),
                        ],
                        asm_string=("s_mov_b32 m0, $2\n" "s_nop 1\n" "buffer_load_dwordx4 $0, $1, 0 offen sc0 lds"),
                        constraints="v,s,s,~{memory}",
                        has_side_effects=True,
                    )
                else:
                    rocdl.raw_ptr_buffer_load_lds(
                        kv_rsrc,
                        lds_ptr,
                        dma_size,
                        voffset,
                        dma_zero,
                        dma_zero,
                        dma_aux,
                    )

            for step in range_constexpr(8):
                _dma_row(step)

        def _publish_g2l_pair():
            # Direct G2L completion is VMEM-tracked; mask stores are
            # LGKM-tracked. Drain both before the common workgroup barrier.
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=("s_waitcnt vmcnt(0) lgkmcnt(0)\n" "s_barrier"),
                constraints="",
                has_side_effects=True,
            )

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DT)]
        NPAIR = NUM_TILES // 2
        if const_expr(resident_qdo_g2l):
            # Software-pipeline prologue: issue pair zero, then publish it at
            # the loop's sole barrier site. Only dQ AGPRs cross the backedge.
            _stage_pair_g2l(fx.Index(0), fx.Index(0))
            init = list(dq_acc0)
        else:
            idx0 = load_topk(fx.Index(0))
            kva0 = gather_load(idx0)
            idx1 = load_topk(fx.Index(TILE_K))
            kvb0 = gather_load(idx1)
            idx2 = load_topk(fx.Index(2 * TILE_K))
            init = list(dq_acc0) + list(kva0) + list(kvb0) + [idx0, idx1, idx2]
        loop_results = init
        for p, iter_args in range(fx.Index(0), fx.Index(NPAIR), fx.Index(1), init=init):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DT)]
            ta = p * fx.Index(2)
            tb = ta + fx.Index(1)
            if const_expr(double_buf):
                # double_buf: pair p writes LDS set (p%2); next pair writes the OTHER set (no WAR barrier).
                pmod = p % fx.Index(2)
                ao = pmod * fx.Index(2 * LDS_ELEMS)
                bo = ao + fx.Index(LDS_ELEMS)
                amo = pmod * fx.Index(2 * TILE_K)
                bmo = amo + fx.Index(TILE_K)
            else:
                ao = fx.Index(0)
                bo = fx.Index(LDS_ELEMS)
                amo = fx.Index(0)
                bmo = fx.Index(TILE_K)
            if const_expr(resident_qdo_g2l):
                # Publish the current pair, then issue p+1 into the opposite
                # bank. Its VMEM/LDS traffic overlaps current QK/PV; the next
                # iteration's same barrier drains and publishes it.
                _publish_g2l_pair()
                if p + fx.Index(1) < fx.Index(NPAIR):
                    _stage_pair_g2l(
                        p + fx.Index(1),
                        (p + fx.Index(1)) % fx.Index(2),
                    )
            else:
                kva = [iter_args[DT + c] for c in range_constexpr(4)]
                kvb = [iter_args[DT + 4 + c] for c in range_constexpr(4)]
                idxa = iter_args[DT + 8]
                idxb = iter_args[DT + 9]
                idxn = iter_args[DT + 10]
                gather_store(kva, idxa, ao, amo)
                gather_store(kvb, idxb, bo, bmo)
                # Next-pair register prefetch. The short-R tail-masked variant
                # skips both unused KV rows after the final useful pair and
                # avoids carrying the unused look-ahead TOPK index.
                if const_expr(tail_mask):
                    has_next = ArithValue(p + fx.Index(1) < fx.Index(NPAIR))
                    has_after_next = ArithValue(p + fx.Index(2) < fx.Index(NPAIR))
                    kva_n = gather_load_masked(idxn, has_next)
                    idxn2 = load_topk_masked(
                        (ta + fx.Index(3)) * fx.Index(TILE_K),
                        has_next,
                    )
                    kvb_n = gather_load_masked(idxn2, has_next)
                    idxn3 = load_topk_masked(
                        (ta + fx.Index(4)) * fx.Index(TILE_K),
                        has_after_next,
                    )
                else:
                    kva_n = gather_load(idxn)
                    idxn2 = load_topk((ta + fx.Index(3)) * fx.Index(TILE_K))
                    kvb_n = gather_load(idxn2)
                    idxn3 = load_topk((ta + fx.Index(4)) * fx.Index(TILE_K))
                gpu.barrier()
            ds_a = qk_softmax(ao, amo, ta)  # per-16 QK/softmax tile a
            ds_b = qk_softmax(bo, bmo, tb)  # per-16 tile b
            # PV K=32: dq_acc += [dS_a | dS_b] @ [V_a | V_b]
            pB = _raw(
                Vec.from_elements(
                    [fx.BFloat16(_raw(ds_a[i])) for i in range_constexpr(4)]
                    + [fx.BFloat16(_raw(ds_b[i])) for i in range_constexpr(4)],
                    elem,
                )
            )
            base_a = fx.Int64(
                (ao + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2)
                + fx.Index(kv_off)
            )
            base_b = fx.Int64(
                (bo + (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)) * fx.Index(2)
                + fx.Index(kv_off)
            )
            new_dq = [None] * DT
            trq = [v32(base_a, base_b, dt) for dt in range_constexpr(pv_prefetch)]
            for dt in range_constexpr(DT):
                if dt + pv_prefetch < DT:
                    trq.append(v32(base_a, base_b, dt + pv_prefetch))
                if dt == 0:
                    new_dq[dt] = _mma_agpr_after_valu(trq[dt], pB, dq_acc[dt])
                else:
                    new_dq[dt] = _mma_agpr(trq[dt], pB, dq_acc[dt])
            if const_expr(not double_buf):
                # WAR barrier (single-buffer only): next pair reuses buf0/buf1 -> its gather_store
                # must wait for this pair's PV tr16 reads to drain (else cross-wave race). double_buf avoids
                # this by writing the OTHER set next pair.
                gpu.barrier()
            if const_expr(resident_qdo_g2l):
                yielded = list(new_dq)
            else:
                yielded = list(new_dq) + list(kva_n) + list(kvb_n) + [idxn, idxn2, idxn3]
            loop_results = yield yielded

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        head_i = head_wave_base + lo
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (
                token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (
                token * Hn * fx.Index(DQK) + head_i * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        gy = fx.Index(H) // fx.Index(BLOCK_H)
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, DS, PP, O, T, H, NKV).launch(
            grid=(fx.Index(T), gy, 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


# ============================================================================
# kernel: interm
# ============================================================================

THREADS = 256
WAVES = THREADS // 64  # 4
D_TILES = D_V // 16  # 32
DT_PER_WAVE = D_TILES // WAVES  # 8


def build_bwd_dq_twostage(
    topk_len,
    scale,
    num_heads=None,
    qk_prefetch=2,
    pv_prefetch=8,
):
    """Exact Pro/cr4 H128/WG512 producer plus D-major MFMA16 PV.

    This is materially different from the rejected H64/WG256 N07 split:
    one 512-thread workgroup owns all 128 heads, and its 32 sixteen-thread
    subgroups stage the two 16-row KV tiles exactly once per pair.  The score
    producer is byte-identical to P13. The consumer maps its eight waves to
    disjoint D64 slices, while each wave covers all 128 heads with 32
    independent 16x16x32 BF16 MFMA accumulator chains. dS is cooperatively
    staged once so each V operand is reused across eight H16 tiles.
    """
    if int(topk_len) != 1152 or int(num_heads) != 128:
        raise ValueError("twostage requires exact Pro/cr4 R1152/H128")
    elem = fx.BFloat16
    dq_block_h = 128
    dq_waves = dq_block_h // HPW
    dq_threads = dq_waves * 64
    gather_groups = dq_threads // 16
    assert (dq_block_h, dq_waves, dq_threads, gather_groups) == (
        128,
        8,
        512,
        32,
    )
    d_lds = 528
    num_tiles = topk_len // TILE_K
    num_pairs = num_tiles // 2
    lds_elems = TILE_K * d_lds
    kv_buffers = 4
    assert num_tiles == 72 and num_pairs == 36

    score_allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=("mla_bwd_dq_pro_cr4_h128_twostage_score_smem"),
    )
    score_kv_off = score_allocator._align(score_allocator.ptr, 16)
    score_mask_off = score_allocator._align(
        score_kv_off + kv_buffers * lds_elems * 2,
        16,
    )
    score_allocator.ptr = score_allocator._align(
        score_mask_off + kv_buffers * TILE_K * 4,
        16,
    )
    assert score_allocator.ptr == 67840

    pv_allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=("mla_bwd_dq_pro_cr4_h128_pv_dmajor_smem"),
    )
    pv_kv_off = pv_allocator._align(pv_allocator.ptr, 16)
    pv_ds_off = pv_allocator._align(
        pv_kv_off + kv_buffers * lds_elems * 2,
        16,
    )
    pv_allocator.ptr = pv_allocator._align(
        pv_ds_off + 2 * 2 * dq_block_h * TILE_K * 2,
        16,
    )
    assert pv_allocator.ptr == 83968

    @flyc.kernel(known_block_size=[dq_threads, 1, 1])
    def score_k(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_kv = SmemPtr(
            score_allocator.get_base(),
            score_kv_off,
            elem.ir_type,
            shape=(kv_buffers * lds_elems,),
        ).get()
        lds_mask = SmemPtr(
            score_allocator.get_base(),
            score_mask_off,
            fx.Float32.ir_type,
            shape=(kv_buffers * TILE_K,),
        ).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)
        head_A = hg * fx.Index(dq_block_h) + wave * fx.Index(HPW) + lo

        q_rsrc = buffer_ops.create_buffer_resource(
            Q,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)),
        )
        kv_rsrc = buffer_ops.create_buffer_resource(
            KV,
            max_size=False,
            num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)),
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)),
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2)),
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4)),
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)),
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4)),
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)),
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)),
        )

        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        q_packs = [
            buffer_ops.buffer_load(
                q_rsrc,
                q_row + fx.Index(ks * 32) + grp * fx.Index(8),
                vec_width=8,
                dtype=elem,
            )
            for ks in range_constexpr(KS)
        ]
        do_packs = [
            buffer_ops.buffer_load(
                do_rsrc,
                do_row + fx.Index(ks * 32) + grp * fx.Index(8),
                vec_width=8,
                dtype=elem,
            )
            for ks in range_constexpr(KS)
        ]
        lse_h = fx.Float32(
            buffer_ops.buffer_load(
                lse_rsrc,
                token * Hn + head_A,
                vec_width=1,
                dtype=fx.Float32,
            )
        )
        lse_h = fx.Float32(
            arith.MaxNumFOp(
                _raw(lse_h),
                _raw(fx.Float32(-3.0e38)),
            ).result
        )
        lse_l2 = fx.Float32(lse_h * c_log2e)

        # Preserve the accepted scalar O*dO accumulation and reduction order.
        o_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        dpart = fx.Float32(0.0)
        for ks in range_constexpr(KS):
            ov = buffer_ops.buffer_load(
                o_rsrc,
                o_row + fx.Index(ks * 32) + grp * fx.Index(8),
                vec_width=8,
                dtype=elem,
            )
            for i in range_constexpr(8):
                oi = fx.Float32(
                    arith.ExtFOp(
                        fx.Float32.ir_type,
                        _raw(Vec(ov)[i]),
                    ).result
                )
                doi = fx.Float32(
                    arith.ExtFOp(
                        fx.Float32.ir_type,
                        _raw(Vec(do_packs[ks])[i]),
                    ).result
                )
                dpart = fx.Float32(
                    arith.AddFOp(
                        _raw(dpart),
                        _raw(oi * doi),
                    ).result
                )
        part_i = _raw(Vec.from_elements([dpart], fx.Float32).bitcast(fx.Int32)[0])
        delta_h = fx.Float32(0.0)
        for g in range_constexpr(4):
            idx = _raw(fx.Int32(lo + fx.Index(g * 16)) * fx.Int32(4))
            gi = rocdl.ds_bpermute(fx.Int32.ir_type, idx, part_i)
            gf = fx.Float32(
                _raw(
                    Vec.from_elements(
                        [fx.Int32(gi)],
                        fx.Int32,
                    ).bitcast(
                        fx.Float32
                    )[0]
                )
            )
            delta_h = fx.Float32(
                arith.AddFOp(
                    _raw(delta_h),
                    _raw(gf),
                ).result
            )
        buffer_ops.buffer_store(
            _raw(delta_h),
            delta_rsrc,
            (token * Hn + head_A) * fx.Index(4),
            mask=_raw(
                arith.CmpIOp(
                    arith.CmpIPredicate.eq,
                    _raw(grp),
                    _raw(fx.Index(0)),
                ).result
            ),
            offset_is_bytes=True,
        )

        # The 32 sixteen-thread subgroups cover both 16-row tiles in a pair:
        # subgroup 0:15 owns tile A, 16:31 owns tile B.  Thus every selected
        # KV row is loaded and published once by the whole H128 workgroup.
        g_group = tid // fx.Index(16)
        g_tile = g_group // fx.Index(TILE_K)
        g_row = g_group % fx.Index(TILE_K)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def _load_topk_pair(pair):
            return fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + (pair * fx.Index(2) + g_tile) * fx.Index(TILE_K) + g_row,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )

        def _gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(4)
            ]

        def _gather_store(vvs, idx, pair_bank):
            valid = ArithValue(idx >= fx.Int32(0))
            bo = pair_bank * fx.Index(2 * lds_elems) + g_tile * fx.Index(lds_elems)
            mo = pair_bank * fx.Index(2 * TILE_K) + g_tile * fx.Index(TILE_K)
            for c in range_constexpr(4):
                Vec(vvs[c]).store(
                    lds_kv,
                    [bo + g_row * fx.Index(d_lds) + g_within * fx.Index(32) + fx.Index(c * 8)],
                )
            if g_within == fx.Index(0):
                mask = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
                Vec.from_elements([mask], fx.Float32).store(
                    lds_mask,
                    [mo + g_row],
                )

        def _score_tile(buf_off, mask_off, tile):
            mask4 = Vec.load(
                v4f,
                lds_mask,
                [mask_off + grp * fx.Index(4)],
            )
            acc_s = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
            acc_dp = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]

            def _kv_b(ks):
                return Vec.load(
                    v8,
                    lds_kv,
                    [buf_off + lo * fx.Index(d_lds) + fx.Index(ks * 32) + grp * fx.Index(8)],
                )

            bvq = [_kv_b(k) for k in range_constexpr(qk_prefetch)]
            for ks in range_constexpr(KS):
                if ks + qk_prefetch < KS:
                    bvq.append(_kv_b(ks + qk_prefetch))
                r = ks % 4
                acc_s[r] = rocdl.mfma_f32_16x16x32_bf16(
                    v4f,
                    [_raw(bvq[ks]), q_packs[ks], acc_s[r]],
                )
                acc_dp[r] = rocdl.mfma_f32_16x16x32_bf16(
                    v4f,
                    [_raw(bvq[ks]), do_packs[ks], acc_dp[r]],
                )
            score4 = [sum([fx.Float32(_raw(Vec(acc_s[j])[i])) for j in range_constexpr(4)]) for i in range_constexpr(4)]
            dp4 = [sum([fx.Float32(_raw(Vec(acc_dp[j])[i])) for j in range_constexpr(4)]) for i in range_constexpr(4)]
            pvals = [None] * 4
            dsvals = [None] * 4
            for i in range_constexpr(4):
                arg = score4[i] * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                pvals[i] = p
                dsvals[i] = p * (dp4[i] - delta_h) * c_scale
            sp = ((token * Hn + head_A) * fx.Index(topk_len) + tile * fx.Index(TILE_K) + grp * fx.Index(4)) * fx.Index(
                2
            )
            ds_pk0 = fx.Int32(
                _raw(
                    rocdl.cvt_pk_bf16_f32(
                        _raw(dsvals[0]),
                        _raw(dsvals[1]),
                    )
                )
            )
            ds_pk1 = fx.Int32(
                _raw(
                    rocdl.cvt_pk_bf16_f32(
                        _raw(dsvals[2]),
                        _raw(dsvals[3]),
                    )
                )
            )
            pp_pk0 = fx.Int32(
                _raw(
                    rocdl.cvt_pk_bf16_f32(
                        _raw(pvals[0]),
                        _raw(pvals[1]),
                    )
                )
            )
            pp_pk1 = fx.Int32(
                _raw(
                    rocdl.cvt_pk_bf16_f32(
                        _raw(pvals[2]),
                        _raw(pvals[3]),
                    )
                )
            )
            buffer_ops.buffer_store(
                _raw(
                    Vec.from_elements(
                        [ds_pk0, ds_pk1],
                        fx.Int32,
                    )
                ),
                ds_rsrc,
                sp,
                offset_is_bytes=True,
            )
            buffer_ops.buffer_store(
                _raw(
                    Vec.from_elements(
                        [pp_pk0, pp_pk1],
                        fx.Int32,
                    )
                ),
                pp_rsrc,
                sp,
                offset_is_bytes=True,
            )

        idx0 = _load_topk_pair(fx.Index(0))
        kv0 = _gather_load(idx0)
        idx1 = _load_topk_pair(fx.Index(1))
        init = list(kv0) + [idx0, idx1]
        loop_results = init
        for pair, iter_args in range(
            fx.Index(0),
            fx.Index(num_pairs),
            fx.Index(1),
            init=init,
        ):
            kv = [iter_args[c] for c in range_constexpr(4)]
            idx = iter_args[4]
            idxn = iter_args[5]
            tile_a = pair * fx.Index(2)
            tile_b = tile_a + fx.Index(1)
            pair_bank = pair % fx.Index(2)
            buf_a = pair_bank * fx.Index(2 * lds_elems)
            buf_b = buf_a + fx.Index(lds_elems)
            mask_a = pair_bank * fx.Index(2 * TILE_K)
            mask_b = mask_a + fx.Index(TILE_K)

            _gather_store(kv, idx, pair_bank)
            kv_n = _gather_load(idxn)
            idxn2 = _load_topk_pair(pair + fx.Index(2))
            gpu.barrier()
            _score_tile(buf_a, mask_a, tile_a)
            _score_tile(buf_b, mask_b, tile_b)
            loop_results = yield (list(kv_n) + [idxn, idxn2])

    @flyc.kernel(known_block_size=[dq_threads, 1, 1])
    def pv_k_dmajor(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        """D-major H128 PV: one D64 slice/wave and all heads/wave.

        The eight waves own disjoint D64 slices. Each wave retains P13's
        MFMA16 geometry and 32 independent accumulator chains: eight H16
        tiles times four D16 tiles. V is transposed once per D tile and
        reused across all head tiles. The workgroup stages each pair's
        complete H128xK32 dS panel once in MFMA-fragment layout; one b128
        read per head tile is reused across its four D tiles.
        """
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        i16 = fx.Int16
        lds_kv = SmemPtr(
            pv_allocator.get_base(),
            pv_kv_off,
            elem.ir_type,
            shape=(kv_buffers * lds_elems,),
        ).get()
        pair_ds_elems = dq_block_h * 2 * TILE_K
        lds_ds = SmemPtr(
            pv_allocator.get_base(),
            pv_ds_off,
            elem.ir_type,
            shape=(2 * pair_ds_elems,),
        ).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)
        token = fx.Index(gpu.block_idx.x)
        hg = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(H)

        kv_rsrc = buffer_ops.create_buffer_resource(
            KV,
            max_size=False,
            num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2)),
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4)),
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2)),
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS,
            max_size=False,
            num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(topk_len * 2)),
        )

        # P13's exact 32-subgroup selected-KV publication.
        g_group = tid // fx.Index(16)
        g_tile = g_group // fx.Index(TILE_K)
        g_row = g_group % fx.Index(TILE_K)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def _load_topk_pair(pair):
            return fx.Int32(
                buffer_ops.buffer_load(
                    tk_rsrc,
                    tk_row + (pair * fx.Index(2) + g_tile) * fx.Index(TILE_K) + g_row,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            )

        def _gather_load(idx):
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            return [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(4)
            ]

        def _gather_store(vvs, pair_bank):
            bo = pair_bank * fx.Index(2 * lds_elems) + g_tile * fx.Index(lds_elems)
            for c in range_constexpr(4):
                Vec(vvs[c]).store(
                    lds_kv,
                    [bo + g_row * fx.Index(d_lds) + g_within * fx.Index(32) + fx.Index(c * 8)],
                )

        # Thread 64*w+lane owns one K4 quarter for head 16*w+lane//4;
        # all 512 threads cover H128xK32 exactly once.
        ds_head = tid // fx.Index(4)
        ds_grp = tid % fx.Index(4)
        ds_k4 = ds_grp * fx.Index(4)
        ds_head_A = hg * fx.Index(dq_block_h) + ds_head

        def _load_ds_pair(pair):
            tile_a = pair * fx.Index(2)
            tile_b = tile_a + fx.Index(1)
            base = (token * Hn + ds_head_A) * fx.Index(topk_len)
            ds_a = buffer_ops.buffer_load(
                ds_rsrc,
                base + tile_a * fx.Index(TILE_K) + ds_k4,
                vec_width=4,
                dtype=elem,
            )
            ds_b = buffer_ops.buffer_load(
                ds_rsrc,
                base + tile_b * fx.Index(TILE_K) + ds_k4,
                vec_width=4,
                dtype=elem,
            )
            return ds_a, ds_b

        def _store_ds_pair(ds_a, ds_b, pair_bank):
            # Consumer-lane-major: both K4 tiles become one b128 read.
            ds_lane = ds_grp * fx.Index(16) + (ds_head % fx.Index(16))
            ds_ht = ds_head // fx.Index(16)
            values = Vec.from_elements(
                [Vec(ds_a)[i] for i in range_constexpr(4)] + [Vec(ds_b)[i] for i in range_constexpr(4)],
                elem,
            )
            Vec(values).store(
                lds_ds,
                [pair_bank * fx.Index(pair_ds_elems) + ds_ht * fx.Index(64 * 8) + ds_lane * fx.Index(8)],
            )

        def _load_ds_operand(pair_bank, ht):
            return Vec.load(
                v8,
                lds_ds,
                [pair_bank * fx.Index(pair_ds_elems) + fx.Index(ht * 64 * 8) + lane * fx.Index(8)],
            )

        # P13's exact MFMA16 K32 map. Two tr16 reads supply consecutive
        # K16 tiles and the v8 is shared across all eight H16 tiles.
        def _paired_v32(base_a, base_b, dt):
            global_dt = wave * fx.Index(4) + fx.Index(dt)
            pa = buffer_ops.create_llvm_ptr(
                _raw(base_a + fx.Int64(global_dt * fx.Index(32))),
                address_space=3,
            )
            pb = buffer_ops.create_llvm_ptr(
                _raw(base_b + fx.Int64(global_dt * fx.Index(32))),
                address_space=3,
            )
            va = Vec(rocdl.ds_read_tr16_b64(v4, pa).result).bitcast(i16)
            vb = Vec(rocdl.ds_read_tr16_b64(v4, pb).result).bitcast(i16)
            return _raw(
                Vec.from_elements(
                    [
                        va[0],
                        va[1],
                        va[2],
                        va[3],
                        vb[0],
                        vb[1],
                        vb[2],
                        vb[3],
                    ],
                    i16,
                ).bitcast(elem)
            )

        v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

        def _mma_agpr(a, b, c, v4f32_ir=v4f32_ir):
            op = _llvm.InlineAsmOp(
                res=v4f32_ir,
                operands_=[_raw(a), _raw(b), _raw(c)],
                asm_string=("v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0"),
                constraints="=a,v,v,0",
                has_side_effects=False,
            )
            return op.result

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8 * 4)]
        idx0 = _load_topk_pair(fx.Index(0))
        kv0 = _gather_load(idx0)
        idx1 = _load_topk_pair(fx.Index(1))
        ds_a0, ds_b0 = _load_ds_pair(fx.Index(0))
        init = list(dq_acc0) + list(kv0) + [idx0, idx1, ds_a0, ds_b0]
        loop_results = init
        for pair, iter_args in range(
            fx.Index(0),
            fx.Index(num_pairs),
            fx.Index(1),
            init=init,
        ):
            dq_acc = [iter_args[i] for i in range_constexpr(8 * 4)]
            kv = [iter_args[8 * 4 + c] for c in range_constexpr(4)]
            idx = iter_args[8 * 4 + 4]
            idxn = iter_args[8 * 4 + 5]
            ds_a = iter_args[8 * 4 + 6]
            ds_b = iter_args[8 * 4 + 7]
            pair_bank = pair % fx.Index(2)
            buf_a = pair_bank * fx.Index(2 * lds_elems)
            buf_b = buf_a + fx.Index(lds_elems)

            _gather_store(kv, pair_bank)
            _store_ds_pair(ds_a, ds_b, pair_bank)
            kv_n = _gather_load(idxn)
            idxn2 = _load_topk_pair(pair + fx.Index(2))
            ds_a_n, ds_b_n = _load_ds_pair(pair + fx.Index(1))
            gpu.barrier()

            pv_row = (grp * fx.Index(4) + lo_d4) * fx.Index(d_lds) + lo_m4 * fx.Index(4)
            base_a = fx.Int64((buf_a + pv_row) * fx.Index(2) + fx.Index(pv_kv_off))
            base_b = fx.Int64((buf_b + pv_row) * fx.Index(2) + fx.Index(pv_kv_off))
            v_ops = [_paired_v32(base_a, base_b, dt) for dt in range_constexpr(4)]
            new_dq = list(dq_acc)
            for ht in range_constexpr(8):
                pB = _load_ds_operand(pair_bank, ht)
                for dt in range_constexpr(4):
                    acc_i = ht * 4 + dt
                    new_dq[acc_i] = _mma_agpr(
                        v_ops[dt],
                        pB,
                        new_dq[acc_i],
                    )
            loop_results = yield (
                list(new_dq)
                + list(kv_n)
                + [
                    idxn,
                    idxn2,
                    ds_a_n,
                    ds_b_n,
                ]
            )

        dq_acc = [loop_results[i] for i in range_constexpr(8 * 4)]
        for ht in range_constexpr(8):
            head_A = hg * fx.Index(dq_block_h) + fx.Index(ht * 16) + lo
            for dt in range_constexpr(4):
                ov = Vec(dq_acc[ht * 4 + dt])
                out_d = wave * fx.Index(64) + fx.Index(dt * 16) + grp * fx.Index(4)
                out_base = (token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + out_d) * fx.Index(2)
                pk0 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(Vec(ov)[0]),
                            _raw(Vec(ov)[1]),
                        )
                    )
                )
                pk1 = fx.Int32(
                    _raw(
                        rocdl.cvt_pk_bf16_f32(
                            _raw(Vec(ov)[2]),
                            _raw(Vec(ov)[3]),
                        )
                    )
                )
                buffer_ops.buffer_store(
                    _raw(
                        Vec.from_elements(
                            [pk0, pk1],
                            fx.Int32,
                        )
                    ),
                    dq_rsrc,
                    out_base,
                    offset_is_bytes=True,
                )

            # Only wave 0 owns the 64-column RoPE tail. Its four lane
            # groups own disjoint K16 segments per head; two b128 stores
            # per group cover the segment exactly once (128 B/head).
            zero_v4 = Vec.from_elements(
                [
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                ],
                fx.Int32,
            )
            for rope_half in range_constexpr(2):
                rope_base = (
                    token * Hn * fx.Index(DQK)
                    + head_A * fx.Index(DQK)
                    + fx.Index(D)
                    + grp * fx.Index(16)
                    + fx.Index(rope_half * 8)
                ) * fx.Index(2)
                buffer_ops.buffer_store(
                    _raw(zero_v4),
                    dq_rsrc,
                    rope_base,
                    mask=_raw(ArithValue(wave) < fx.Index(1)),
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream,
    ):
        score_allocator.finalized = False
        pv_allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            score_allocator.finalize()
            pv_allocator.finalize()
        gy = fx.Index(H) // fx.Index(dq_block_h)
        score_k(
            Q,
            KV,
            DO,
            TOPK,
            LSE,
            DELTA,
            DQ,
            DS,
            PP,
            O,
            T,
            H,
            NKV,
            value_attrs=make_value_attrs(
                2,
                0,
                "512,512",
            ),
        ).launch(
            grid=(fx.Index(T), gy, 1),
            block=(dq_threads, 1, 1),
            stream=stream,
        )
        pv_k_dmajor(
            Q,
            KV,
            DO,
            TOPK,
            LSE,
            DELTA,
            DQ,
            DS,
            PP,
            O,
            T,
            H,
            NKV,
            value_attrs=make_value_attrs(
                2,
                128,
                "512,512",
            ),
        ).launch(
            grid=(fx.Index(T), gy, 1),
            block=(dq_threads, 1, 1),
            stream=stream,
        )

    return _attach(launch)


# build_interm_regtr: REGISTER-transpose Q/dO (no Q/dO LDS) -> BD=512 at occ-2. Each WAVE owns
# DT_PER_WAVE=8 contiguous d-tiles and loops all rank-tiles, holding only its 128 d of Q/dO
# compact. A[m=d,k=h] is built once via ds_bpermute 16x16 transpose; dS/P stay LDS-staged.
def build_interm_regtr(
    topk_len,
    num_heads,
    BD=512,
    stacked_k=False,
):
    if bool(stacked_k) and not (int(topk_len) == 1152 and int(num_heads) == 128 and int(BD) == 256):
        raise ValueError("stacked_k requires exact Pro/cr4 R1152/H128/BD256")
    elem = fx.BFloat16
    R_CHUNK = topk_len
    KS16 = num_heads // 16  # h-blocks of 16 (flash 4, pro 8)
    pro_cr0_gsz64_dbuf2 = int(topk_len) == 128 and int(num_heads) == 128
    # GSZ: rank-group size. Larger -> fewer RAW barriers / rank-group prologues. Must divide
    # R_CHUNK exactly (else the tail ranks drop). R%128==0 defaults to GSZ128
    # (RGROUPS halved; DBUF=1 keeps occ-2); un-divisible topk (e.g. cr128 R160) falls to 32.
    # Exact Pro cr0 uses two GSZ64 groups so group 1 can prefetch during group 0's MFMA.
    GSZ = 64 if pro_cr0_gsz64_dbuf2 else (128 if topk_len % 128 == 0 else (64 if topk_len % 64 == 0 else 32))
    RGROUPS = R_CHUNK // GSZ  # rank-groups (GSZ//16 rank-tiles each)
    RT_PER_G = GSZ // 16
    DBLK = D_V // BD  # d-blocks (grid.y); flash BD512->1, pro BD256->2
    DTB = BD // 16  # d-tiles in a block
    DTW = DTB // WAVES  # d-tiles per wave (a_q/a_do register footprint driver)
    SROW = GSZ + 8  # dS/P LDS row pad (tr16 bank-conflict avoid)
    DS_LDS = num_heads * SROW
    SDATA8 = (num_heads * GSZ) // 8
    SIT = SDATA8 // THREADS
    # DBUF: double-buffer dS/P LDS. DBUF=1 halves the dS/P LDS -> lifts the LDS-capped
    # occupancy -> hides the mfma/read latency. Short R (R128) prefers DBUF2.
    # The default GSZ128 path doubles DS_LDS -> DBUF=1 to keep occ-2 (2WG within LDS cap).
    DBUF = 2 if pro_cr0_gsz64_dbuf2 else (1 if (topk_len >= 192 or GSZ >= 128) else 2)
    allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=("mla_bwd_interm_regtr_pro_cr4_stacked_k_smem" if stacked_k else "mla_bwd_interm_regtr_smem"),
    )
    ds_off = allocator._align(allocator.ptr, 16)
    p_off = allocator._align(ds_off + DBUF * DS_LDS * 2, 16)
    allocator.ptr = allocator._align(p_off + DBUF * DS_LDS * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor, INTERM: fx.Tensor, T: fx.Int32):
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        i16 = fx.Int16
        lds_ds = SmemPtr(allocator.get_base(), ds_off, elem.ir_type, shape=(DS_LDS,)).get()
        lds_p = SmemPtr(allocator.get_base(), p_off, elem.ir_type, shape=(DS_LDS,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        dblk = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(num_heads)
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D_V * 2))
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2))
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2))
        )
        interm_rsrc = make_bf16_rebased_rsrc(
            INTERM, token * fx.Index(R_CHUNK) * fx.Index(D_V), fx.Index(R_CHUNK) * fx.Index(D_V * 2)
        )

        q_tok = token * Hn * fx.Index(DQK)
        do_tok = token * Hn * fx.Index(D_V)
        ds_tok = token * Hn * fx.Index(R_CHUNK)
        wbase_d = dblk * fx.Index(BD) + wave * fx.Index(DTW * 16)  # this wave's d-block start

        # register-transpose compact -> A[m=d,k=h] (_tr4 = one 16-h-block -> 4 i16). VPERM:
        # extract bf16 elem lo_m4 from the 2 bpermuted dwords via ONE v_perm_b32 byte-permute
        # (vs shift+select). Gated to short-topk (VALU-bound); neutral on large R.
        _VPERM = 1 if topk_len <= 192 else 0

        def _tr4(cv):
            vi = Vec(Vec(cv).bitcast(fx.Int32))  # 2 x i32 (dw0,dw1)
            dw0 = _raw(vi[0])
            dw1 = _raw(vi[1])
            sel = _raw(fx.Int32(lo_m4 * fx.Index(514) + fx.Index(256)))  # v_perm byte selector for elem lo_m4
            outs = []
            for i in range_constexpr(4):
                si = fx.Index(16) * grp + fx.Index(4 * i) + lo_d4
                idx = _raw(fx.Int32(si) * fx.Int32(4))
                s0 = rocdl.ds_bpermute(fx.Int32.ir_type, idx, dw0)
                s1 = rocdl.ds_bpermute(fx.Int32.ir_type, idx, dw1)
                if const_expr(_VPERM):
                    perm = _raw(rocdl.perm_b32(s1, s0, sel))
                    outs.append(fx.Int16(arith.TruncIOp(i16.ir_type, perm).result))
                else:
                    dwsel = ArithValue(lo_m4 < fx.Index(2)).select(s0, s1)
                    shift = ArithValue(lo_m4 % fx.Index(2) == fx.Index(0)).select(_raw(fx.Int32(0)), _raw(fx.Int32(16)))
                    shifted = _raw(fx.Int32(_raw(dwsel)) >> fx.Int32(_raw(shift)))
                    outs.append(fx.Int16(arith.TruncIOp(i16.ir_type, shifted).result))
            return outs  # list of 4 i16

        def transpose(cv):
            return _raw(Vec.from_elements(_tr4(cv), i16))  # v4 (K=16 A operand)

        # v8 DIRECT (2 h-blocks -> 8 i16, no v4 intermediate) for K=32 MFMA: building v8 straight
        # from 8 scalars avoids holding both v4 and v8 -> fits occ-2.
        def transpose8(cv0, cv1):
            return _raw(Vec.from_elements(_tr4(cv0) + _tr4(cv1), i16).bitcast(elem))  # v8 bf16 (K=32 A operand)

        def load_tile(rsrc, tok, dstride, u, ks):
            h = fx.Index(ks * 16) + lane // fx.Index(4)
            d = wbase_d + fx.Index(u * 16) + (lane % fx.Index(4)) * fx.Index(4)
            return buffer_ops.buffer_load(rsrc, tok + h * dstride + d, vec_width=4, dtype=elem)

        def tr_ks(off, coltile, ks):
            row = (
                (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * fx.Index(SROW)
                + coltile * fx.Index(16)
                + lo_m4 * fx.Index(4)
            )
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(i16))

        # ASM: issue all KS16 ds_read_b64_tr_b16 of a rank-tile as ONE inline-asm block so the
        # backend does not drain lgkmcnt per read and their LDS latencies overlap (caller drains
        # once before the mfma). Gated to R>=192; returns KS16 v2i32 (2 i32 = 4 bf16).
        _ASM = 1 if topk_len >= 192 else 0
        # ILP: split acc chains -> 4-way mfma ILP hides the mfma RAW latency. Gated to
        # 192<=R<=640 (elsewhere the extra acc VGPR drops occ where the chain isn't the bound).
        _ILP = 1 if 192 <= topk_len <= 640 else 0
        # s_setprio(1) around mfma: prioritizes mfma issue.
        _PRIO = 1
        # BSTACK (stacked-K): accumulate both GEMMs (Q@dS + dO@P, same [d,r]) into ONE acc chain
        # -> removes the 4 fp32 combine adds per output (interm is issue-bound). Gated to
        # topk<192; ILP owns 192<=R<=640 (mutually exclusive, _ILP takes precedence).
        _BSTACK = 1 if (topk_len < 192 or stacked_k) else 0
        v2i32_ty = ir.VectorType.get([2], ir.IntegerType.get_signless(32))

        def _tr16_packed(off, coltile):
            base_row = (grp * fx.Index(4) + lo_d4) * fx.Index(SROW) + fx.Index(coltile * 16) + lo_m4 * fx.Index(4)
            bptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(base_row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            N = KS16
            struct_t = _llvm.StructType.get_literal([v2i32_ty] * N)
            lines = [f"ds_read_b64_tr_b16 ${k}, ${N} offset:{k * 16 * SROW * 2}" for k in range(N)]
            constraints = ",".join(["=&v"] * N + ["v"] + ["~{memory}"])
            op = _llvm.InlineAsmOp(
                res=struct_t,
                operands_=[_raw(bptr)],
                asm_string="\n".join(lines),
                constraints=constraints,
                has_side_effects=True,
            )
            return [_llvm.extractvalue(v2i32_ty, op.result, [k]) for k in range(N)]

        def _wait_lgkm(n):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt lgkmcnt({n})",
                constraints="",
                has_side_effects=True,
            )

        # dS/P HBM->register prefetch (depth 1): a barrier separates the LDS store from compute,
        # so without prefetch each rank-group fully exposes the dS/P HBM load. Load group g+1
        # into registers during g's compute.
        def hbm_load(g):
            regs = []
            for it in range_constexpr(SIT):
                v = tid + fx.Index(it * THREADS)
                flat = v * fx.Index(8)
                h = flat // fx.Index(GSZ)
                r = flat % fx.Index(GSZ)
                dsv = buffer_ops.buffer_load(
                    ds_rsrc,
                    ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * GSZ) + r,
                    vec_width=8,
                    dtype=elem,
                )
                ppv = buffer_ops.buffer_load(
                    pp_rsrc,
                    ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * GSZ) + r,
                    vec_width=8,
                    dtype=elem,
                )
                regs.append((dsv, ppv, h, r))
            return regs

        # Issue all HBM loads up front (Q/dO compact for the transpose + dS/P group 0) so the
        # bpermute transpose burst (no MFMA to hide it) overlaps the HBM latency.
        q_cmp = [
            [load_tile(q_rsrc, q_tok, fx.Index(DQK), u, ks) for ks in range_constexpr(KS16)]
            for u in range_constexpr(DTW)
        ]
        do_cmp = [
            [load_tile(do_rsrc, do_tok, fx.Index(D_V), u, ks) for ks in range_constexpr(KS16)]
            for u in range_constexpr(DTW)
        ]
        loads = [None] * RGROUPS
        loads[0] = hbm_load(0)
        # K=32 A operands (v8, register-resident, reused across all rank-tiles). transpose8
        # builds v8 directly -> only 128 VGPR held -> K=32 AGPR fits occ-2.
        aq8 = [
            [transpose8(q_cmp[u][2 * k2], q_cmp[u][2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
            for u in range_constexpr(DTW)
        ]
        ado8 = [
            [transpose8(do_cmp[u][2 * k2], do_cmp[u][2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
            for u in range_constexpr(DTW)
        ]
        for g in range_constexpr(RGROUPS):
            bsel = g % DBUF  # which dS/P LDS buffer this group uses
            bs_e = fx.Index(bsel * DS_LDS)  # element offset into lds_ds/lds_p
            bs_b = bsel * DS_LDS * 2  # byte offset for tr_ks
            for dsv, ppv, h, r in loads[g]:
                Vec(dsv).store(lds_ds, [bs_e + h * fx.Index(SROW) + r])
                Vec(ppv).store(lds_p, [bs_e + h * fx.Index(SROW) + r])
            gpu.barrier()
            if g + 1 < RGROUPS:  # prefetch next group during this compute
                loads[g + 1] = hbm_load(g + 1)

            def _asm_pair(rt, bs_b=bs_b):
                return (_tr16_packed(ds_off + bs_b, rt), _tr16_packed(p_off + bs_b, rt))

            def _d8(a, b):  # 2 v2i32 (4 bf16 each) -> direct v8 bf16 (no shuffle crossbar)
                va = Vec(Vec(a).bitcast(i16))
                vb = Vec(Vec(b).bitcast(i16))
                return _raw(
                    Vec.from_elements([va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], i16).bitcast(elem)
                )

            def _asm_v8(pair):
                ds_r, p_r = pair
                bds8 = [_d8(ds_r[2 * k2], ds_r[2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
                bp8 = [_d8(p_r[2 * k2], p_r[2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
                return bds8, bp8

            def _imma(a, b, c):
                return rocdl.mfma_f32_16x16x32_bf16(v4f, [a, b, c])

            for rt in range_constexpr(RT_PER_G):  # rank-tiles per group (GSZ//16)
                rank = fx.Index(g * GSZ + rt * 16) + lo
                if const_expr(_ASM):
                    # within-tile: packed async reads + ONE drain -> the tile's latencies overlap
                    cur = _asm_pair(rt)
                    _wait_lgkm(0)
                    bds8, bp8 = _asm_v8(cur)
                else:
                    b_ds = [tr_ks(ds_off + bs_b, fx.Index(rt), ks) for ks in range_constexpr(KS16)]
                    b_p = [tr_ks(p_off + bs_b, fx.Index(rt), ks) for ks in range_constexpr(KS16)]
                    bds8 = [_concat_bf16_v8(b_ds[2 * k2], b_ds[2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
                    bp8 = [_concat_bf16_v8(b_p[2 * k2], b_p[2 * k2 + 1]) for k2 in range_constexpr(KS16 // 2)]
                for u in range_constexpr(DTW):
                    if const_expr(_PRIO):
                        rocdl.s_setprio(1)
                    if const_expr(_ILP):
                        # split each acc chain into even/odd k2 -> 4 independent mfma chains (vs
                        # 2) so the mfma RAW latency is hidden by more in-wave ILP.
                        a0a = Vec.filled(4, 0.0, fx.Float32)
                        a0b = Vec.filled(4, 0.0, fx.Float32)
                        a1a = Vec.filled(4, 0.0, fx.Float32)
                        a1b = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):
                            if k2 % 2 == 0:
                                a0a = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq8[u][k2], bds8[k2], a0a])
                                a1a = rocdl.mfma_f32_16x16x32_bf16(v4f, [ado8[u][k2], bp8[k2], a1a])
                            else:
                                a0b = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq8[u][k2], bds8[k2], a0b])
                                a1b = rocdl.mfma_f32_16x16x32_bf16(v4f, [ado8[u][k2], bp8[k2], a1b])
                        ov = Vec.from_elements(
                            [
                                fx.Float32(_raw(Vec(a0a)[i]))
                                + fx.Float32(_raw(Vec(a0b)[i]))
                                + fx.Float32(_raw(Vec(a1a)[i]))
                                + fx.Float32(_raw(Vec(a1b)[i]))
                                for i in range_constexpr(4)
                            ],
                            fx.Float32,
                        )
                    elif const_expr(_BSTACK):
                        # stacked-K: both mfmas feed ONE acc (interm[d,r] = Q@dS + dO@P) -> no
                        # combine adds. Trades dual-acc ILP for fewer instructions (issue-bound).
                        acc = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):
                            acc = _imma(aq8[u][k2], bds8[k2], acc)
                            acc = _imma(ado8[u][k2], bp8[k2], acc)
                        ov = Vec.from_elements([fx.Float32(_raw(Vec(acc)[i])) for i in range_constexpr(4)], fx.Float32)
                    else:
                        acc0 = Vec.filled(4, 0.0, fx.Float32)
                        acc1 = Vec.filled(4, 0.0, fx.Float32)
                        for k2 in range_constexpr(KS16 // 2):  # K=32 MFMA: half the MFMA count
                            acc0 = _imma(aq8[u][k2], bds8[k2], acc0)
                            acc1 = _imma(ado8[u][k2], bp8[k2], acc1)
                        ov = Vec.from_elements(
                            [
                                fx.Float32(_raw(Vec(acc0)[i])) + fx.Float32(_raw(Vec(acc1)[i]))
                                for i in range_constexpr(4)
                            ],
                            fx.Float32,
                        )
                    if const_expr(_PRIO):
                        rocdl.s_setprio(0)
                    d = wbase_d + fx.Index(u * 16) + grp * fx.Index(4)  # global d
                    base = (rank * fx.Index(D_V) + d) * fx.Index(2)  # token folded into SRD base
                    bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(ov)[i])) for i in range_constexpr(4)], elem)
                    buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, base, offset_is_bytes=True)
            # WAR barrier only when single-buffered (DBUF=1): the next group's store reuses the
            # same buffer, so wait for this group's dS/P reads to drain. DBUF=2 stores the other
            # buffer -> no WAR barrier needed.
            if DBUF == 1:
                gpu.barrier()

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        DO: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        INTERM: fx.Tensor,
        T: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, DO, DS, PP, INTERM, T).launch(
            grid=(fx.Index(T), fx.Index(DBLK), 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


def build_interm_blocked(topk_len, num_heads, BD, qpad=16, spad=8):
    elem = fx.BFloat16
    R_CHUNK = topk_len
    KS16 = num_heads // 16
    DBLK = D_V // BD  # number of d-blocks (grid.y)
    DT_BLK = BD // 16  # d-tiles per block
    _BLKPRIO = 1  # s_setprio(1) around blocked mfma
    # BLKSTACK (stacked-K, blocked twin of regtr BSTACK): accumulate Q@dS + dO@P into ONE acc chain
    # -> removes the 4 fp32 combine adds/output (blocked is issue-bound). Gated to topk<=128 or
    # topk>=512 (excludes R160 local-band). Only flash uses blocked (pro is regtr).
    _BLKSTACK = 1 if (topk_len <= 128 or topk_len >= 512) else 0
    RGROUPS = R_CHUNK // 64  # rank-groups (each = 4 rank-tiles = 4 waves)
    # tr16 bank-pad: pad the LDS row stride so consecutive tr16 rows land in distinct banks
    # (gfx950 = 64 banks). Q/dO row BD->BD+16, dS/P row 64->72. Occupancy-gated: pro lands at
    # 76KB (2WG/CU); a wider pad crosses 80KB -> 1WG and collapses.
    QROW = BD + qpad  # padded LDS row stride for Q/dO
    SROW = 64 + spad  # padded LDS row stride for dS/P
    Q_LDS = num_heads * QROW  # [h][BD] padded
    DS_LDS = num_heads * SROW  # [h][64] padded
    QDATA8 = (num_heads * BD) // 8  # coalesced vec8 count for Q/dO stage (actual data)
    SDATA8 = (num_heads * 64) // 8  # for dS/P stage (actual data)
    QIT = QDATA8 // THREADS  # stage iters (must divide)
    SIT = SDATA8 // THREADS

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_interm_blocked_smem")
    q_off = allocator._align(allocator.ptr, 16)
    do_off = allocator._align(q_off + Q_LDS * 2, 16)
    ds_off = allocator._align(do_off + Q_LDS * 2, 16)
    p_off = allocator._align(ds_off + DS_LDS * 2, 16)
    allocator.ptr = allocator._align(p_off + DS_LDS * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Q: fx.Tensor, DO: fx.Tensor, DS: fx.Tensor, PP: fx.Tensor, INTERM: fx.Tensor, T: fx.Int32):
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(Q_LDS,)).get()
        lds_do = SmemPtr(allocator.get_base(), do_off, elem.ir_type, shape=(Q_LDS,)).get()
        lds_ds = SmemPtr(allocator.get_base(), ds_off, elem.ir_type, shape=(DS_LDS,)).get()
        lds_p = SmemPtr(allocator.get_base(), p_off, elem.ir_type, shape=(DS_LDS,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        dblk = fx.Index(gpu.block_idx.y)
        Hn = fx.Index(num_heads)

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D_V * 2))
        )
        ds_rsrc = buffer_ops.create_buffer_resource(
            DS, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2))
        )
        pp_rsrc = buffer_ops.create_buffer_resource(
            PP, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(R_CHUNK * 2))
        )
        # interm can exceed 4GB (pro cr4 = 4.8GB) -> flat entry offset overflows the 32-bit
        # buffer voffset. Rebase the SRD to THIS token's slab (i64 base) so the in-slab
        # offset (rank*D_V+d < R_CHUNK*D_V) stays int32-safe.
        interm_rsrc = make_bf16_rebased_rsrc(
            INTERM, token * fx.Index(R_CHUNK) * fx.Index(D_V), fx.Index(R_CHUNK) * fx.Index(D_V * 2)
        )

        q_tok = token * Hn * fx.Index(DQK)
        do_tok = token * Hn * fx.Index(D_V)
        ds_tok = token * Hn * fx.Index(R_CHUNK)
        d0 = dblk * fx.Index(BD)

        # Software-pipeline the dS/P staging: the per-group flow (load -> LDS store -> barrier ->
        # compute) fully exposes the HBM load each group. Prefetch the next group's dS/P into
        # registers during this group's compute so it overlaps the MFMAs (single LDS buffer).
        def hbm_load(g):
            regs = []
            for it in range_constexpr(SIT):
                v = tid + fx.Index(it * THREADS)
                flat = v * fx.Index(8)
                h = flat // fx.Index(64)
                r = flat % fx.Index(64)
                dsv = buffer_ops.buffer_load(
                    ds_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * 64) + r, vec_width=8, dtype=elem
                )
                ppv = buffer_ops.buffer_load(
                    pp_rsrc, ds_tok + h * fx.Index(R_CHUNK) + fx.Index(g * 64) + r, vec_width=8, dtype=elem
                )
                regs.append((dsv, ppv, h, r))
            return regs

        # Issue group 0's dS/P HBM load FIRST (before the Q/dO stage) so it overlaps the whole
        # Q/dO staging + its barrier. Depth 1: one load in flight (prefetched during the
        # previous group's compute) already hides the load; deeper prefetch only adds registers.
        PF_DEPTH = 1
        pending = [hbm_load(0)]

        # ---- stage Q/dO[H][BD] once (coalesced vec8, [h][d] natural).
        for it in range_constexpr(QIT):
            v = tid + fx.Index(it * THREADS)
            flat = v * fx.Index(8)
            h = flat // fx.Index(BD)
            d = flat % fx.Index(BD)
            qv = buffer_ops.buffer_load(q_rsrc, q_tok + h * fx.Index(DQK) + d0 + d, vec_width=8, dtype=elem)
            dov = buffer_ops.buffer_load(do_rsrc, do_tok + h * fx.Index(D_V) + d0 + d, vec_width=8, dtype=elem)
            Vec(qv).store(lds_q, [h * fx.Index(QROW) + d])
            Vec(dov).store(lds_do, [h * fx.Index(QROW) + d])
        gpu.barrier()

        def tr_ks(off, stride, coltile, ks):
            # h-block ks -> rows ks*16.., col tile (d-tile or rank-tile) -> coltile*16 within row.
            row = (
                (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * fx.Index(stride)
                + coltile * fx.Index(16)
                + lo_m4 * fx.Index(4)
            )
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

        # A-operand tr16 reads are kept inline (re-read per group): the kernel is dS/P-re-read
        # bound, not A-LDS-read bound. Any build-time branch here must use a ternary, NOT an
        # in-kernel `if` (flydsl rewrites `if` to scf.if, scoping out assigned values).
        for g in range_constexpr(RGROUPS):
            rbase = fx.Index(g * 64)
            # store this group's (already HBM-loaded) dS/P into LDS.
            for dsv, ppv, h, r in pending[0]:
                Vec(dsv).store(lds_ds, [h * fx.Index(SROW) + r])
                Vec(ppv).store(lds_p, [h * fx.Index(SROW) + r])
            gpu.barrier()

            # keep PF_DEPTH loads in flight: prefetch group g+PF_DEPTH during this compute.
            gnext = g + PF_DEPTH
            if gnext < RGROUPS:
                pending.append(hbm_load(gnext))

            rank = rbase + wave * fx.Index(16) + lo  # this wave's rank-tile, row=lo
            # tr16-read batching + 2-acc: (1) HOIST the B operands (dS/P, index only wave,ks) out
            # of the dt loop -> read once, reuse across d-tiles. (2) PRE-ISSUE the A operands per
            # d-tile so all KS16 tr16 reads are in flight. (3) 2-acc split summed at store.
            b_ds = [tr_ks(ds_off, SROW, wave, ks) for ks in range_constexpr(KS16)]
            b_p = [tr_ks(p_off, SROW, wave, ks) for ks in range_constexpr(KS16)]
            # This kernel is LDS-read-latency bound at occ-1 (MFMAs stall on tr16 reads); more
            # occupancy, not more in-flight depth, is what would hide it.
            for dt in range_constexpr(DT_BLK):
                a_q = [tr_ks(q_off, QROW, fx.Index(dt), ks) for ks in range_constexpr(KS16)]
                a_do = [tr_ks(do_off, QROW, fx.Index(dt), ks) for ks in range_constexpr(KS16)]
                if const_expr(_BLKPRIO):
                    rocdl.s_setprio(1)
                if const_expr(_BLKSTACK):
                    acc0 = Vec.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(KS16):
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_q[ks], b_ds[ks], acc0])
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_do[ks], b_p[ks], acc0])
                    acc1 = None
                else:
                    acc0 = Vec.filled(4, 0.0, fx.Float32)
                    acc1 = Vec.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(KS16):
                        acc0 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_q[ks], b_ds[ks], acc0])
                        acc1 = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_do[ks], b_p[ks], acc1])
                if const_expr(_BLKPRIO):
                    rocdl.s_setprio(0)
                if const_expr(_BLKSTACK):
                    ov = Vec.from_elements([fx.Float32(_raw(Vec(acc0)[i])) for i in range_constexpr(4)], fx.Float32)
                else:
                    ov = Vec.from_elements(
                        [fx.Float32(_raw(Vec(acc0)[i])) + fx.Float32(_raw(Vec(acc1)[i])) for i in range_constexpr(4)],
                        fx.Float32,
                    )
                d = d0 + fx.Index(dt * 16) + grp * fx.Index(4)
                base = (rank * fx.Index(D_V) + d) * fx.Index(2)  # token folded into SRD base
                bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(ov)[i])) for i in range_constexpr(4)], elem)
                buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, base, offset_is_bytes=True)
            gpu.barrier()  # before next group re-stages dS/P
            pending.pop(0)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        DO: fx.Tensor,
        DS: fx.Tensor,
        PP: fx.Tensor,
        INTERM: fx.Tensor,
        T: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, DO, DS, PP, INTERM, T).launch(
            grid=(fx.Index(T), fx.Index(DBLK), 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


# The transpose for this head-contract GEMM is HW-cheapest via ds_read_tr16 from LDS: the
# dS/P LDS staging is essential to coalesce the dS/P read, so dropping it to fit a larger BD
# loses, and global_load_tr* (a gfx12/RDNA WMMA intrinsic) is not in the CDNA4 (gfx950) ISA.

# v2: LDS-staged, coalesced. Each WG owns one token x one d-tile(16): 256 threads coalesced-
# stage that d-tile's Q/dO[H][16] into LDS once, then 4 waves loop rank-tiles feeding the MFMA
# A operand via ds_read_tr16_b64. MFMA = mfma_f32_16x16x16bf16_1k (K=16). Grid=(T, D_V//16).
QLDS = 16  # staged d-width per h-row (one d-tile)


# ============================================================================
# kernel: gather
# ============================================================================

LANES = 64
EPL = D_V // LANES  # 8 fp32 cols per lane -> two v4 halves (lo/hi)


def _store_dkv_bf16(dkv_rsrc, col_base, raw_lo, raw_hi, lane, zero_rope_tail=True):
    """Cast this lane's fp32 v4 lo/hi dKV accumulators to bf16 and store the 8 dKV cols
    [col_base .. col_base+7]. Folds the host dkv_acc.to(bf16) cast into the final write
    (reduction stays fp32 in-register, store rounds to bf16); two dwordx2 stores.

    The first 8 lanes also zero the dead 64-column RoPE tail. Every final gather launches
    exactly one writer wave for every dKV row, so this folds the tail initialization into
    the gather and avoids a full-tensor zero pass over the 512 live columns."""
    elem = fx.BFloat16
    bf_lo = Vec.from_elements([fx.BFloat16(_raw(Vec(raw_lo)[i])) for i in range_constexpr(4)], elem)
    bf_hi = Vec.from_elements([fx.BFloat16(_raw(Vec(raw_hi)[i])) for i in range_constexpr(4)], elem)
    buffer_ops.buffer_store(_raw(bf_lo.bitcast(fx.Int32)), dkv_rsrc, col_base * fx.Index(2), offset_is_bytes=True)
    buffer_ops.buffer_store(
        _raw(bf_hi.bitcast(fx.Int32)), dkv_rsrc, (col_base + fx.Index(4)) * fx.Index(2), offset_is_bytes=True
    )
    if zero_rope_tail:
        zero_v4 = Vec.from_elements([fx.Int32(0), fx.Int32(0), fx.Int32(0), fx.Int32(0)], fx.Int32)
        buffer_ops.buffer_store(
            _raw(zero_v4),
            dkv_rsrc,
            (col_base + fx.Index(D_V)) * fx.Index(2),
            mask=_raw(ArithValue(lane) < fx.Index((DQK - D_V) // 8)),
            offset_is_bytes=True,
        )


def build_gather(num_waves=8):
    elem = fx.BFloat16
    THREADS = num_waves * LANES
    RED_ELEMS = num_waves * D_V  # LDS: [NW][512] fp32 partials

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Interm: fx.Tensor,
        InvPtr: fx.Tensor,
        InvData: fx.Tensor,
        dKV: fx.Tensor,
        NKV: fx.Int32,
        N_TR: fx.Int32,
    ):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        kv = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)

        invptr_rsrc = buffer_ops.create_buffer_resource(
            InvPtr, max_size=False, num_records_bytes=_raw((fx.Index(NKV) + fx.Index(1)) * fx.Index(4))
        )
        invdata_rsrc = buffer_ops.create_buffer_resource(
            InvData, max_size=False, num_records_bytes=_raw(fx.Index(N_TR) * fx.Index(4))
        )
        dkv_rsrc = buffer_ops.create_buffer_resource(
            dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )  # bf16 dKV

        start = fx.Index(fx.Int32(buffer_ops.buffer_load(invptr_rsrc, kv, vec_width=1, dtype=fx.Int32)))
        end = fx.Index(fx.Int32(buffer_ops.buffer_load(invptr_rsrc, kv + fx.Index(1), vec_width=1, dtype=fx.Int32)))

        col_lo = lane * fx.Index(EPL)  # this lane's 8 d-cols (lo=0..3, hi=4..7)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for e, iter_args in range(start + wave, end, fx.Index(num_waves), init=[acc_lo, acc_hi]):
            entry = fx.Index(fx.Int32(buffer_ops.buffer_load(invdata_rsrc, e, vec_width=1, dtype=fx.Int32)))
            i_reb = make_bf16_rebased_rsrc(Interm, entry * fx.Index(D_V), fx.Index(D_V * 2))
            # single vec8 load (dwordx4) of this lane's 8 contiguous d-cols, split lo/hi ->
            # halves the load instruction count vs 2x vec4 (gather is issue-bound).
            iv8 = Vec(buffer_ops.buffer_load(i_reb, col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = Vec(iter_args[0]) + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = Vec(iter_args[1]) + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        # write this wave's partial to LDS[wave][col], reduce over waves, wave0 -> dkv.
        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if num_waves > 1:
            if wave == fx.Index(0):
                s_lo = Vec(loop_res[0])
                s_hi = Vec(loop_res[1])
                for w in range_constexpr(1, num_waves):
                    s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                    s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
                _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, _raw(s_lo), _raw(s_hi), lane)
        else:
            _store_dkv_bf16(dkv_rsrc, kv * fx.Index(DQK) + col_lo, loop_res[0], loop_res[1], lane)

    @flyc.jit
    def launch(Interm, InvPtr, InvData, dKV, NKV, N_TR, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Interm, InvPtr, InvData, dKV, NKV, N_TR).launch(
            grid=(fx.Index(NKV), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


def build_partial_reduce(num_splits):
    """Sum Partial[pidx, 0..num_splits-1, :D_V] -> dKV[KV_BASE+pidx, :D_V] (f32). One WG per
    pool kv, 64 lanes x EPL cols. Companion pass-2 for build_gather_pool (num_splits-split)."""

    def _b():
        @flyc.kernel(known_block_size=[LANES, 1, 1])
        def k_fn(Partial: fx.Tensor, dKV: fx.Tensor, NKV: fx.Int32, KV_BASE: fx.Int32, NPOOL: fx.Int32):
            pidx = fx.Index(gpu.block_idx.x)
            kv = pidx + fx.Index(KV_BASE)
            lane = fx.Index(gpu.thread_idx.x)
            col_lo = lane * fx.Index(EPL)
            part_rsrc = buffer_ops.create_buffer_resource(
                Partial,
                max_size=False,
                num_records_bytes=_raw(fx.Index(NPOOL) * fx.Index(num_splits) * fx.Index(D_V * 4)),
            )
            dkv_rsrc = buffer_ops.create_buffer_resource(
                dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
            )  # bf16 dKV
            base0 = (pidx * fx.Index(num_splits)) * fx.Index(D_V)
            acc_lo = Vec(buffer_ops.buffer_load(part_rsrc, base0 + col_lo, vec_width=4, dtype=fx.Float32))
            acc_hi = Vec(buffer_ops.buffer_load(part_rsrc, base0 + col_lo + fx.Index(4), vec_width=4, dtype=fx.Float32))
            for sx in range_constexpr(1, num_splits):
                b = (pidx * fx.Index(num_splits) + fx.Index(sx)) * fx.Index(D_V)
                acc_lo = acc_lo + Vec(buffer_ops.buffer_load(part_rsrc, b + col_lo, vec_width=4, dtype=fx.Float32))
                acc_hi = acc_hi + Vec(
                    buffer_ops.buffer_load(part_rsrc, b + col_lo + fx.Index(4), vec_width=4, dtype=fx.Float32)
                )
            _store_dkv_bf16(
                dkv_rsrc,
                kv * fx.Index(DQK) + col_lo,
                _raw(acc_lo),
                _raw(acc_hi),
                lane,
                zero_rope_tail=False,
            )

        @flyc.jit
        def launch(Partial, dKV, NKV, KV_BASE, NPOOL, stream):
            k_fn(Partial, dKV, NKV, KV_BASE, NPOOL).launch(
                grid=(fx.Index(NPOOL), 1, 1), block=(LANES, 1, 1), stream=stream
            )

        return _attach(launch)

    return _b()


def build_gather_banded(num_waves=8, row_chunk=None):
    """Closed-form banded gather for the SWA local window (W=128); the inverse window map
    is closed-form so the whole CSR path is skipped. For kv=j, flat interm row =
    (j+local)*RC + (W-1-local), count = min(W, T-j); ``row_chunk`` = interm row stride (R_CHUNK)."""
    elem = fx.BFloat16
    W = 128
    RC = W if row_chunk is None else row_chunk  # interm row stride (ranks per token)
    THREADS = num_waves * LANES
    RED_ELEMS = num_waves * D_V

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_banded_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(Interm: fx.Tensor, dKV: fx.Tensor, NKV: fx.Int32, T: fx.Int32):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        kv = fx.Index(gpu.block_idx.x)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)

        dkv_rsrc = buffer_ops.create_buffer_resource(
            dKV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )  # bf16 dKV
        # cr0 interm is 512MB < 4GB, so the flat element offset is int32-safe -> a single plain
        # buffer resource works; no per-entry SRD rebase (that i64 rebase is only for pro cr4's
        # >4GB interm, which never uses this banded path).
        interm_rsrc = buffer_ops.create_buffer_resource(
            Interm, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(RC) * fx.Index(D_V * 2))
        )

        # count = min(W, T - kv): only the last W-1 kv have a short (edge) window.
        rem = fx.Index(T) - kv
        count = fx.Index(arith.select(rem < fx.Index(W), _raw(rem), _raw(fx.Index(W))))

        col_lo = lane * fx.Index(EPL)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for local, iter_args in range(wave, count, fx.Index(num_waves), init=[acc_lo, acc_hi]):
            a_lo = Vec(iter_args[0])
            a_hi = Vec(iter_args[1])
            entry = (kv + local) * fx.Index(RC) + fx.Index(W - 1) - local
            erow = entry * fx.Index(D_V)
            iv8 = Vec(buffer_ops.buffer_load(interm_rsrc, erow + col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)  # 1x dwordx4 vs 2x dwordx2
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = a_lo + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = a_hi + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if num_waves > 1:
            if wave == fx.Index(0):
                s_lo = Vec(loop_res[0])
                s_hi = Vec(loop_res[1])
                for w in range_constexpr(1, num_waves):
                    s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                    s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
                _store_dkv_bf16(
                    dkv_rsrc,
                    kv * fx.Index(DQK) + col_lo,
                    _raw(s_lo),
                    _raw(s_hi),
                    lane,
                    zero_rope_tail=row_chunk is None,
                )
        else:
            _store_dkv_bf16(
                dkv_rsrc,
                kv * fx.Index(DQK) + col_lo,
                loop_res[0],
                loop_res[1],
                lane,
                zero_rope_tail=row_chunk is None,
            )

    @flyc.jit
    def launch(Interm, dKV, NKV, T, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        # Launch only the T LOCAL kv (banded window). For cr0 T==NKV (all kv are local);
        # for cr128 the P pool kv (>=T) are handled by the pool-only CSR gather separately.
        k_fn(Interm, dKV, NKV, T).launch(grid=(fx.Index(T), 1, 1), block=(THREADS, 1, 1), stream=stream)

    return _attach(launch)


def build_gather_pool(num_waves=16, num_splits=32):
    """Closed-form pool gather for cr=128 (deterministic causal HCA pool): token i attends
    pool block b (kv=T+b) iff i >= (b+1)*cr_pool - 1 and visibility is monotone in b, so
    column b == block b == kv gives a closed-form inverse. Same NS-split + LDS reduce."""
    THREADS = num_waves * LANES
    RED_ELEMS = num_waves * D_V
    GW = num_splits * num_waves
    elem = fx.BFloat16

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="mla_bwd_gather_pool_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(red_off + RED_ELEMS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Interm: fx.Tensor,
        Partial: fx.Tensor,
        NKV: fx.Int32,
        N_TR: fx.Int32,
        T: fx.Int32,
        NPOOL: fx.Int32,
        CR_POOL: fx.Int32,
        R_CHUNK: fx.Int32,
        R_OFF: fx.Int32,
    ):
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        lds_red = SmemPtr(allocator.get_base(), red_off, fx.Float32.ir_type, shape=(RED_ELEMS,)).get()
        pidx = fx.Index(gpu.block_idx.x)  # pool block b (0..NPOOL-1)
        s = fx.Index(gpu.block_idx.y)
        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(LANES)
        lane = tid % fx.Index(LANES)
        gw = s * fx.Index(num_waves) + wave

        interm_rsrc = buffer_ops.create_buffer_resource(
            Interm, max_size=False, num_records_bytes=_raw(fx.Index(N_TR) * fx.Index(D_V * 2))
        )
        part_rsrc = buffer_ops.create_buffer_resource(
            Partial,
            max_size=False,
            num_records_bytes=_raw(fx.Index(NPOOL) * fx.Index(num_splits) * fx.Index(D_V * 4)),
        )

        # start_i(b) = (b+1)*cr_pool - 1 (first token that sees block b); count = T - start_i
        # (>= 1 for every block since the last block's start_i = T - 1). rank = R_OFF + b.
        start_i = (pidx + fx.Index(1)) * fx.Index(CR_POOL) - fx.Index(1)
        count = fx.Index(T) - start_i
        rank = fx.Index(R_OFF) + pidx

        col_lo = lane * fx.Index(EPL)
        acc_lo = Vec.filled(4, 0.0, fx.Float32)
        acc_hi = Vec.filled(4, 0.0, fx.Float32)

        loop_res = [acc_lo, acc_hi]
        for j, iter_args in range(gw, count, fx.Index(GW), init=[acc_lo, acc_hi]):
            a_lo = Vec(iter_args[0])
            a_hi = Vec(iter_args[1])
            tok = start_i + j
            erow = (tok * fx.Index(R_CHUNK) + rank) * fx.Index(D_V)
            iv8 = Vec(buffer_ops.buffer_load(interm_rsrc, erow + col_lo, vec_width=8, dtype=elem))
            iv_lo = Vec.from_elements([iv8[i] for i in range_constexpr(4)], elem)  # 1x dwordx4 vs 2x dwordx2
            iv_hi = Vec.from_elements([iv8[4 + i] for i in range_constexpr(4)], elem)
            n_lo = a_lo + Vec(arith.ExtFOp(v4f_ty, _raw(iv_lo)).result)
            n_hi = a_hi + Vec(arith.ExtFOp(v4f_ty, _raw(iv_hi)).result)
            loop_res = yield [_raw(n_lo), _raw(n_hi)]

        Vec(loop_res[0]).store(lds_red, [wave * fx.Index(D_V) + col_lo])
        Vec(loop_res[1]).store(lds_red, [wave * fx.Index(D_V) + col_lo + fx.Index(4)])
        gpu.barrier()

        if wave == fx.Index(0):
            s_lo = Vec(loop_res[0])
            s_hi = Vec(loop_res[1])
            for w in range_constexpr(1, num_waves):
                s_lo = s_lo + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo])
                s_hi = s_hi + Vec.load(v4f_ty, lds_red, [fx.Index(w * D_V) + col_lo + fx.Index(4)])
            poff = (pidx * fx.Index(num_splits) + s) * fx.Index(D_V)
            buffer_ops.buffer_store(_raw(s_lo), part_rsrc, (poff + col_lo) * fx.Index(4), offset_is_bytes=True)
            buffer_ops.buffer_store(
                _raw(s_hi), part_rsrc, (poff + col_lo + fx.Index(4)) * fx.Index(4), offset_is_bytes=True
            )

    @flyc.jit
    def launch(Interm, Partial, NKV, N_TR, T, NPOOL, CR_POOL, R_CHUNK, R_OFF, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Interm, Partial, NKV, N_TR, T, NPOOL, CR_POOL, R_CHUNK, R_OFF).launch(
            grid=(fx.Index(NPOOL), num_splits, 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


# ============================================================================
# kernel: fused
# ============================================================================

HPW = 16
D_LDS = 528
TILE_K = 16
KS = D // 32  # QK K-steps (16)
DT = D // 16  # PV / interm d-tiles (32)
BD_INT = 128  # interm Q/dO resident d-block width
NDB = D // BD_INT  # 4 d-blocks
BD_LDSB = BD_INT + 8  # padded LDS row stride for the d-block (136)
KSB = BD_INT // 32  # QK-packs per d-block (4)


def build_bwd_fused(
    topk_len,
    scale,
    num_heads=64,
    flash_cr4_kb=0,
    flash_cr4_pv2x=False,
):
    # 3 = K32 2-chain ILP; gated to the short-R flash fused shapes (cr0 R128, cr128 R160).
    interm_chains = 3 if topk_len <= 160 else 1
    elem = fx.BFloat16
    NUM_TILES = (topk_len + TILE_K - 1) // TILE_K
    flash_cr4_kb = (
        int(flash_cr4_kb) if int(topk_len) == 640 and int(num_heads) == 64 and int(flash_cr4_kb) in {5, 10, 20} else 0
    )
    flash_cr4_pv2x = bool(flash_cr4_pv2x) and int(topk_len) == 640 and int(num_heads) == 64 and flash_cr4_kb == 10
    # batch factor: kv-tiles per d-block stage, amortizing the Q/dO restage barriers. For the
    # few-tile flash fused shapes (NUM_TILES<=10: cr0=8, cr128=10) a single batch (KB=NUM_TILES)
    # restages Q/dO once (halving the interm-phase restage barriers) and wins isolated; larger
    # tile counts keep the 2-pass split to retain the dQ/interm pipeline overlap.
    KB = (
        flash_cr4_kb
        if flash_cr4_kb
        else (
            NUM_TILES
            if NUM_TILES <= 10
            else (5 if NUM_TILES % 5 == 0 else (4 if NUM_TILES % 4 == 0 else (2 if NUM_TILES % 2 == 0 else 1)))
        )
    )
    BLOCK_H = num_heads  # flash: 64 -> 1 WG all heads
    WAVES = BLOCK_H // HPW  # 4
    THREADS = WAVES * 64  # 256
    KSH = num_heads // 16  # interm head K-steps (flash 4)
    DUW = DT // WAVES // NDB  # interm d-tiles per wave per d-block (32/4/4 = 2)
    KV1 = TILE_K * D_LDS  # one KV tile in LDS (elems)
    # Generic fused keeps its accepted two-tile ping-pong. The exact Flash/cr4 pv2x
    # candidate alternates two *pairs* of buffers: pair p writes set p%2, so the
    # intervening publication barrier also proves the set from p-2 is no longer read.
    # This permits one barrier per pair without a cross-wave PV-read/KV-write race.
    KVBUF = 4 if flash_cr4_pv2x else 2
    KV_ELEMS = KVBUF * KV1
    QB_ELEMS = num_heads * BD_LDSB  # one d-block of Q/dO resident

    allocator = SmemAllocator(
        None,
        arch=get_hip_arch(),
        global_sym_name=(
            "mla_bwd_fused_flash_cr4_kb10_pv2x_smem"
            if flash_cr4_pv2x
            else (f"mla_bwd_fused_flash_cr4_kb{flash_cr4_kb}_smem" if flash_cr4_kb else "mla_bwd_fused_smem")
        ),
    )
    kv_off = allocator._align(allocator.ptr, 16)
    qb_off = allocator._align(kv_off + KV_ELEMS * 2, 16)
    dob_off = allocator._align(qb_off + QB_ELEMS * 2, 16)
    ds_off = allocator._align(dob_off + QB_ELEMS * 2, 16)
    pp_off = allocator._align(ds_off + KB * num_heads * TILE_K * 2, 16)
    mask_off = allocator._align(pp_off + KB * num_heads * TILE_K * 2, 16)
    allocator.ptr = allocator._align(mask_off + KB * TILE_K * 4, 16)
    if flash_cr4_pv2x:
        assert allocator.ptr == 144000
        assert allocator.ptr <= 160 * 1024

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        INTERM: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        base = allocator.get_base()
        lds_kv = SmemPtr(base, kv_off, elem.ir_type, shape=(KV_ELEMS,)).get()
        lds_qb = SmemPtr(base, qb_off, elem.ir_type, shape=(QB_ELEMS,)).get()
        lds_dob = SmemPtr(base, dob_off, elem.ir_type, shape=(QB_ELEMS,)).get()
        lds_ds = SmemPtr(base, ds_off, elem.ir_type, shape=(KB * num_heads * TILE_K,)).get()
        lds_p = SmemPtr(base, pp_off, elem.ir_type, shape=(KB * num_heads * TILE_K,)).get()
        lds_mask = SmemPtr(base, mask_off, fx.Float32.ir_type, shape=(KB * TILE_K,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(64)
        wave = tid // fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        lo_d4 = lo // fx.Index(4)
        lo_m4 = lo % fx.Index(4)

        token = fx.Index(gpu.block_idx.x)
        Hn = fx.Index(H)
        head_A = wave * fx.Index(HPW) + lo

        kv_rsrc = buffer_ops.create_buffer_resource(
            KV, max_size=False, num_records_bytes=_raw(fx.Index(NKV) * fx.Index(DQK * 2))
        )
        tk_rsrc = buffer_ops.create_buffer_resource(
            TOPK, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len * 4))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        delta_rsrc = buffer_ops.create_buffer_resource(
            DELTA, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(4))
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        interm_rsrc = buffer_ops.create_buffer_resource(
            INTERM, max_size=False, num_records_bytes=_raw(fx.Index(T) * fx.Index(topk_len) * fx.Index(D * 2))
        )
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(DQK * 2))
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            DO, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(fx.Index(T) * Hn * fx.Index(D * 2))
        )

        c_log2e = fx.Float32(_LOG2E)
        c_sl = fx.Float32(scale * _LOG2E)
        c_scale = fx.Float32(scale)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero = fx.Float32(0.0)

        # ---- Q/dO register-resident (B operands for QK; also source for interm d-block stage) ----
        q_row = token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK)
        do_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        q_packs = [
            buffer_ops.buffer_load(q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for ks in range_constexpr(KS)
        ]
        do_packs = [
            buffer_ops.buffer_load(do_rsrc, do_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for ks in range_constexpr(KS)
        ]

        lse_h = fx.Float32(buffer_ops.buffer_load(lse_rsrc, token * Hn + head_A, vec_width=1, dtype=fx.Float32))
        lse_h = fx.Float32(arith.MaxNumFOp(_raw(lse_h), _raw(fx.Float32(-3.0e38))).result)
        lse_l2 = fx.Float32(lse_h * c_log2e)
        # FOLD delta = rowsum(O*dO) into fused (was a separate BW-bound kernel fully on the wall).
        # dO already resident in do_packs; stream O (1 vec8 held at a time), fp32-accumulate the
        # bf16 product, then sum the 4 d-groups (lanes lo, lo+16, lo+32, lo+48) via ds_bpermute.
        o_row = token * Hn * fx.Index(D) + head_A * fx.Index(D)
        _dpart = fx.Float32(0.0)
        for ks in range_constexpr(KS):
            _ov = buffer_ops.buffer_load(o_rsrc, o_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem)
            for _i in range_constexpr(8):
                _oi = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(_ov)[_i])).result)
                _di = fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(Vec(do_packs[ks])[_i])).result)
                _dpart = fx.Float32(arith.AddFOp(_raw(_dpart), _raw(_oi * _di)).result)
        _part_i = _raw(Vec.from_elements([_dpart], fx.Float32).bitcast(fx.Int32)[0])
        delta_h = fx.Float32(0.0)
        for _g in range_constexpr(4):
            _idx = _raw(fx.Int32(lo + fx.Index(_g * 16)) * fx.Int32(4))
            _gi = rocdl.ds_bpermute(fx.Int32.ir_type, _idx, _part_i)
            _gf = fx.Float32(_raw(Vec.from_elements([fx.Int32(_gi)], fx.Int32).bitcast(fx.Float32)[0]))
            delta_h = fx.Float32(arith.AddFOp(_raw(delta_h), _raw(_gf)).result)
        # write delta out (one lane per head; d_sink reads it after this kernel)
        buffer_ops.buffer_store(
            _raw(delta_h),
            delta_rsrc,
            (token * Hn + head_A) * fx.Index(4),
            mask=_raw(arith.CmpIOp(arith.CmpIPredicate.eq, _raw(grp), _raw(fx.Index(0))).result),
            offset_is_bytes=True,
        )

        g_row = tid // fx.Index(16)
        g_within = tid % fx.Index(16)
        tk_row = token * fx.Index(topk_len)

        def load_topk(tbase):
            return fx.Int32(buffer_ops.buffer_load(tk_rsrc, tk_row + tbase + g_row, vec_width=1, dtype=fx.Int32))

        def gather_regs(idx):  # HBM -> registers (overlaps compute)
            valid = ArithValue(idx >= fx.Int32(0))
            src = fx.Index(valid.select(idx, fx.Int32(0)))
            vvs = [
                buffer_ops.buffer_load(
                    kv_rsrc,
                    src * fx.Index(DQK) + g_within * fx.Index(32) + fx.Index(c * 8),
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(4)
            ]
            mval = fx.Float32(valid.select(_raw(c_zero), _raw(c_neg_inf)))
            return vvs, mval

        def store_kv(vvs, mval, kt):  # registers -> LDS (KV double-buffer; mask per-kt)
            bufo = (kt % 2) * KV1
            for c in range_constexpr(4):
                Vec(vvs[c]).store(
                    lds_kv,
                    [fx.Index(bufo) + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)],
                )
            if g_within == fx.Index(0):
                Vec.from_elements([mval], fx.Float32).store(lds_mask, [fx.Index(kt * TILE_K) + g_row])

        def store_kv_pair(vvs, mval, kt, bufo):
            # Candidate-only explicit buffer selection. Masks remain indexed by kt
            # because the fused intermediate consumes the full KB dS/P batch.
            for c in range_constexpr(4):
                Vec(vvs[c]).store(
                    lds_kv,
                    [bufo + g_row * fx.Index(D_LDS) + g_within * fx.Index(32) + fx.Index(c * 8)],
                )
            if g_within == fx.Index(0):
                Vec.from_elements([mval], fx.Float32).store(lds_mask, [fx.Index(kt * TILE_K) + g_row])

        # tr16 LDS transpose read helper (A/B operand for the interm head-contraction GEMM)
        def tr_h(off, stride, coltile, ks):
            row = (
                (fx.Index(ks * 16) + grp * fx.Index(4) + lo_d4) * stride + coltile * fx.Index(16) + lo_m4 * fx.Index(4)
            )
            ptr = buffer_ops.create_llvm_ptr(_raw(fx.Int64(row) * fx.Int64(2) + fx.Int64(off)), address_space=3)
            return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

        dq_acc0 = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(DT)]
        for kb, iter_args in range(fx.Index(0), fx.Index(NUM_TILES), fx.Index(KB), init=dq_acc0):
            dq_acc = [iter_args[dt] for dt in range_constexpr(DT)]

            # ============ dQ phase: per-tile QK/dS/dP, stash dS/P to lds_ds[kt] ==============
            if const_expr(flash_cr4_pv2x):
                # Exact KB10 candidate: stage adjacent tiles into the active pair set,
                # publish both with one barrier, retain the generic per-tile QK/softmax
                # arithmetic, then combine only the two PV K16 steps into one K32 MFMA.
                assert KB == 10 and KB % 2 == 0 and NUM_TILES % KB == 0
                _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

                def _paired_qk_softmax_store(kt, bufo):
                    mask4 = Vec.load(
                        v4f,
                        lds_mask,
                        [fx.Index(kt * TILE_K) + grp * fx.Index(4)],
                    )
                    acc_s0 = Vec.filled(4, 0.0, fx.Float32)
                    acc_s1 = Vec.filled(4, 0.0, fx.Float32)
                    acc_dp0 = Vec.filled(4, 0.0, fx.Float32)
                    acc_dp1 = Vec.filled(4, 0.0, fx.Float32)

                    def _bv(ks):
                        return Vec.load(
                            v8,
                            lds_kv,
                            [bufo + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)],
                        )

                    PF = 4
                    bvq = [_bv(k) for k in range_constexpr(PF)]
                    for ks in range_constexpr(KS):
                        if ks + PF < KS:
                            bvq.append(_bv(ks + PF))
                        if ks % 2 == 0:
                            acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s0])
                            acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp0])
                        else:
                            acc_s1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s1])
                            acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp1])
                    acc_s = Vec.from_elements(
                        [
                            fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i]))
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )
                    acc_dp = Vec.from_elements(
                        [
                            fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i]))
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )

                    pvals = [None] * 4
                    dsvals = [None] * 4
                    for i in range_constexpr(4):
                        arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                        p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                        pvals[i] = p
                        dsvals[i] = p * (fx.Float32(_raw(Vec(acc_dp)[i])) - delta_h) * c_scale

                    base_ds = fx.Index(kt * num_heads * TILE_K) + head_A * fx.Index(TILE_K) + grp * fx.Index(4)
                    for i in range_constexpr(4):
                        Vec.from_elements([fx.BFloat16(_raw(dsvals[i]))], elem).store(lds_ds, [base_ds + fx.Index(i)])
                        Vec.from_elements([fx.BFloat16(_raw(pvals[i]))], elem).store(lds_p, [base_ds + fx.Index(i)])
                    return dsvals

                def _paired_v32(base_a, base_b, dt):
                    pa = buffer_ops.create_llvm_ptr(
                        _raw(base_a + fx.Int64(dt * 32)),
                        address_space=3,
                    )
                    pb = buffer_ops.create_llvm_ptr(
                        _raw(base_b + fx.Int64(dt * 32)),
                        address_space=3,
                    )
                    va = Vec(rocdl.ds_read_tr16_b64(v4, pa).result).bitcast(fx.Int16)
                    vb = Vec(rocdl.ds_read_tr16_b64(v4, pb).result).bitcast(fx.Int16)
                    return _raw(
                        Vec.from_elements(
                            [
                                va[0],
                                va[1],
                                va[2],
                                va[3],
                                vb[0],
                                vb[1],
                                vb[2],
                                vb[3],
                            ],
                            fx.Int16,
                        ).bitcast(elem)
                    )

                def _mma_agpr_k32(a, b, c, _v4f32_ir=_v4f32_ir):
                    op = _llvm.InlineAsmOp(
                        res=_v4f32_ir,
                        operands_=[_raw(a), _raw(b), _raw(c)],
                        asm_string=("v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0"),
                        constraints="=a,v,v,0",
                        has_side_effects=False,
                    )
                    return op.result

                for kp in range_constexpr(KB // 2):
                    kt_a = 2 * kp
                    kt_b = kt_a + 1
                    t_a = kb + fx.Index(kt_a)
                    t_b = kb + fx.Index(kt_b)
                    pair_set = kp % 2
                    buf_a = fx.Index(pair_set * 2 * KV1)
                    buf_b = buf_a + fx.Index(KV1)

                    kv_a, m_a = gather_regs(load_topk(t_a * fx.Index(TILE_K)))
                    store_kv_pair(kv_a, m_a, kt_a, buf_a)
                    kv_b, m_b = gather_regs(load_topk(t_b * fx.Index(TILE_K)))
                    store_kv_pair(kv_b, m_b, kt_b, buf_b)
                    gpu.barrier()

                    ds_a = _paired_qk_softmax_store(kt_a, buf_a)
                    ds_b = _paired_qk_softmax_store(kt_b, buf_b)
                    pB = _raw(
                        Vec.from_elements(
                            [fx.BFloat16(_raw(ds_a[i])) for i in range_constexpr(4)]
                            + [fx.BFloat16(_raw(ds_b[i])) for i in range_constexpr(4)],
                            elem,
                        )
                    )

                    pv_row = (grp * fx.Index(4) + lo_d4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4)
                    base_a = fx.Int64((buf_a + pv_row) * fx.Index(2) + fx.Index(kv_off))
                    base_b = fx.Int64((buf_b + pv_row) * fx.Index(2) + fx.Index(kv_off))
                    _PVPF = 8
                    trq = [_paired_v32(base_a, base_b, dt) for dt in range_constexpr(_PVPF)]
                    new_dq = [None] * DT
                    for dt in range_constexpr(DT):
                        if dt + _PVPF < DT:
                            trq.append(
                                _paired_v32(
                                    base_a,
                                    base_b,
                                    dt + _PVPF,
                                )
                            )
                        new_dq[dt] = _mma_agpr_k32(trq[dt], pB, dq_acc[dt])
                    dq_acc = new_dq
            else:
                # Generic fused control: retain the accepted per-tile K16 PV path.
                for kt in range_constexpr(KB):
                    t = kb + fx.Index(kt)
                    bufo = (kt % 2) * KV1  # KV double-buffer offset (ping-pong by kt%2)
                    kv_cur, m_cur = gather_regs(load_topk(t * fx.Index(TILE_K)))
                    store_kv(kv_cur, m_cur, kt)
                    # KV/mask visible; kt+2 reuses this buf only after the next tile's barrier below.
                    gpu.barrier()

                    mask4 = Vec.load(v4f, lds_mask, [fx.Index(kt * TILE_K) + grp * fx.Index(4)])

                    # QK x2 (S, dP): A=KV(lds), B=Q/dO(regs). 2-acc ILP.
                    acc_s0 = Vec.filled(4, 0.0, fx.Float32)
                    acc_s1 = Vec.filled(4, 0.0, fx.Float32)
                    acc_dp0 = Vec.filled(4, 0.0, fx.Float32)
                    acc_dp1 = Vec.filled(4, 0.0, fx.Float32)

                    def _bv(ks, bufo=bufo):
                        return Vec.load(
                            v8,
                            lds_kv,
                            [fx.Index(bufo) + lo * fx.Index(D_LDS) + fx.Index(ks * 32) + grp * fx.Index(8)],
                        )

                    PF = 4
                    bvq = [_bv(k) for k in range_constexpr(PF)]
                    for ks in range_constexpr(KS):
                        if ks + PF < KS:
                            bvq.append(_bv(ks + PF))
                        if ks % 2 == 0:
                            acc_s0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s0])
                            acc_dp0 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp0])
                        else:
                            acc_s1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), q_packs[ks], acc_s1])
                            acc_dp1 = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(bvq[ks]), do_packs[ks], acc_dp1])
                    acc_s = Vec.from_elements(
                        [
                            fx.Float32(_raw(Vec(acc_s0)[i])) + fx.Float32(_raw(Vec(acc_s1)[i]))
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )
                    acc_dp = Vec.from_elements(
                        [
                            fx.Float32(_raw(Vec(acc_dp0)[i])) + fx.Float32(_raw(Vec(acc_dp1)[i]))
                            for i in range_constexpr(4)
                        ],
                        fx.Float32,
                    )

                    # PVPF: PV's V tr16 reads (bvv) depend only on the resident KV-LDS (bufo), not on
                    # softmax -> hoist the first _PVPF reads before the exp2 loop so their LDS-read
                    # latency overlaps the softmax VALU.
                    pv_base = fx.Int64(
                        (lo_d4 * fx.Index(D_LDS) + grp * fx.Index(4) * fx.Index(D_LDS) + lo_m4 * fx.Index(4))
                        * fx.Index(2)
                        + fx.Index(kv_off + bufo * 2)
                    )

                    def _bvv(dt, pv_base=pv_base):
                        ptr = buffer_ops.create_llvm_ptr(
                            _raw(pv_base + fx.Int64(dt * 32)),
                            address_space=3,
                        )
                        return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))

                    _PVPF = 8
                    bvv_pf = [_bvv(dt) for dt in range_constexpr(_PVPF)]

                    pvals = [None] * 4
                    dsvals = [None] * 4
                    for i in range_constexpr(4):
                        arg = fx.Float32(_raw(Vec(acc_s)[i])) * c_sl + fx.Float32(_raw(Vec(mask4)[i])) - lse_l2
                        p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(arg)))
                        pvals[i] = p
                        dsvals[i] = p * (fx.Float32(_raw(Vec(acc_dp)[i])) - delta_h) * c_scale

                    pB = _raw(
                        Vec.from_elements(
                            [fx.BFloat16(_raw(dsvals[i])) for i in range_constexpr(4)],
                            elem,
                        ).bitcast(fx.Int16)
                    )

                    # hand dS/P to LDS scratch [kt][head_A][kv=grp*4+i] for the interm head-contraction.
                    # (Single-bf16 stores pipeline better than a v4-packed store, which adds VALU.)
                    base_ds = fx.Index(kt * num_heads * TILE_K) + head_A * fx.Index(TILE_K) + grp * fx.Index(4)
                    for i in range_constexpr(4):
                        Vec.from_elements([fx.BFloat16(_raw(dsvals[i]))], elem).store(lds_ds, [base_ds + fx.Index(i)])
                        Vec.from_elements([fx.BFloat16(_raw(pvals[i]))], elem).store(lds_p, [base_ds + fx.Index(i)])

                    # ---- PV: dQ += dS@K (A=V tr16, B=dS regs) ---- (per-tile, all DT d-tiles)
                    # dq_acc (loop-carried, occ wall) pinned in AGPR via inline-asm MFMA (=a,v,v,0,
                    # D=C in-place, no accvgpr shuffle) -> frees arch-VGPR.
                    _v4f32_ir = ir.VectorType.get([4], ir.F32Type.get())

                    def _mma_agpr_k16(a, b, c, _v4f32_ir=_v4f32_ir):
                        op = _llvm.InlineAsmOp(
                            res=_v4f32_ir,
                            operands_=[_raw(a), _raw(b), _raw(c)],
                            asm_string=("v_mfma_f32_16x16x16_bf16 " "$0, $1, $2, $0"),
                            constraints="=a,v,v,0",
                            has_side_effects=False,
                        )
                        return op.result

                    new_dq = [None] * DT
                    for dt in range_constexpr(DT):
                        if dt + _PVPF < DT:
                            bvv_pf.append(_bvv(dt + _PVPF))
                        new_dq[dt] = _mma_agpr_k16(bvv_pf[dt], pB, dq_acc[dt])
                    dq_acc = new_dq

            gpu.barrier()  # all KB tiles' dS/P visible before the interm phase

            # ============ interm phase: d-block outer, kv-tile inner (reuse staged Q/dO) ======
            for double_buf in range_constexpr(NDB):
                # stage one 128-d block of Q/dO from registers to LDS (once per KB tiles).
                for kl in range_constexpr(KSB):
                    ks = double_buf * KSB + kl
                    d_local = fx.Index(kl * 32) + grp * fx.Index(8)
                    Vec(q_packs[ks]).store(lds_qb, [head_A * fx.Index(BD_LDSB) + d_local])
                    Vec(do_packs[ks]).store(lds_dob, [head_A * fx.Index(BD_LDSB) + d_local])
                gpu.barrier()  # Q/dO d-block visible

                # Hoist the Q/dO transpose reads (A operands) OUT of the kt loop: aq/ao index only
                # (d-tile, ks), not kt, so they are identical across all KB kv-tiles. The unrolled
                # kt loop is not CSE'd by the compiler, so without hoisting each aq/ao is re-read KB
                # times (the dominant tr16 traffic on the LDS-read-bound interm phase). Read once per
                # d-block, reuse across kt. (Staging 2 d-blocks at once offsets the win at occ-1.)
                aq_h = [
                    [
                        _concat_bf16_v8(
                            tr_h(qb_off, fx.Index(BD_LDSB), wave + fx.Index(u * WAVES), 2 * k2),
                            tr_h(qb_off, fx.Index(BD_LDSB), wave + fx.Index(u * WAVES), 2 * k2 + 1),
                        )
                        for k2 in range_constexpr(KSH // 2)
                    ]
                    for u in range_constexpr(DUW)
                ]
                ao_h = [
                    [
                        _concat_bf16_v8(
                            tr_h(dob_off, fx.Index(BD_LDSB), wave + fx.Index(u * WAVES), 2 * k2),
                            tr_h(dob_off, fx.Index(BD_LDSB), wave + fx.Index(u * WAVES), 2 * k2 + 1),
                        )
                        for k2 in range_constexpr(KSH // 2)
                    ]
                    for u in range_constexpr(DUW)
                ]

                for kt in range_constexpr(KB):
                    t = kb + fx.Index(kt)
                    ds_off_kt = ds_off + kt * num_heads * TILE_K * 2
                    pp_off_kt = pp_off + kt * num_heads * TILE_K * 2
                    # The interm B operands (dS/P) index only (kt, ks), not the d-tile, so their
                    # tr16 reads are hoisted once per (double_buf, kt) and reused across the d-tile loop.
                    bd_h = [
                        _concat_bf16_v8(
                            tr_h(ds_off_kt, fx.Index(TILE_K), fx.Index(0), 2 * k2),
                            tr_h(ds_off_kt, fx.Index(TILE_K), fx.Index(0), 2 * k2 + 1),
                        )
                        for k2 in range_constexpr(KSH // 2)
                    ]
                    bp_h = [
                        _concat_bf16_v8(
                            tr_h(pp_off_kt, fx.Index(TILE_K), fx.Index(0), 2 * k2),
                            tr_h(pp_off_kt, fx.Index(TILE_K), fx.Index(0), 2 * k2 + 1),
                        )
                        for k2 in range_constexpr(KSH // 2)
                    ]
                    for u in range_constexpr(DUW):
                        dt = double_buf * (DT // NDB) + wave + fx.Index(u * WAVES)  # global d-tile
                        if const_expr(interm_chains == 3):
                            # K=32 (halved MFMA count) + 2 independent iacc chains (Q@dS, dO@P)
                            # -> 2-way ILP hides the mfma RAW latency on this head-contract GEMM.
                            iacc_a = Vec.filled(4, 0.0, fx.Float32)
                            iacc_b = Vec.filled(4, 0.0, fx.Float32)
                            for k2 in range_constexpr(KSH // 2):
                                iacc_a = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq_h[u][k2], bd_h[k2], iacc_a])
                                iacc_b = rocdl.mfma_f32_16x16x32_bf16(v4f, [ao_h[u][k2], bp_h[k2], iacc_b])
                            iacc = Vec.from_elements(
                                [
                                    fx.Float32(_raw(Vec(iacc_a)[i])) + fx.Float32(_raw(Vec(iacc_b)[i]))
                                    for i in range_constexpr(4)
                                ],
                                fx.Float32,
                            )
                        else:
                            # K=32 MFMA: concat two K=16 tr16 h-blocks -> v8 (halves MFMA count).
                            iacc = Vec.filled(4, 0.0, fx.Float32)
                            for k2 in range_constexpr(KSH // 2):
                                iacc = rocdl.mfma_f32_16x16x32_bf16(v4f, [aq_h[u][k2], bd_h[k2], iacc])
                            for k2 in range_constexpr(KSH // 2):
                                iacc = rocdl.mfma_f32_16x16x32_bf16(v4f, [ao_h[u][k2], bp_h[k2], iacc])
                        iov = Vec(iacc)
                        kv_g = t * fx.Index(TILE_K) + lo
                        d_g = dt * fx.Index(16) + grp * fx.Index(4)
                        ibase = (token * fx.Index(topk_len) * fx.Index(D) + kv_g * fx.Index(D) + d_g) * fx.Index(2)
                        bf4 = Vec.from_elements([fx.BFloat16(_raw(Vec(iov)[i])) for i in range_constexpr(4)], elem)
                        buffer_ops.buffer_store(_raw(bf4.bitcast(fx.Int32)), interm_rsrc, ibase, offset_is_bytes=True)
                gpu.barrier()  # before next d-block restages lds_qb / lds_dob

            loop_results = yield list(dq_acc)

        dq_acc = [loop_results[dt] for dt in range_constexpr(DT)]
        for dt in range_constexpr(DT):
            ov = Vec(dq_acc[dt])
            base = (
                token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(dt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            pk0 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))))
            pk1 = fx.Int32(_raw(rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))))
            buffer_ops.buffer_store(_raw(Vec.from_elements([pk0, pk1], fx.Int32)), dq_rsrc, base, offset_is_bytes=True)
        # Zero the 64 rope cols (512..575) in-kernel (rope grad is dead).
        zero_v2 = Vec.from_elements([fx.Int32(0), fx.Int32(0)], fx.Int32)
        for rt in range_constexpr(4):
            rbase = (
                token * Hn * fx.Index(DQK) + head_A * fx.Index(DQK) + fx.Index(D + rt * 16) + grp * fx.Index(4)
            ) * fx.Index(2)
            buffer_ops.buffer_store(_raw(zero_v2), dq_rsrc, rbase, offset_is_bytes=True)

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        KV: fx.Tensor,
        DO: fx.Tensor,
        TOPK: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        DQ: fx.Tensor,
        INTERM: fx.Tensor,
        O: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        NKV: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, KV, DO, TOPK, LSE, DELTA, DQ, INTERM, O, T, H, NKV).launch(
            grid=(fx.Index(T), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    return _attach(launch)


# ============================================================================
# host-side dispatch
# ============================================================================

_GATHER_NS = 32  # cr128 pool-only multi-WG split factor (grid.y); pool is ~32 kv so cheap


def _build_inverted_topk_fast(flat_kv, num_kv):
    """CSR inverted-topk index via sort+searchsorted: one stable sort yields the permutation
    (inv_data) and the sorted keys, inv_ptr[k] = searchsorted(sorted_vals, k, 'left'). Narrows
    the sort key to int16 when num_kv fits (bit-identical, less data moved)."""
    if num_kv < 32768:
        keys = flat_kv.to(torch.int16)
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=torch.int16)
    else:
        keys = flat_kv
        ar = torch.arange(num_kv + 1, device=flat_kv.device, dtype=flat_kv.dtype)
    sorted_vals, inv_data = torch.sort(keys, stable=True)
    inv_ptr = torch.searchsorted(sorted_vals, ar).to(torch.int32)
    return inv_ptr, inv_data.to(torch.int32)


_DQ_CACHE: dict = {}
_PRO_CR0_RESIDENT_KV_ENV = "FLYDSL_V4_TRAIN_PRO_CR0_RESIDENT_KV"
_PRO_CR0_RESIDENT_G2L_OVERLAP_ENV = "FLYDSL_V4_TRAIN_PRO_CR0_RESIDENT_G2L_OVERLAP"
_PRO_CR0_PV_PREFETCH_ENV = "FLYDSL_V4_TRAIN_PRO_CR0_PV_PREFETCH"
_PRO_CR4_RESIDENT_QDO_G2L_ENV = "FLYDSL_V4_TRAIN_PRO_CR4_RESIDENT_QDO_G2L"
_PRO_CR4_DQ_OPAQUE_G2L_OVERLAP_ENV = "FLYDSL_V4_TRAIN_PRO_CR4_DQ_OPAQUE_G2L_OVERLAP"
_PRO_CR4_DQ_TWOSTAGE_ENV = "FLYDSL_V4_TRAIN_PRO_CR4_DQ_TWOSTAGE"
_PRO_CR4_DQ_TWOSTAGE_QK_PREFETCH_ENV = "FLYDSL_V4_TRAIN_PRO_CR4_DQ_TWOSTAGE_QK_PREFETCH"
_PRO_CR4_INTERM_STACKED_K_ENV = "FLYDSL_V4_TRAIN_PRO_CR4_INTERM_STACKED_K"
_PRO_CR128_PV2X_ENV = "FLYDSL_V4_TRAIN_PRO_CR128_PV2X"
_PRO_CR128_PV2X_DB_ENV = "FLYDSL_V4_TRAIN_PRO_CR128_PV2X_DB"
_PRO_CR128_PV2X_PF_ENV = "FLYDSL_V4_TRAIN_PRO_CR128_PV2X_PF"
_PRO_CR128_PV2X_TAIL_ENV = "FLYDSL_V4_TRAIN_PRO_CR128_PV2X_TAIL"


def _get_dq(
    topk_len,
    scale,
    num_heads=None,
    delta_dot2_cr128=False,
    pro_cr128_pv2x=False,
    pro_cr128_pv2x_double_buf=1,
    pro_cr128_pv2x_prefetch=8,
    pro_cr128_pv2x_tail=False,
    resident_kv_cr0_h128=False,
    resident_g2l_overlap=False,
    resident_qdo_g2l_cr4=False,
    opaque_g2l_overlap_cr4=False,
    pro_cr4_twostage=False,
    pro_cr4_twostage_qk_prefetch=2,
    pro_cr0_pv_prefetch=8,
):
    nh = int(num_heads) if num_heads is not None else -1
    if bool(pro_cr128_pv2x) and not (int(topk_len) == 160 and nh == 128):
        raise ValueError("pro_cr128_pv2x requires exact effective R160/H128")
    if bool(pro_cr128_pv2x) and int(pro_cr128_pv2x_double_buf) not in {0, 1}:
        raise ValueError("pro_cr128_pv2x_double_buf must be 0 or 1")
    if bool(pro_cr128_pv2x) and int(pro_cr128_pv2x_prefetch) not in {4, 6, 8}:
        raise ValueError("pro_cr128_pv2x_prefetch must be 4, 6, or 8")
    if bool(pro_cr128_pv2x_tail) and not bool(pro_cr128_pv2x):
        raise ValueError("pro_cr128_pv2x_tail requires pro_cr128_pv2x")
    if bool(resident_qdo_g2l_cr4) and not (int(topk_len) == 1152 and nh == 128):
        raise ValueError("resident_qdo_g2l_cr4 requires exact R1152/H128")
    if bool(opaque_g2l_overlap_cr4) and not bool(resident_qdo_g2l_cr4):
        raise ValueError("opaque_g2l_overlap_cr4 requires resident_qdo_g2l_cr4")
    if bool(pro_cr4_twostage) and not (int(topk_len) == 1152 and nh == 128):
        raise ValueError("pro_cr4_twostage requires exact R1152/H128")
    if bool(pro_cr4_twostage) and int(pro_cr4_twostage_qk_prefetch) not in {2, 4, 6, 8}:
        raise ValueError("pro_cr4_twostage_qk_prefetch must be one of 2/4/6/8")
    resident_kv_cr0_h128 = bool(resident_kv_cr0_h128) and int(topk_len) == 128 and nh == 128
    resident_g2l_overlap = bool(resident_g2l_overlap) and resident_kv_cr0_h128
    delta_dot2_cr0 = int(topk_len) == 128 and nh == 128
    delta_dot2_cr128 = bool(delta_dot2_cr128) and nh == 128
    delta_dot2 = delta_dot2_cr0 or delta_dot2_cr128
    pro_cr128_pv2x = bool(pro_cr128_pv2x)
    pro_cr128_pv2x_double_buf = int(pro_cr128_pv2x_double_buf)
    pro_cr128_pv2x_prefetch = int(pro_cr128_pv2x_prefetch)
    pro_cr128_pv2x_tail = bool(pro_cr128_pv2x_tail)
    resident_qdo_g2l_cr4 = bool(resident_qdo_g2l_cr4)
    opaque_g2l_overlap_cr4 = bool(opaque_g2l_overlap_cr4)
    pro_cr4_twostage = bool(pro_cr4_twostage)
    pro_cr4_twostage_qk_prefetch = int(pro_cr4_twostage_qk_prefetch)
    pro_cr0_pv_prefetch = int(pro_cr0_pv_prefetch)
    if pro_cr0_pv_prefetch not in {2, 8, 10}:
        raise ValueError("pro_cr0_pv_prefetch must be 2, 8, or 10")
    base_key = (topk_len, float(scale), nh)
    if pro_cr4_twostage:
        key = (
            *base_key,
            "pro_cr4_r1152_h128_twostage",
            f"qkpf{pro_cr4_twostage_qk_prefetch}",
        )
    elif opaque_g2l_overlap_cr4:
        key = (*base_key, "pro_cr4_opaque_g2l_overlap")
    elif resident_qdo_g2l_cr4:
        key = (*base_key, "pro_cr4_resident_qdo_g2l")
    elif pro_cr128_pv2x:
        key = (
            *base_key,
            "pro_cr128_r160_pv2x",
            f"db{pro_cr128_pv2x_double_buf}",
            f"pf{pro_cr128_pv2x_prefetch}",
            f"tail{int(pro_cr128_pv2x_tail)}",
        )
    elif resident_g2l_overlap:
        key = (
            *base_key,
            "resident_kv_cr0_h128_r128_g2l_overlap",
            f"pvpf{pro_cr0_pv_prefetch}",
        )
    elif resident_kv_cr0_h128:
        key = (
            *base_key,
            "resident_kv_cr0_h128_r128",
            f"pvpf{pro_cr0_pv_prefetch}",
        )
    elif delta_dot2_cr0:
        key = (*base_key, "delta_dot2")
    elif delta_dot2_cr128:
        key = (*base_key, "delta_dot2_cr128")
    else:
        key = base_key
    fn = _DQ_CACHE.get(key)
    if fn is None:
        # PV-K32: per-16 QK/softmax kept, PV batches 2 tiles into mfma_16x16x32 with direct-v8
        # operands. Both dQ kernels pin dq_acc in AGPR via inline-asm. Threshold 384 (not 512):
        # build_bwd_dq races at TOPK=384; the race-free pv2x path handles any even tile count.
        # Both variants fold delta from O*dO before their register-heavy tile loop.
        if pro_cr4_twostage:
            fn = build_bwd_dq_twostage(
                topk_len,
                float(scale),
                num_heads=num_heads,
                qk_prefetch=pro_cr4_twostage_qk_prefetch,
                pv_prefetch=8,
            )
        elif resident_kv_cr0_h128:
            fn = build_bwd_dq_resident_kv_cr0_h128(
                float(scale),
                pv_prefetch=pro_cr0_pv_prefetch,
                prologue_g2l_overlap=resident_g2l_overlap,
            )
        elif pro_cr128_pv2x:
            fn = build_bwd_dq_pv2x(
                topk_len,
                float(scale),
                num_heads=num_heads,
                qk_prefetch=4,
                pv_prefetch=pro_cr128_pv2x_prefetch,
                double_buf=pro_cr128_pv2x_double_buf,
                tail_mask=pro_cr128_pv2x_tail,
            )
        elif topk_len >= 384 and topk_len % 32 == 0:
            # cross-pair double-buffer (1 barrier/pair) + QK prefetch depth 4.
            fn = build_bwd_dq_pv2x(
                topk_len,
                float(scale),
                num_heads=num_heads,
                qk_prefetch=4,
                double_buf=1,
                resident_qdo_g2l=resident_qdo_g2l_cr4,
                opaque_g2l_overlap=opaque_g2l_overlap_cr4,
            )
        else:
            fn = build_bwd_dq(
                topk_len,
                float(scale),
                num_heads=num_heads,
                delta_dot2=delta_dot2,
            )
        _DQ_CACHE[key] = fn
    return fn


_FLASH_CR4_FUSED_KB_ENV = "FLYDSL_V4_TRAIN_FLASH_CR4_FUSED_KB"
_FLASH_CR4_FUSED_PV2X_ENV = "FLYDSL_V4_TRAIN_FLASH_CR4_FUSED_PV2X"


def _get_fused(
    topk_len,
    scale,
    num_heads,
    flash_cr4_kb=0,
    flash_cr4_pv2x=False,
):
    flash_cr4_kb = (
        int(flash_cr4_kb) if int(topk_len) == 640 and int(num_heads) == 64 and int(flash_cr4_kb) in {5, 10, 20} else 0
    )
    flash_cr4_pv2x = bool(flash_cr4_pv2x) and int(topk_len) == 640 and int(num_heads) == 64 and flash_cr4_kb == 10
    base_key = ("fused", topk_len, float(scale), int(num_heads))
    if flash_cr4_pv2x:
        key = (*base_key, "flash_cr4_kb10_pv2x")
    elif flash_cr4_kb:
        key = (*base_key, f"flash_cr4_kb{flash_cr4_kb}")
    else:
        key = base_key
    fn = _DQ_CACHE.get(key)
    if fn is None:
        fn = build_bwd_fused(
            topk_len,
            float(scale),
            int(num_heads),
            flash_cr4_kb=flash_cr4_kb,
            flash_cr4_pv2x=flash_cr4_pv2x,
        )
        _DQ_CACHE[key] = fn
    return fn


_DELTA_CACHE: dict = {}
_DSINK_SPLIT_CACHE: dict = {}


def _get_delta():
    fn = _DELTA_CACHE.get("fn")
    if fn is None:
        fn = build_delta()
        _DELTA_CACHE["fn"] = fn
    return fn


_GATHER_CACHE: dict = {}


def _get_gather():
    fn = _GATHER_CACHE.get("fn")
    if fn is None:
        # num_waves=16: the HCA cr128 pool-entry-0 CSR list is ~4000 long and one WG per kv walked it
        # serially; 16 waves split the list. cr0/cr4 (balanced/BW-bound) neutral.
        fn = build_gather(16)
        _GATHER_CACHE["fn"] = fn
    return fn


def _get_gather_banded(row_chunk=None):
    # row_chunk = interm row stride (R_CHUNK). None -> W=128 (cr0). cr128 local band uses row_chunk=192
    # (the padded topk width) since the local ranks 0..127 live in a wider interm row.
    key = ("banded", row_chunk)
    fn = _GATHER_CACHE.get(key)
    if fn is None:
        fn = build_gather_banded(8, row_chunk=row_chunk)
        _GATHER_CACHE[key] = fn
    return fn


def _get_gather_pool():
    fn = _GATHER_CACHE.get("pool")
    if fn is None:
        fn = build_gather_pool(16, _GATHER_NS)
        _GATHER_CACHE["pool"] = fn
    return fn


def _get_partial_reduce():
    fn = _GATHER_CACHE.get("partred")
    if fn is None:
        fn = build_partial_reduce(_GATHER_NS)
        _GATHER_CACHE["partred"] = fn
    return fn


_INTERM_CACHE: dict = {}


def _get_interm(
    topk_len,
    num_heads,
    stacked_k_pro_cr4=False,
):
    # regtr (register-transpose Q/dO, no Q/dO LDS -> BD256) for pro (num_heads>64); flash (H<=64)
    # uses blocked (tuned wide BD already minimizes re-read; regtr's transpose overhead not amortized
    # at H=64).
    use_regtr = int(num_heads) > 64
    mode = "regtr" if use_regtr else "blocked"
    if bool(stacked_k_pro_cr4) and not (mode == "regtr" and int(topk_len) == 1152 and int(num_heads) == 128):
        raise ValueError("stacked_k_pro_cr4 requires exact R1152/H128")
    stacked_k_pro_cr4 = bool(stacked_k_pro_cr4)
    pro_cr0_gsz64_dbuf2 = mode == "regtr" and int(topk_len) == 128 and int(num_heads) == 128
    base_key = (topk_len, num_heads, mode)
    if stacked_k_pro_cr4:
        key = (*base_key, "pro_cr4_stacked_k")
    elif pro_cr0_gsz64_dbuf2:
        key = (*base_key, "gsz64_dbuf2")
    else:
        key = base_key
    fn = _INTERM_CACHE.get(key)
    if fn is None:
        if mode == "regtr":
            # interm stays baseline (functional rocdl.mfma): inline-asm in-place MFMA breaks det
            # on interm's per-output fresh accumulator (the AGPR win only holds for dQ's
            # loop-carried acc).
            fn = build_interm_regtr(
                topk_len,
                num_heads,
                256,
                stacked_k=stacked_k_pro_cr4,
            )
        else:
            # BD is R_CHUNK-dependent (interm is latency-bound). flash (H<=64): wider BD reduces
            # the d-block count -> less dS/P HBM re-read; pro (H=128): BD128 (BD256=176KB > cap).
            if num_heads <= 64:
                # flash: cr4 (R=640) prefers BD512; small-R cr0/cr128 prefer BD256 (BD512
                # under-utilizes at 1WG). interm is latency-bound so ILP > occupancy.
                bd = 512 if topk_len >= 640 else 256
            else:
                # pro (H=128) BD128 beats BD64 at every R: the read-batching makes BD128's fewer
                # d-blocks (less dS/P re-read) dominate BD64's 2WG occupancy. BD256 = 176KB > cap.
                bd = 128
            fn = build_interm_blocked(topk_len, num_heads, bd)
        _INTERM_CACHE[key] = fn
    return fn


def sparse_mla_bwd_flydsl(
    q,
    kv,
    o,
    do,
    topk_indices,
    lse,
    attn_sink=None,
    kv_lora_rank=512,
    scale=None,
    *,
    canonical_topk=False,
):
    """Run DSV4 sparse-MLA training backward on gfx950.

    The input contract matches :func:`sparse_mla_fwd_flydsl`; ``o`` and
    ``do`` are BF16 ``[T,H,512]`` and ``lse`` is FP32 ``[T,H]``. Set
    ``canonical_topk=True`` only for canonical DSV4 SWA/cr128 indices.
    """
    if not torch.cuda.is_available() or q.device.type != "cuda":
        raise RuntimeError("DSV4 sparse-MLA training requires a ROCm GPU")
    arch = torch.cuda.get_device_properties(q.device).gcnArchName.split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(f"DSV4 sparse-MLA training requires gfx950, got {arch}")
    if q.device.index != torch.cuda.current_device():
        raise ValueError("q must be on the current CUDA device; set the process device before calling")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError("q and kv must use torch.bfloat16")
    if o.dtype != torch.bfloat16 or do.dtype != torch.bfloat16:
        raise TypeError("o and do must use torch.bfloat16")
    if topk_indices.dtype != torch.int32 or lse.dtype != torch.float32:
        raise TypeError("topk_indices must be INT32 and lse must be FP32")
    tensors = (q, kv, o, do, topk_indices, lse)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all backward inputs must be contiguous")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all backward inputs must be on the same device")
    if q.ndim != 3 or kv.ndim not in (2, 3) or topk_indices.ndim != 2:
        raise ValueError("expected q[T,H,576], kv[NKV,1,576], and topk_indices[T,R]")
    total_tokens, num_heads, d_qk = q.shape
    D = kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (kv_lora_rank**0.5)
    kv_was_2d = kv.dim() == 2
    if kv_was_2d:
        kv = kv.unsqueeze(1)
    num_kv = kv.shape[0]
    if D != 512 or d_qk != 576 or kv.shape[1:] != (1, 576):
        raise ValueError(f"expected D=512 and q/kv d_qk=576, got D={D}, q={d_qk}, kv={kv.shape}")
    if num_heads not in (64, 128):
        raise ValueError(f"num_heads must be 64 (Flash) or 128 (Pro), got {num_heads}")
    if topk_indices.shape[0] != total_tokens or topk % 32:
        raise ValueError("topk must match q tokens and be padded to a multiple of 32")
    if o.shape != (total_tokens, num_heads, D) or do.shape != o.shape:
        raise ValueError(f"o and do must have shape {(total_tokens, num_heads, D)}")
    if lse.shape != (total_tokens, num_heads):
        raise ValueError(f"lse must have shape {(total_tokens, num_heads)}")

    # cr128 padding drop: the caller pads topk to a multiple of 64; the pad ranks are always -1
    # (contribute 0 dS/P), so slicing to the real 128+npool width shrinks the dQ/interm work
    # with no behavior change. Guarded to the deterministic HCA shape; cr0/cr4 untouched.
    _npool = num_kv - total_tokens
    _topk_real = 128 + _npool
    # Record the exact padded Pro-cr128 release case before R192 is sliced to effective R160.
    _pro_cr128_delta_dot2 = (
        int(topk) == 192
        and int(num_heads) == 128
        and _npool > 0
        and total_tokens % _npool == 0
        and (total_tokens // _npool) >= 64
    )
    _pro_cr128_pv2x = False
    _pro_cr128_pv2x_double_buf = 1
    _pro_cr128_pv2x_prefetch = 8
    _pro_cr128_pv2x_tail = False
    _exact_pro_cr128 = int(total_tokens) == 4096 and int(topk) == 192 and int(num_heads) == 128 and int(num_kv) == 4128
    if canonical_topk and _exact_pro_cr128:
        _pro_cr128_pv2x_setting = (
            os.getenv(
                _PRO_CR128_PV2X_ENV,
                "",
            )
            .strip()
            .lower()
        )
        if _pro_cr128_pv2x_setting in {"", "0", "false", "no", "off"}:
            _pro_cr128_pv2x = False
        elif _pro_cr128_pv2x_setting in {"1", "true", "yes", "on"}:
            _pro_cr128_pv2x = True
        else:
            raise ValueError(f"{_PRO_CR128_PV2X_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
        if _pro_cr128_pv2x:
            try:
                _pro_cr128_pv2x_double_buf = int(os.getenv(_PRO_CR128_PV2X_DB_ENV, "1").strip())
            except ValueError as exc:
                raise ValueError(f"{_PRO_CR128_PV2X_DB_ENV} must be 0 or 1") from exc
            if _pro_cr128_pv2x_double_buf not in {0, 1}:
                raise ValueError(f"{_PRO_CR128_PV2X_DB_ENV} must be 0 or 1")
            try:
                _pro_cr128_pv2x_prefetch = int(os.getenv(_PRO_CR128_PV2X_PF_ENV, "8").strip())
            except ValueError as exc:
                raise ValueError(f"{_PRO_CR128_PV2X_PF_ENV} must be 4, 6, or 8") from exc
            if _pro_cr128_pv2x_prefetch not in {4, 6, 8}:
                raise ValueError(f"{_PRO_CR128_PV2X_PF_ENV} must be 4, 6, or 8")
            _pro_cr128_pv2x_tail_setting = (
                os.getenv(
                    _PRO_CR128_PV2X_TAIL_ENV,
                    "",
                )
                .strip()
                .lower()
            )
            if _pro_cr128_pv2x_tail_setting in {
                "",
                "0",
                "false",
                "no",
                "off",
            }:
                _pro_cr128_pv2x_tail = False
            elif _pro_cr128_pv2x_tail_setting in {
                "1",
                "true",
                "yes",
                "on",
            }:
                _pro_cr128_pv2x_tail = True
            else:
                raise ValueError(f"{_PRO_CR128_PV2X_TAIL_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
    if canonical_topk and 0 < _npool and topk <= 256 and topk > _topk_real and _topk_real % 32 == 0:
        topk = _topk_real
        topk_indices = topk_indices[:, :topk].contiguous()

    if attn_sink is None:
        raise ValueError("attn_sink is required by the DSV4 training contract")
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (num_heads,):
        raise TypeError(f"attn_sink must be contiguous FP32[{num_heads}]")
    if attn_sink.device != q.device or not attn_sink.is_contiguous():
        raise ValueError("attn_sink must be contiguous and on the q device")
    has_sink = True

    # The CSR build is kept sequential (overlapping it on a side stream contends with the
    # compute kernels; the memory-bound sort + stream setup dominates the short flash path).

    # delta = rowsum(O*dO): every dQ producer (fused, regular, and pv2x) computes it inline
    # while dO is already resident, then writes [T,H] delta for the optional d_sink reduction.
    # This removes the standalone delta launch and its redundant full dO read.
    assert o.shape[-1] == D and o.is_contiguous()
    # All dQ variants consume the contiguous lora columns.
    do_lora = do[..., :D].contiguous()  # [T, H, 512]
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)  # [T, H]
    _flash_cr4_fused_kb = 0
    _flash_cr4_fused_pv2x = False
    _resident_qdo_g2l_cr4 = False
    _opaque_g2l_overlap_cr4 = False
    _pro_cr4_twostage = False
    _pro_cr4_twostage_qk_prefetch = 2
    _interm_stacked_k_pro_cr4 = False
    _exact_flash_cr4 = int(total_tokens) == 4096 and int(topk) == 640 and int(num_heads) == 64 and int(num_kv) == 5120
    if _exact_flash_cr4:
        _flash_cr4_fused_kb_setting = os.getenv(
            _FLASH_CR4_FUSED_KB_ENV,
            "",
        ).strip()
        try:
            # KB10 is the largest spill-free exact-shape fusion batch.  Keep
            # explicit 0 as a strict rollback/control for release A/B checks.
            _flash_cr4_fused_kb = int(_flash_cr4_fused_kb_setting or "10")
        except ValueError as exc:
            raise ValueError(f"{_FLASH_CR4_FUSED_KB_ENV} must be one of 0/5/10/20") from exc
        if _flash_cr4_fused_kb not in {0, 5, 10, 20}:
            raise ValueError(f"{_FLASH_CR4_FUSED_KB_ENV} must be one of 0/5/10/20")
        _flash_cr4_fused_pv2x_setting = (
            os.getenv(
                _FLASH_CR4_FUSED_PV2X_ENV,
                "",
            )
            .strip()
            .lower()
        )
        # Paired K32 PV is the validated exact-shape default.  Keep explicit
        # false values as a strict generic-KB10 rollback/control.
        if _flash_cr4_fused_pv2x_setting in {"", "1", "true", "yes", "on"}:
            _flash_cr4_fused_pv2x = True
        elif _flash_cr4_fused_pv2x_setting in {"0", "false", "no", "off"}:
            _flash_cr4_fused_pv2x = False
        else:
            raise ValueError(f"{_FLASH_CR4_FUSED_PV2X_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
        if _flash_cr4_fused_pv2x and _flash_cr4_fused_kb != 10:
            raise ValueError(f"{_FLASH_CR4_FUSED_PV2X_ENV}=1 requires " f"{_FLASH_CR4_FUSED_KB_ENV}=10")
    _exact_pro_cr4 = int(total_tokens) == 4096 and int(topk) == 1152 and int(num_heads) == 128 and int(num_kv) == 5120
    if _exact_pro_cr4:
        _pro_cr4_twostage_setting = (
            os.getenv(
                _PRO_CR4_DQ_TWOSTAGE_ENV,
                "",
            )
            .strip()
            .lower()
        )
        # The exact H128 split is the profiled-positive default. Explicit
        # false values retain the accepted 7a512 dQ route as a strict control.
        if _pro_cr4_twostage_setting in {"", "1", "true", "yes", "on"}:
            _pro_cr4_twostage = True
        elif _pro_cr4_twostage_setting in {"0", "false", "no", "off"}:
            _pro_cr4_twostage = False
        else:
            raise ValueError(f"{_PRO_CR4_DQ_TWOSTAGE_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
        try:
            _pro_cr4_twostage_qk_prefetch = int(
                os.getenv(
                    _PRO_CR4_DQ_TWOSTAGE_QK_PREFETCH_ENV,
                    "2",
                ).strip()
            )
        except ValueError as exc:
            raise ValueError(f"{_PRO_CR4_DQ_TWOSTAGE_QK_PREFETCH_ENV} " "must be one of 2/4/6/8") from exc
        if _pro_cr4_twostage_qk_prefetch not in {2, 4, 6, 8}:
            raise ValueError(f"{_PRO_CR4_DQ_TWOSTAGE_QK_PREFETCH_ENV} " "must be one of 2/4/6/8")
        _resident_qdo_g2l_setting = (
            os.getenv(
                _PRO_CR4_RESIDENT_QDO_G2L_ENV,
                "",
            )
            .strip()
            .lower()
        )
        # Owner-8 direct-G2L is the validated exact-shape default. Explicit
        # false values retain the prior gather-to-register route for A/B.
        if _resident_qdo_g2l_setting in {"", "1", "true", "yes", "on"}:
            _resident_qdo_g2l_cr4 = True
        elif _resident_qdo_g2l_setting in {"0", "false", "no", "off"}:
            _resident_qdo_g2l_cr4 = False
        else:
            raise ValueError(f"{_PRO_CR4_RESIDENT_QDO_G2L_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
        _opaque_g2l_overlap_setting = (
            os.getenv(
                _PRO_CR4_DQ_OPAQUE_G2L_OVERLAP_ENV,
                "",
            )
            .strip()
            .lower()
        )
        # The structurally overlapped schedule is the integrated default only
        # while the exact accepted owner-8 resident route is active. Explicit
        # false values select the accepted typed-G2L control. If the resident
        # route itself is disabled, an absent opaque selector stays disabled.
        if _opaque_g2l_overlap_setting == "":
            _opaque_g2l_overlap_cr4 = _resident_qdo_g2l_cr4
        elif _opaque_g2l_overlap_setting in {
            "0",
            "false",
            "no",
            "off",
        }:
            _opaque_g2l_overlap_cr4 = False
        elif _opaque_g2l_overlap_setting in {
            "1",
            "true",
            "yes",
            "on",
        }:
            _opaque_g2l_overlap_cr4 = True
        else:
            raise ValueError(f"{_PRO_CR4_DQ_OPAQUE_G2L_OVERLAP_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
        if _opaque_g2l_overlap_cr4 and not _resident_qdo_g2l_cr4:
            raise ValueError(f"{_PRO_CR4_DQ_OPAQUE_G2L_OVERLAP_ENV}=1 requires " f"{_PRO_CR4_RESIDENT_QDO_G2L_ENV}=1")
        _interm_stacked_k_setting = (
            os.getenv(
                _PRO_CR4_INTERM_STACKED_K_ENV,
                "",
            )
            .strip()
            .lower()
        )
        if _interm_stacked_k_setting in {"", "0", "false", "no", "off"}:
            _interm_stacked_k_pro_cr4 = False
        elif _interm_stacked_k_setting in {"1", "true", "yes", "on"}:
            _interm_stacked_k_pro_cr4 = True
        else:
            raise ValueError(f"{_PRO_CR4_INTERM_STACKED_K_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
    use_fused = (int(num_heads) <= 64 and int(topk) <= 256) or bool(_flash_cr4_fused_kb)
    # The dQ/fused kernels write ALL D_V=512 lora cols for every (token,head) AND
    # zero the 64 rope cols (512..575) in-kernel (matches triton's zeroed rope),
    # folding away the strided host dq[..., D:].zero_().
    dq = torch.empty_like(q)
    stream = torch.cuda.current_stream(q.device)
    # The validated resident-R128 one-workgroup dQ is the exact Pro-cr0 default.
    # Explicit false values retain the accepted two-workgroup path as a strict A/B
    # control; parse the setting only for this shape so every other dispatch is unchanged.
    _exact_pro_cr0_h128 = (
        bool(canonical_topk) and int(topk) == 128 and int(num_heads) == 128 and int(num_kv) == int(total_tokens)
    )
    _resident_kv_setting = os.getenv(_PRO_CR0_RESIDENT_KV_ENV, "").strip().lower()
    if _exact_pro_cr0_h128:
        if _resident_kv_setting in {"", "1", "true", "yes", "on"}:
            _resident_kv_cr0_h128 = True
        elif _resident_kv_setting in {"0", "false", "no", "off"}:
            _resident_kv_cr0_h128 = False
        else:
            raise ValueError(f"{_PRO_CR0_RESIDENT_KV_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
    else:
        _resident_kv_cr0_h128 = False
    # The direct-to-LDS prologue is the validated Pro-cr0 resident default.
    # Explicit false values retain the pre-overlap resident kernel as a strict
    # performance/control rollback without changing any other dispatch.
    _resident_g2l_overlap_setting = (
        os.getenv(
            _PRO_CR0_RESIDENT_G2L_OVERLAP_ENV,
            "",
        )
        .strip()
        .lower()
    )
    if _resident_kv_cr0_h128:
        if _resident_g2l_overlap_setting in {"", "1", "true", "yes", "on"}:
            _resident_g2l_overlap = True
        elif _resident_g2l_overlap_setting in {"0", "false", "no", "off"}:
            _resident_g2l_overlap = False
        else:
            raise ValueError(f"{_PRO_CR0_RESIDENT_G2L_OVERLAP_ENV} must be one of " "1/true/yes/on or 0/false/no/off")
    else:
        _resident_g2l_overlap = False
    _pro_cr0_pv_prefetch = 8
    if _resident_kv_cr0_h128:
        try:
            # PF10 is the integrated exact-resident default. PF8 is the
            # accepted control and PF2 remains available only for audit.
            _pro_cr0_pv_prefetch = int(os.getenv(_PRO_CR0_PV_PREFETCH_ENV, "10").strip())
        except ValueError as exc:
            raise ValueError(f"{_PRO_CR0_PV_PREFETCH_ENV} must be 2, 8, or 10") from exc
        if _pro_cr0_pv_prefetch not in {2, 8, 10}:
            raise ValueError(f"{_PRO_CR0_PV_PREFETCH_ENV} must be 2, 8, or 10")
    # FUSED dQ+interm (flash small-topk only): dS/P handed through LDS in one kernel (no
    # chunk_dS/P HBM), producing dq + interm together. Wins for small topk (occ-2
    # kv-block-batching); pro (H=128) never fuses (8 waves -> register spill, occ-2 lost).
    if not use_fused:
        chunk_dS = torch.empty(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        chunk_P = torch.empty(total_tokens, num_heads, topk, dtype=torch.bfloat16, device=q.device)
        # ---- flydsl dQ (also produces dS / P for the dKV-interm kernel) ----
        fn = _get_dq(
            topk,
            scale,
            num_heads,
            delta_dot2_cr128=_pro_cr128_delta_dot2,
            pro_cr128_pv2x=_pro_cr128_pv2x,
            pro_cr128_pv2x_double_buf=_pro_cr128_pv2x_double_buf,
            pro_cr128_pv2x_prefetch=_pro_cr128_pv2x_prefetch,
            pro_cr128_pv2x_tail=_pro_cr128_pv2x_tail,
            resident_kv_cr0_h128=_resident_kv_cr0_h128,
            resident_g2l_overlap=_resident_g2l_overlap,
            resident_qdo_g2l_cr4=_resident_qdo_g2l_cr4,
            opaque_g2l_overlap_cr4=_opaque_g2l_overlap_cr4,
            pro_cr4_twostage=_pro_cr4_twostage,
            pro_cr4_twostage_qk_prefetch=(_pro_cr4_twostage_qk_prefetch),
            pro_cr0_pv_prefetch=_pro_cr0_pv_prefetch,
        )
        args = (
            q,
            kv,
            do_lora,
            topk_indices,
            lse,
            delta,
            dq,
            chunk_dS,
            chunk_P,
            o,
            int(total_tokens),
            int(num_heads),
            int(num_kv),
            stream,
        )
        _dq_key = ("c", int(topk), float(scale), int(num_heads))
        if _pro_cr4_twostage:
            _dq_key = (
                *_dq_key,
                "pro_cr4_r1152_h128_twostage",
                f"qkpf{_pro_cr4_twostage_qk_prefetch}",
            )
        elif _opaque_g2l_overlap_cr4:
            _dq_key = (
                *_dq_key,
                "pro_cr4_opaque_g2l_overlap",
            )
        elif _resident_qdo_g2l_cr4:
            _dq_key = (
                *_dq_key,
                "pro_cr4_resident_qdo_g2l",
            )
        elif _pro_cr128_pv2x:
            _dq_key = (
                *_dq_key,
                "pro_cr128_r160_pv2x",
                f"db{_pro_cr128_pv2x_double_buf}",
                f"pf{_pro_cr128_pv2x_prefetch}",
                f"tail{int(_pro_cr128_pv2x_tail)}",
            )
        elif _resident_g2l_overlap:
            _dq_key = (
                *_dq_key,
                "resident_kv_cr0_h128_r128_g2l_overlap",
                f"pvpf{_pro_cr0_pv_prefetch}",
            )
        elif _resident_kv_cr0_h128:
            _dq_key = (
                *_dq_key,
                "resident_kv_cr0_h128_r128",
                f"pvpf{_pro_cr0_pv_prefetch}",
            )
        elif int(topk) == 128 and int(num_heads) == 128:
            _dq_key = (*_dq_key, "delta_dot2")
        elif _pro_cr128_delta_dot2:
            _dq_key = (*_dq_key, "delta_dot2_cr128")
        _cached_run(_DQ_CACHE, _dq_key, fn, args)

    # interm only ever writes/reads the D_V=512 lora cols (rope cols are dead) -> allocate it
    # D_V-wide (not d_qk=576): cuts the dominant dKV-interm HBM write AND the gather read.
    R_CHUNK = topk
    interm = torch.empty(total_tokens, R_CHUNK, D, dtype=torch.bfloat16, device=q.device)
    # cr128 has two short final writers (band + pool reduction); extending both with tail
    # stores costs more than its highly optimized fill. The longer cr0/cr4 gathers amortize
    # those stores and safely remove the redundant full [NKV,576] zero pass.
    is_cr0 = bool(canonical_topk) and (num_kv == total_tokens) and (topk == 128)
    _npool_bwd = num_kv - total_tokens
    is_cr128 = (
        bool(canonical_topk)
        and _npool_bwd > 0
        and topk <= 256
        and total_tokens % _npool_bwd == 0
        and (total_tokens // _npool_bwd) >= 64
    )
    if is_cr128:
        dkv = torch.zeros(num_kv, d_qk, dtype=kv.dtype, device=q.device)
    else:
        dkv = torch.empty(num_kv, d_qk, dtype=kv.dtype, device=q.device)

    # ---- flydsl dKV-interm: register-transpose (regtr) for pro, 2D-blocked (blocked) for flash.
    if use_fused:
        # one kernel produced dq + interm already (dS/P via LDS, no chunk_dS/P HBM).
        ffn = _get_fused(
            int(topk),
            float(scale),
            int(num_heads),
            flash_cr4_kb=_flash_cr4_fused_kb,
            flash_cr4_pv2x=_flash_cr4_fused_pv2x,
        )
        fargs = (
            q,
            kv,
            do_lora,
            topk_indices,
            lse,
            delta,
            dq,
            interm,
            o,
            int(total_tokens),
            int(num_heads),
            int(num_kv),
            stream,
        )
        _fused_key = ("fc", int(topk), float(scale), int(num_heads))
        if _flash_cr4_fused_pv2x:
            _fused_key = (
                *_fused_key,
                "flash_cr4_kb10_pv2x",
            )
        elif _flash_cr4_fused_kb:
            _fused_key = (
                *_fused_key,
                f"flash_cr4_kb{_flash_cr4_fused_kb}",
            )
        _cached_run(_DQ_CACHE, _fused_key, ffn, fargs)
    else:
        ifn = _get_interm(
            int(topk),
            int(num_heads),
            stacked_k_pro_cr4=_interm_stacked_k_pro_cr4,
        )
        iargs = (q, do_lora, chunk_dS, chunk_P, interm, int(total_tokens), stream)
        _interm_key = ("c", int(topk), int(num_heads))
        if _interm_stacked_k_pro_cr4:
            _interm_key = (*_interm_key, "pro_cr4_stacked_k")
        elif int(topk) == 128 and int(num_heads) == 128:
            _interm_key = (*_interm_key, "gsz64_dbuf2")
        _cached_run(_INTERM_CACHE, _interm_key, ifn, iargs)

    # cr=0 (pure SWA): num_kv==T and topk==W=128 -> use the closed-form banded gather
    # (skips the CSR argsort/bincount/cumsum + InvPtr/InvData buffers entirely; the
    # inverse window map is closed-form). Bit-exact vs the CSR path.
    # FIX (dsv4 cr=4 small-seq dkv misdispatch): the cr=128 closed-form pool gather assumes a
    # deterministic causal pool. A bare `topk<=256` also catches cr=4 (random pool) at small seq
    # (S<=512 -> topk<=256) -> wrong dkv/dpool. Guard with the deterministic-pool condition the
    # forward uses so cr=4 (pool_cr=4) falls through to the CSR gather. Production S=4096 already CSR.
    if is_cr0:
        _gargs = (interm.reshape(-1, D), dkv, int(num_kv), int(total_tokens), stream)
        _cached_run(_GATHER_CACHE, "cb", _get_gather_banded(), _gargs)
    elif is_cr128:
        # cr=128 (HCA): both the local SWA band and the pool are deterministic, so both invert
        # closed-form. (1) banded-local gather (row_chunk=R_CHUNK) for kv 0..T-1. (2) closed-form pool
        # gather for kv>=T. Local and pool kv write disjoint dkv rows -> no conflict.
        _bargs = (interm.reshape(-1, D), dkv, int(num_kv), int(total_tokens), stream)
        _bkey = ("cb128", int(R_CHUNK))
        _cached_run(_GATHER_CACHE, _bkey, _get_gather_banded(row_chunk=int(R_CHUNK)), _bargs)

        # The REAL pool occupies ranks 128..128+P-1 (P = npool); ranks 128+P..R_CHUNK are
        # -1 padding (topk padded to a multiple of 64).
        npool = int(num_kv - total_tokens)
        partial = torch.empty(npool, _GATHER_NS, D, dtype=torch.float32, device=q.device)
        # Closed-form pool: token i attends pool block b (kv T+b) iff i >= (b+1)*cr_pool - 1;
        # visibility is monotone in b so column b == block b, giving a closed-form inverse.
        # Production shapes always have clean pool blocks (total_tokens % npool == 0).
        assert total_tokens % npool == 0, "cr128 pool blocks must be clean"
        cr_pool = total_tokens // npool
        _pargs = (
            interm.reshape(-1, D),
            partial,
            int(num_kv),
            int(total_tokens * R_CHUNK),
            int(total_tokens),
            int(npool),
            int(cr_pool),
            int(R_CHUNK),
            128,
            stream,
        )
        _cached_run(_GATHER_CACHE, "cpool", _get_gather_pool(), _pargs)
        _rargs = (partial.reshape(-1, D), dkv, int(num_kv), int(total_tokens), int(npool), stream)
        _cached_run(_GATHER_CACHE, "cr", _get_partial_reduce(), _rargs)
    else:
        # cr4: full CSR single-WG gather. CSR inverted-topk scatter of interm -> dkv[:, :D]
        # (bf16 cast in-kernel), rope cols untouched. The NS-split loses here: cr4's long pool
        # lists already saturate all CUs (BW-bound), so it would only add partial traffic.
        inv_ptr, inv_data = _build_inverted_topk_fast(topk_indices.reshape(-1), num_kv)
        _gargs = (
            interm.reshape(-1, D),
            inv_ptr.contiguous(),
            inv_data.contiguous(),
            dkv,
            int(num_kv),
            int(total_tokens * R_CHUNK),
            stream,
        )
        _cached_run(_GATHER_CACHE, "c", _get_gather(), _gargs)

    d_sink = None
    if has_sink:
        # flydsl d_sink[h] = -sum_t exp(sink[h]-lse[t,h])*delta[t,h], 2-pass coalesced: pass 1
        # reads full rows contiguously into per-block partials[nblk,H], pass 2 reduces the nblk
        # blocks -> d_sink. fp32.
        d_sink = torch.empty(num_heads, dtype=torch.float32, device=q.device)
        _nblk = (int(total_tokens) + DSINK_TB - 1) // DSINK_TB
        _dspart = torch.empty(_nblk, int(num_heads), dtype=torch.float32, device=q.device)
        _p1args = (
            lse.reshape(-1),
            delta.reshape(-1),
            attn_sink,
            _dspart.reshape(-1),
            int(total_tokens),
            int(_nblk),
            stream,
        )
        _p2args = (_dspart.reshape(-1), d_sink, stream)
        _dskey = (int(total_tokens), int(num_heads))
        _dsc = _DSINK_SPLIT_CACHE.get(_dskey)
        if _dsc is None:
            _p1 = build_dsink_split(int(total_tokens), int(num_heads), DSINK_TB).compile(*_p1args)
            _p2 = build_dsink_reduce(int(_nblk), int(num_heads)).compile(*_p2args)
            _dsc = (_p1, _p2)
            _DSINK_SPLIT_CACHE[_dskey] = _dsc
        _dsc[0](*_p1args)
        _dsc[1](*_p2args)

    if not kv_was_2d:
        dkv = dkv.unsqueeze(1)
    return dq, dkv, d_sink
