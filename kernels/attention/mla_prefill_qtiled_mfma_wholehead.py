# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Four-wave gfx950 absorbed-MLA prefill using one opaque fixed phase engine."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, gpu
from kernels.attention.mla_prefill_qtiled_mfma_intrinsics import (
    whole_loop_fixed_clobber,
)

BLOCK_THREADS = 256
SHORT_PERSISTENT_GRID = 256
SHORT_PERSISTENT_MAX_Q = 1024


@fx.struct
class WholeHeadShared:
    # stage0 [0,18560), stage1 [18560,37120), VT [37120,53504),
    # max planes [53504,54528) and [54528,55552); reserve the reference class.
    storage: fx.Array[fx.Int8, 65536, 16]


def build_mla_prefill_mfma_wholehead_module(is_causal: bool = True):
    if not bool(is_causal):
        raise ValueError("the fixed phase engine implements causal prefill only")

    value_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": "256,256",
        "passthrough": [
            ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
            ["no-nans-fp-math", "true"],
            ["unsafe-fp-math", "true"],
        ],
    }
    compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
        "llvm_options": {
            "enable-post-misched": False,
            "lsr-drop-solution": True,
        },
    }

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def wholehead_kernel(
        query2d: fx.Tensor,
        kv_flat: fx.Tensor,
        kv_page_indices: fx.Tensor,
        kv_base: fx.Int32,
        out2d: fx.Tensor,
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
    ):
        lds = fx.SharedAllocator().allocate(WholeHeadShared).peek()
        whole_loop_fixed_clobber(
            fx.thread_idx.x,
            fx.block_idx.x,
            buffer_ops.create_buffer_resource(out2d),
            buffer_ops.create_buffer_resource(query2d),
            buffer_ops.create_buffer_resource(kv_flat),
            buffer_ops.create_buffer_resource(kv_page_indices),
            sm_scale,
            seq_len,
            kv_base,
            fx.ptrtoint(lds.storage.ptr),
        )

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def wholehead_short_persistent_kernel(
        query2d: fx.Tensor,
        kv_flat: fx.Tensor,
        kv_page_indices: fx.Tensor,
        kv_base: fx.Int32,
        out2d: fx.Tensor,
        total_q: fx.Int32,
        seq_len: fx.Int32,
        sm_scale: fx.Float32,
    ):
        """Causally balanced S<=1024 route inside one fixed-engine call.

        The opaque engine owns the alternating pass loop as well as the
        816-MFMA token body.  Its common completion drain protects output
        stores, and a converged barrier is executed only when another token
        pass will reuse this workgroup's LDS.
        """
        lds = fx.SharedAllocator().allocate(WholeHeadShared).peek()
        whole_loop_fixed_clobber(
            fx.thread_idx.x,
            fx.block_idx.x,
            buffer_ops.create_buffer_resource(out2d),
            buffer_ops.create_buffer_resource(query2d),
            buffer_ops.create_buffer_resource(kv_flat),
            buffer_ops.create_buffer_resource(kv_page_indices),
            sm_scale,
            seq_len,
            kv_base,
            fx.ptrtoint(lds.storage.ptr),
            persistent_total_q=total_q,
        )

    @flyc.jit
    def _launch_long(
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
        wholehead_kernel(
            query2d,
            kv_flat,
            kv_page_indices,
            kv_base,
            out2d,
            seq_len,
            sm_scale,
            value_attrs=value_attrs,
        ).launch(
            grid=(total_q, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    @flyc.jit
    def _launch_short(
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
        grid_x = (
            total_q < fx.Int32(SHORT_PERSISTENT_GRID)
        ).select(total_q, fx.Int32(SHORT_PERSISTENT_GRID))
        wholehead_short_persistent_kernel(
            query2d,
            kv_flat,
            kv_page_indices,
            kv_base,
            out2d,
            total_q,
            seq_len,
            sm_scale,
            value_attrs=value_attrs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch(
        query2d,
        kv_flat,
        kv_page_indices,
        kv_base,
        out2d,
        total_q,
        seq_len,
        sm_scale,
        stream=None,
    ):
        with CompilationContext.compile_hints(compile_hints):
            launcher = (
                _launch_short
                if int(total_q) <= SHORT_PERSISTENT_MAX_Q
                else _launch_long
            )
            return launcher(
                query2d,
                kv_flat,
                kv_page_indices,
                kv_base,
                out2d,
                total_q,
                seq_len,
                sm_scale,
                stream=stream,
            )

    return launch
