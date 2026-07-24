"""FlyDSL absorbed-MLA PREFILL — query-tiled 576/512 causal attention (gfx950).

STABLE INTERFACE for the KernelForge run-mode driver. The driver calls
``flydsl_mla_prefill(...)`` with a FIXED signature; the FELLOW implements the
device kernel behind it. Do NOT change this signature — the driver depends on it.

Why query-tiled (NOT the decode kernel): the decode kernel's split-KV +
partial-reduce metadata is O(queries²) (reduce_partial_map=65536 at S=256 -> a
4.4 TB partial buffer). Prefill must TILE QUERY TOKENS (flash-attention style):
each workgroup owns BLOCK_M query rows, streams KV tiles with online softmax and
a causal mask, and writes O directly — no split buffers, no partial reduce.

Absorbed MLA math (bf16 target; fp8 seed OK): all 128 heads attend to ONE shared
576-wide latent per KV token (512 c_kv + 64 k_pe). Score(head, q, k) =
q[576]·kv[576] * sm_scale; Out(head, q) = softmax_k(Score) · kv[:512]. So
QK_HEAD_DIM=576, V_HEAD_DIM=512, 128 query heads, 1 KV "head" (shared latent).

REFERENCES for the implementation (Read these):
- kernels/attention/flash_attn_gfx950.py  -> the query-tile + causal + online-softmax
  + dualwave-SWP structure to mirror (but it is head_dim 64/128 only; you extend to 576/512).
- kernels/attention/mla_fwd_decode_m16x8_fp8_fp8.py -> the 576/512 MMA / V3 KV-LDS /
  ds_read_b64_tr_b8 layout + the -inf masking helper (_softmax_scale_p) to reuse.
- knowledge_base flydsl/attention_optimization.md + lds_optimization.md.
"""
from __future__ import annotations

import os

import torch

import functools

QK_HEAD_DIM = 576
V_HEAD_DIM = 512
NUM_QO_HEADS = 128

# The chained-MFMA flash-attention kernel (mla_prefill_qtiled_mfma.py) is the
# performance path and is VERIFIED GREEN at S in {256,512,1024}
# (allclose True, cos_diff ~2e-6, SNR ~54 dB) and ~6x faster than the scalar
# kernel at S=1024 (5.77 ms vs 34.4 ms).  It is the default; on ANY exception
# it falls back to the correctness-banked scalar kernel.  Set
# FLYDSL_MLA_PREFILL_MFMA=0 to force the scalar path.
_USE_MFMA = os.environ.get("FLYDSL_MLA_PREFILL_MFMA", "1") not in ("0", "false", "False")


@functools.lru_cache(maxsize=None)
def _get_launcher(is_causal: bool):
    from kernels.attention.mla_prefill_qtiled_kernel import build_mla_prefill_qtiled_module

    return build_mla_prefill_qtiled_module(is_causal=is_causal)


@functools.lru_cache(maxsize=None)
def _get_mfma_launcher(is_causal: bool):
    from kernels.attention.mla_prefill_qtiled_mfma import build_mla_prefill_mfma_module

    return build_mla_prefill_mfma_module(is_causal=is_causal)


def flydsl_mla_prefill(
    query: torch.Tensor,       # [total_q, 128, 576]  (bf16 target; fp8 seed accepted)
    kv_buffer: torch.Tensor,   # [num_page, 1, 1, 576] paged latent (same dtype as query)
    kv_page_indices: torch.Tensor,   # [num_page] logical->physical page map
    qo_indptr: torch.Tensor,   # [batch+1] query-token CSR
    kv_indptr: torch.Tensor,   # [batch+1] kv-page CSR
    out: torch.Tensor,         # [total_q, 128, 512] bf16  (written in place)
    sm_scale: float,
    causal: bool = True,
) -> None:
    """Query-tiled absorbed-MLA prefill. Writes attention output into ``out``.

    For full prefill (one sequence, kv_len == q_len == S): query token j attends
    KV logical positions 0..j (causal). Multi-sequence via the CSR indptrs.

    Batch=1 full prefill: token j attends KV logical 0..j (causal). Logical KV
    position i lives in physical page kv_page_indices[kv_indptr[b] + i].
    """
    assert query.ndim == 3 and query.size(1) == NUM_QO_HEADS and query.size(2) == QK_HEAD_DIM, (
        f"query: expected [total_q, {NUM_QO_HEADS}, {QK_HEAD_DIM}], got {list(query.shape)}"
    )
    assert out.shape == (query.size(0), NUM_QO_HEADS, V_HEAD_DIM), (
        f"out: expected [{query.size(0)}, {NUM_QO_HEADS}, {V_HEAD_DIM}], got {list(out.shape)}"
    )
    assert out.is_contiguous(), "out must be contiguous (kernel writes it in place)"

    total_q = query.size(0)
    num_page = kv_buffer.size(0)

    # batch=1 CSR: kv logical positions live at kv_page_indices[kv_indptr[0] + i]
    kv_base = int(kv_indptr[0].item())

    query_flat = query.reshape(-1).contiguous() if not query.is_contiguous() else query.reshape(-1)
    kv_flat = kv_buffer.reshape(num_page * QK_HEAD_DIM)
    out_flat = out.view(-1)
    kv_idx = kv_page_indices.contiguous().to(torch.int32)

    # ---- Performance path: chained-MFMA flash attention.  On ANY failure
    # (compile / spill / launch), fall back to the correctness-banked scalar
    # kernel so correctness can never regress. ----
    if _USE_MFMA:
        try:
            query2d = query_flat.view(total_q * NUM_QO_HEADS, QK_HEAD_DIM)
            out2d = out_flat.view(total_q * NUM_QO_HEADS, V_HEAD_DIM)
            mfma_launcher = _get_mfma_launcher(bool(causal))
            mfma_launcher(
                query2d,
                kv_flat,
                kv_idx,
                int(kv_base),
                out2d,
                int(total_q),
                int(total_q),
                float(sm_scale),
                stream=torch.cuda.current_stream(),
            )
            return
        except Exception as exc:  # noqa: BLE001 -- fall back to banked scalar path
            if os.environ.get("FLYDSL_MLA_PREFILL_DEBUG"):
                import traceback

                print("[mla_prefill] MFMA path failed, falling back to scalar:", exc)
                traceback.print_exc()

    launcher = _get_launcher(bool(causal))
    launcher(
        query_flat,
        kv_flat,
        kv_idx,
        int(kv_base),
        out_flat,
        int(total_q),
        int(total_q),
        float(sm_scale),
        stream=torch.cuda.current_stream(),
    )
