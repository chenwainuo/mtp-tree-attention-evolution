"""Triton sparse-prefill candidate kernels for H100 experiments.

These kernels are intentionally isolated from the FlashMLA source-of-truth
adapter. They are candidate implementations for the remote evolution loop, not
the production baseline.
"""

from __future__ import annotations

import math
from typing import Any


_SPARSE_PREFILL_KERNEL: Any | None = None


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def get_sparse_prefill_kernel(triton: Any) -> Any:
    global _SPARSE_PREFILL_KERNEL
    if _SPARSE_PREFILL_KERNEL is not None:
        return _SPARSE_PREFILL_KERNEL

    import triton.language as tl

    globals()["tl"] = tl

    @triton.jit
    def _kernel(
        q,
        kv,
        indices,
        out,
        sm_scale: tl.constexpr,
        num_heads: tl.constexpr,
        qk_dim: tl.constexpr,
        value_dim: tl.constexpr,
        topk: tl.constexpr,
        q_stride_t: tl.constexpr,
        q_stride_h: tl.constexpr,
        q_stride_d: tl.constexpr,
        kv_stride_t: tl.constexpr,
        kv_stride_h: tl.constexpr,
        kv_stride_d: tl.constexpr,
        index_stride_t: tl.constexpr,
        index_stride_k: tl.constexpr,
        out_stride_t: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_d: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        token_head = tl.program_id(0)
        value_block = tl.program_id(1)
        token = token_head // num_heads
        head = token_head - token * num_heads

        offs_k = tl.arange(0, BLOCK_K)
        offs_d = tl.arange(0, BLOCK_D)
        offs_v = value_block * BLOCK_V + tl.arange(0, BLOCK_V)

        m_i = tl.full((), -float("inf"), tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((BLOCK_V,), tl.float32)

        for k_base in range(0, topk, BLOCK_K):
            k_pos = k_base + offs_k
            selected = tl.load(
                indices + token * index_stride_t + k_pos * index_stride_k,
                mask=k_pos < topk,
                other=0,
            )
            scores = tl.zeros((BLOCK_K,), tl.float32)

            for d_base in range(0, qk_dim, BLOCK_D):
                d_pos = d_base + offs_d
                q_tile = tl.load(
                    q
                    + token * q_stride_t
                    + head * q_stride_h
                    + d_pos * q_stride_d,
                    mask=d_pos < qk_dim,
                    other=0.0,
                ).to(tl.float32)
                k_tile = tl.load(
                    kv
                    + selected[:, None] * kv_stride_t
                    + d_pos[None, :] * kv_stride_d,
                    mask=(k_pos[:, None] < topk) & (d_pos[None, :] < qk_dim),
                    other=0.0,
                ).to(tl.float32)
                scores += tl.sum(k_tile * q_tile[None, :], axis=1)

            scores *= sm_scale
            scores = tl.where(k_pos < topk, scores, -float("inf"))
            block_m = tl.max(scores, axis=0)
            m_new = tl.maximum(m_i, block_m)
            alpha = tl.exp(m_i - m_new)
            probs = tl.exp(scores - m_new)

            v_tile = tl.load(
                kv
                + selected[:, None] * kv_stride_t
                + offs_v[None, :] * kv_stride_d,
                mask=(k_pos[:, None] < topk) & (offs_v[None, :] < value_dim),
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(v_tile * probs[:, None], axis=0)
            l_i = l_i * alpha + tl.sum(probs, axis=0)
            m_i = m_new

        result = acc / l_i
        tl.store(
            out
            + token * out_stride_t
            + head * out_stride_h
            + offs_v * out_stride_d,
            result,
            mask=offs_v < value_dim,
        )

    _SPARSE_PREFILL_KERNEL = _kernel
    return _SPARSE_PREFILL_KERNEL


def triton_sparse_prefill(
    torch: Any,
    triton: Any,
    q: Any,
    kv: Any,
    indices: Any,
    sm_scale: float,
    value_dim: int,
    *,
    block_k: int,
    block_d: int,
    block_v: int,
    num_warps: int,
) -> Any:
    if not q.is_cuda or not kv.is_cuda or not indices.is_cuda:
        raise RuntimeError("Triton sparse prefill requires CUDA tensors")
    if not q.is_contiguous() or not kv.is_contiguous() or not indices.is_contiguous():
        raise RuntimeError("Triton sparse prefill expects contiguous q, kv, and indices")
    if block_k <= 0 or block_d <= 0 or block_v <= 0:
        raise ValueError("Triton block sizes must be positive")
    if block_k != _next_power_of_2(block_k):
        raise ValueError("Triton block_k must be a power of two")
    if block_d != _next_power_of_2(block_d):
        raise ValueError("Triton block_d must be a power of two")
    if block_v != _next_power_of_2(block_v):
        raise ValueError("Triton block_v must be a power of two")

    tokens, num_heads, qk_dim = q.shape
    if kv.shape[1] != 1:
        raise ValueError(f"expected a single KV head, got kv.shape={tuple(kv.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"expected flattened indices [tokens, topk], got {tuple(indices.shape)}")
    if indices.shape[0] != tokens:
        raise ValueError("indices token dimension must match q")
    if value_dim > kv.shape[-1]:
        raise ValueError("value_dim cannot exceed kv feature dimension")

    out = torch.empty((tokens, num_heads, value_dim), device=q.device, dtype=q.dtype)
    kernel = get_sparse_prefill_kernel(triton)
    grid = (tokens * num_heads, math.ceil(value_dim / block_v))
    kernel[grid](
        q,
        kv,
        indices,
        out,
        sm_scale,
        num_heads,
        qk_dim,
        value_dim,
        indices.shape[1],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        indices.stride(0),
        indices.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        BLOCK_V=block_v,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
