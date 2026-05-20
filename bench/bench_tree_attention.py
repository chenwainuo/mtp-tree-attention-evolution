"""Dense FlashInfer MTP chain-attention proxy benchmark.

This is the 3090/4090 development baseline. It validates MTP chain semantics
with an explicit PyTorch reference and a FlashInfer ragged-prefill custom mask.

Targets:
  3090: FP16 Q/K/V proxy.
  4090: FP16 Q/O with FP8 e4m3 K/V proxy and dequantized reference.

This is not the DeepSeek V4 production FlashMLA sparse path; H100 uses
`bench_flashmla_sparse.py`.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any

from bench.common import (
    TARGET_DEFAULTS,
    dry_run_lines,
    estimate_memory_roofline_us,
    require_target,
)
from bench.fp8 import Fp8TensorPair, quantize_fp8_tensor
from bench.shapes import V4FlashShapes, get_v4_flash_shapes


def chain_allows(ctx_len: int, draft_id: int, kv_position: int) -> bool:
    return kv_position < ctx_len + draft_id + 1


def import_runtime_modules() -> tuple[Any, Any]:
    import torch
    import triton

    return torch, triton


def assert_cuda_ready(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for this benchmark. Run it inside WSL2/Linux with "
            "an NVIDIA GPU or on a rented CUDA instance."
        )


def parse_dtype(torch: Any, name: str) -> Any:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype {name!r}; use float16 or bfloat16")


def default_proxy_head_dim(shapes: V4FlashShapes) -> int:
    """Use a dense-kernel-compatible head dimension for the proxy path.

    DeepSeek V4's production MLA path uses a 512-wide latent attention state,
    but the 3090/4090 proxy runs plain dense FlashInfer ragged prefill. The
    configured index head dimension keeps the proxy tied to V4 metadata while
    staying inside FlashInfer's supported dense kernel configurations.
    """
    return shapes.index_head_dim


def make_chain_mask(
    torch: Any,
    batch: int,
    ctx_len: int,
    k_draft: int,
    *,
    device: Any,
) -> Any:
    """Return [batch, K_draft, ctx_len + K_draft] boolean MTP chain mask."""
    draft_i = torch.arange(k_draft, device=device)[:, None]
    kv_j = torch.arange(ctx_len + k_draft, device=device)[None, :]
    per_request = kv_j < (ctx_len + draft_i + 1)
    return per_request.unsqueeze(0).expand(batch, -1, -1).contiguous()


def make_inputs(
    torch: Any,
    batch: int,
    ctx_len: int,
    k_draft: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    device: Any,
    *,
    use_fp8_kv: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, Fp8TensorPair | None, Fp8TensorPair | None]:
    qo_tokens = batch * k_draft
    kv_tokens = batch * (ctx_len + k_draft)
    q = torch.randn((qo_tokens, num_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((kv_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((kv_tokens, num_kv_heads, head_dim), device=device, dtype=dtype)
    mask = make_chain_mask(torch, batch, ctx_len, k_draft, device=device)

    if not use_fp8_kv:
        return q, k, v, k, v, mask, None, None

    k_fp8 = quantize_fp8_tensor(torch, k)
    v_fp8 = quantize_fp8_tensor(torch, v)
    return (
        q,
        k_fp8.quantized,
        v_fp8.quantized,
        k_fp8.dequantized,
        v_fp8.dequantized,
        mask,
        k_fp8,
        v_fp8,
    )


def pytorch_chain_attention(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    mask: Any,
    batch: int,
    ctx_len: int,
    k_draft: int,
    sm_scale: float,
) -> Any:
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    seq_len = ctx_len + k_draft

    q_b = q.view(batch, k_draft, num_q_heads, q.shape[-1]).float()
    k_b = k.view(batch, seq_len, num_kv_heads, k.shape[-1]).float()
    v_b = v.view(batch, seq_len, num_kv_heads, v.shape[-1]).float()

    # GQA reference intentionally materializes expanded K/V. Optimized kernels
    # must avoid this, but clarity matters more in the correctness oracle.
    k_b = k_b.repeat_interleave(group_size, dim=2)
    v_b = v_b.repeat_interleave(group_size, dim=2)

    scores = torch.einsum("bqhd,bkhd->bqhk", q_b, k_b) * sm_scale
    scores = scores.masked_fill(~mask[:, :, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqhk,bkhd->bqhd", probs, v_b)
    return out.to(q.dtype).reshape(batch * k_draft, num_q_heads, q.shape[-1])


def call_flashinfer_run(
    run_fn: Any,
    q: Any,
    k: Any,
    v: Any,
    *,
    use_fp8_kv: bool,
    k_scale: float | None,
    v_scale: float | None,
) -> Any:
    if not use_fp8_kv:
        return run_fn(q, k, v)

    try:
        return run_fn(q, k, v, k_scale=k_scale, v_scale=v_scale, return_lse=False)
    except TypeError as exc:
        raise RuntimeError(
            "Installed FlashInfer wrapper.run does not accept explicit FP8 "
            "`k_scale`, `v_scale`, `return_lse` keyword arguments. This 4090 "
            "FP8 proxy must be patched to the installed FlashInfer API rather "
            "than silently timing the wrong path."
        ) from exc


def make_flashinfer_chain_runner(
    torch: Any,
    q: Any,
    k: Any,
    v: Any,
    mask: Any,
    batch: int,
    ctx_len: int,
    k_draft: int,
    sm_scale: float,
    *,
    use_fp8_kv: bool,
    k_scale: float | None,
    v_scale: float | None,
    workspace_bytes: int,
) -> Any:
    import flashinfer

    device = q.device
    seq_len = ctx_len + k_draft
    qo_indptr = torch.arange(
        0, (batch + 1) * k_draft, k_draft, device=device, dtype=torch.int32
    )
    kv_indptr = torch.arange(
        0, (batch + 1) * seq_len, seq_len, device=device, dtype=torch.int32
    )

    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
    )
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        q.shape[1],
        k.shape[1],
        q.shape[-1],
        custom_mask=mask.reshape(-1),
        causal=False,
        sm_scale=sm_scale,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )

    def run() -> Any:
        return call_flashinfer_run(
            wrapper.run,
            q,
            k,
            v,
            use_fp8_kv=use_fp8_kv,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    return run


def estimate_dense_proxy_bytes(
    batch: int,
    ctx_len: int,
    k_draft: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_bytes_per_elem: int,
    kv_bytes_per_elem: int,
) -> int:
    q_bytes = batch * k_draft * num_q_heads * head_dim * q_bytes_per_elem
    kv_bytes = batch * (ctx_len + k_draft) * num_kv_heads * head_dim * kv_bytes_per_elem * 2
    out_bytes = batch * k_draft * num_q_heads * head_dim * q_bytes_per_elem
    return q_bytes + kv_bytes + out_bytes


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--k-draft", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default=None)
    parser.add_argument(
        "--gpu",
        choices=("3090", "4090"),
        default="3090",
        help="Proxy benchmark GPU target.",
    )
    parser.add_argument("--num-q-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--bandwidth-gbs", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=2048,
        help="FlashInfer planning workspace size in MiB.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-flashinfer", action="store_true")
    args = parser.parse_args(argv)

    target = require_target(args.gpu)
    dtype_name = args.dtype or target.dtype
    bandwidth_gbs = args.bandwidth_gbs or target.bandwidth_gbs
    shapes = get_v4_flash_shapes()
    num_q_heads = args.num_q_heads or shapes.num_attention_heads
    num_kv_heads = args.num_kv_heads or shapes.num_key_value_heads
    head_dim = args.head_dim or default_proxy_head_dim(shapes)

    if args.dry_run:
        for line in dry_run_lines(
            gpu=args.gpu,
            path=target.path,
            batch=args.batch,
            ctx_len=args.ctx_len,
            k_draft=args.k_draft,
            dtype=dtype_name,
            bandwidth_gbs=bandwidth_gbs,
            extra={
                "heads": f"{num_q_heads}/{num_kv_heads}",
                "head_dim": head_dim,
                "fp8_kv": target.uses_fp8_kv,
                "workspace_mb": args.workspace_mb,
            },
        ):
            print(line)
        return

    torch, triton = import_runtime_modules()
    assert_cuda_ready(torch)
    dtype = parse_dtype(torch, dtype_name)
    device = torch.device("cuda")

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    q, k_kernel, v_kernel, k_ref, v_ref, mask, k_fp8, v_fp8 = make_inputs(
        torch,
        args.batch,
        args.ctx_len,
        args.k_draft,
        num_q_heads,
        num_kv_heads,
        head_dim,
        dtype,
        device,
        use_fp8_kv=target.uses_fp8_kv,
    )
    sm_scale = 1.0 / math.sqrt(head_dim)

    ref = pytorch_chain_attention(
        torch,
        q,
        k_ref,
        v_ref,
        mask,
        args.batch,
        args.ctx_len,
        args.k_draft,
        sm_scale,
    )
    torch.cuda.synchronize()

    print("DeepSeek V4-Flash MTP chain-attention proxy microbenchmark")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"Benchmark path: {target.path}")
    print(
        "Shapes: "
        f"batch={args.batch}, ctx={args.ctx_len}, K_draft={args.k_draft}, "
        f"heads={num_q_heads}/{num_kv_heads}, head_dim={head_dim}, "
        f"dtype={dtype_name}, fp8_kv={target.uses_fp8_kv}"
    )
    if k_fp8 is not None and v_fp8 is not None:
        print(f"FP8 scales: k_scale={k_fp8.scale:.8g}, v_scale={v_fp8.scale:.8g}")

    if args.skip_flashinfer:
        print("FlashInfer runtime: SKIPPED")
        return

    flashinfer_run = make_flashinfer_chain_runner(
        torch,
        q,
        k_kernel,
        v_kernel,
        mask,
        args.batch,
        args.ctx_len,
        args.k_draft,
        sm_scale,
        use_fp8_kv=target.uses_fp8_kv,
        k_scale=None if k_fp8 is None else k_fp8.scale,
        v_scale=None if v_fp8 is None else v_fp8.scale,
        workspace_bytes=args.workspace_mb * 1024 * 1024,
    )
    out = flashinfer_run()
    torch.cuda.synchronize()

    atol = 3e-2 if target.uses_fp8_kv else 1e-2
    rtol = 3e-2 if target.uses_fp8_kv else 1e-2
    passed = torch.allclose(out, ref, atol=atol, rtol=rtol)
    max_abs = (out - ref).abs().max().item()
    if not torch.isfinite(out.float()).all():
        raise AssertionError("FlashInfer output contains NaN or inf")
    if not passed:
        raise AssertionError(
            f"FlashInfer mismatch: max_abs={max_abs:.6g}, atol={atol}, rtol={rtol}"
        )

    flashinfer_ms = triton.testing.do_bench(
        flashinfer_run,
        warmup=args.warmup,
        rep=args.rep,
    )
    kv_element_size = 1 if target.uses_fp8_kv else k_kernel.element_size()
    total_bytes = estimate_dense_proxy_bytes(
        args.batch,
        args.ctx_len,
        args.k_draft,
        num_q_heads,
        num_kv_heads,
        head_dim,
        q.element_size(),
        kv_element_size,
    )
    roofline_us = estimate_memory_roofline_us(total_bytes, bandwidth_gbs)

    print(
        "PyTorch reference matches FlashInfer: "
        f"PASS (allclose atol={atol}, rtol={rtol}, max_abs={max_abs:.6g})"
    )
    print(f"FlashInfer runtime: {flashinfer_ms * 1000:.2f} us")
    print(
        "Memory roofline: "
        f"{roofline_us:.2f} us at {bandwidth_gbs:.1f} GB/s "
        f"({roofline_us / (flashinfer_ms * 1000) * 100:.1f}% of measured)"
    )
    if roofline_us > flashinfer_ms * 1000:
        print("Roofline sanity: FAIL (measured time is below memory roofline)")
    else:
        print("Roofline sanity: PASS")


if __name__ == "__main__":
    main()
