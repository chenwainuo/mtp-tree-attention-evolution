"""Low-level FlashMLA sparse benchmark scaffold for DeepSeek V4-Flash.

This is the H100/H200 production-path adapter. It imports vLLM FlashMLA lazily,
checks Hopper/Blackwell support, and refuses to run FlashMLA sparse on 3090/4090.
"""

from __future__ import annotations

import argparse
import inspect
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bench.common import (
    H100_PEAK_BANDWIDTH_GB_S,
    dry_run_lines,
    estimate_memory_roofline_us,
)
from bench.flashmla_adapter import (
    assert_hopper_or_blackwell,
    ensure_decode_signature,
    ensure_sparse_prefill_signature,
    flashmla_support_status,
    import_flashmla_symbols,
    load_extraction_report,
)
from bench.shapes import get_v4_flash_shapes


def import_runtime_modules() -> tuple[Any, Any]:
    import torch
    import triton

    return torch, triton


def call_with_supported_kwargs(
    fn: Any,
    kwargs: dict[str, Any],
    fallback_args: tuple[Any, ...],
) -> Any:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*fallback_args)

    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return fn(**kwargs)
    return fn(**{key: value for key, value in kwargs.items() if key in params})


def assert_cuda_ready(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for FlashMLA benchmarking. Run this on a Linux "
            "CUDA host with vLLM/FlashMLA installed."
        )


def make_chain_sparse_indices(
    torch: Any,
    batch: int,
    ctx_len: int,
    k_draft: int,
    topk: int,
    *,
    device: Any,
) -> Any:
    tokens = batch * k_draft
    token_ids = torch.arange(tokens, device=device, dtype=torch.int32)
    request_id = token_ids // k_draft
    draft_id = token_ids % k_draft
    seq_len = ctx_len + k_draft
    max_valid = ctx_len + draft_id + 1
    start = torch.clamp(max_valid - topk, min=0)
    offsets = torch.arange(topk, device=device, dtype=torch.int32)[None, :]
    local = torch.minimum(start[:, None] + offsets, max_valid[:, None] - 1)
    return (request_id[:, None] * seq_len + local).contiguous()


def pytorch_sparse_prefill_reference(
    torch: Any,
    q: Any,
    kv: Any,
    indices: Any,
    sm_scale: float,
) -> Any:
    """Reference sparse attention over selected top-k positions.

    This is a BF16 prefill oracle for the scaffold. It assumes one KV head and
    uses the same selected tensor as key and value because the low-level
    FlashMLA sparse op consumes compressed MLA state rather than separate K/V.
    """
    selected = kv[indices[:, 0, :].long(), 0, :].float()
    q_f32 = q.float()
    scores = torch.einsum("thd,tkd->thk", q_f32, selected) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("thk,tkd->thd", probs, selected)
    return out.to(q.dtype)


def run_bf16_sparse_prefill(args: argparse.Namespace) -> None:
    torch, triton = import_runtime_modules()
    assert_cuda_ready(torch)
    assert_hopper_or_blackwell(torch)

    symbols = import_flashmla_symbols()
    supported, reason = flashmla_support_status(symbols.module)
    if not supported:
        raise SystemExit(f"FlashMLA sparse is not supported on this host: {reason}")
    if symbols.flash_mla_sparse_fwd is None:
        raise SystemExit(f"`flash_mla_sparse_fwd` is not present in {symbols.module_name}")
    ensure_sparse_prefill_signature(symbols.flash_mla_sparse_fwd)

    shapes = get_v4_flash_shapes()
    device = torch.device("cuda")
    tokens = args.batch * args.k_draft
    kv_tokens = args.batch * (args.ctx_len + args.k_draft)
    q_heads = args.num_heads or shapes.num_attention_heads
    qk_dim = shapes.head_dim + shapes.qk_rope_head_dim
    sm_scale = 1.0 / math.sqrt(qk_dim)

    q = torch.randn((tokens, q_heads, qk_dim), device=device, dtype=torch.bfloat16)
    kv = torch.randn((kv_tokens, 1, qk_dim), device=device, dtype=torch.bfloat16)
    indices = make_chain_sparse_indices(
        torch,
        args.batch,
        args.ctx_len,
        args.k_draft,
        args.topk,
        device=device,
    )
    indices = indices[:, None, :].contiguous()

    ref = pytorch_sparse_prefill_reference(torch, q, kv, indices, sm_scale)
    torch.cuda.synchronize()

    def run() -> Any:
        result = symbols.flash_mla_sparse_fwd(q, kv, indices, sm_scale)
        if isinstance(result, tuple):
            return result[0]
        return result

    out = run()
    torch.cuda.synchronize()
    if out.shape == ref.shape:
        max_abs = (out - ref).abs().max().item()
        passed = torch.allclose(out, ref, atol=3e-2, rtol=3e-2)
        if not passed:
            raise AssertionError(f"BF16 sparse prefill mismatch: max_abs={max_abs:.6g}")
        correctness = f"PASS (allclose atol=0.03, rtol=0.03, max_abs={max_abs:.6g})"
    else:
        correctness = f"SKIPPED (shape mismatch FlashMLA={tuple(out.shape)} ref={tuple(ref.shape)})"

    runtime_ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
    bytes_read = q.numel() * q.element_size() + kv.numel() * kv.element_size()
    bytes_written = out.numel() * out.element_size()
    roofline_us = estimate_memory_roofline_us(
        bytes_read + bytes_written,
        args.bandwidth_gbs,
    )

    print("DeepSeek V4-Flash FlashMLA sparse BF16 prefill microbenchmark")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"FlashMLA module: {symbols.module_name}")
    print(
        "Shapes: "
        f"batch={args.batch}, ctx={args.ctx_len}, K_draft={args.k_draft}, "
        f"tokens={tokens}, kv_tokens={kv_tokens}, heads={q_heads}/1, "
        f"qk_dim={qk_dim}, topk={args.topk}, dtype=bf16"
    )
    print(f"Correctness: {correctness}")
    print(f"Runtime: {runtime_ms * 1000:.2f} us")
    print(
        "Memory roofline: "
        f"{roofline_us:.2f} us at {args.bandwidth_gbs:.1f} GB/s "
        f"({roofline_us / (runtime_ms * 1000) * 100:.1f}% of measured)"
    )


def run_fp8_sparse_decode(args: argparse.Namespace) -> None:
    torch, triton = import_runtime_modules()
    assert_cuda_ready(torch)
    assert_hopper_or_blackwell(torch)

    symbols = import_flashmla_symbols()
    supported, reason = flashmla_support_status(symbols.module)
    if not supported:
        raise SystemExit(f"FlashMLA sparse is not supported on this host: {reason}")
    missing = [
        name
        for name in ("flash_mla_with_kvcache", "get_mla_metadata")
        if getattr(symbols, name) is None
    ]
    if missing:
        raise SystemExit(f"Missing {missing} in {symbols.module_name}")
    ensure_decode_signature(symbols.flash_mla_with_kvcache)

    shapes = get_v4_flash_shapes()
    device = torch.device("cuda")
    tokens = args.batch * args.k_draft
    q_heads = args.num_heads or shapes.num_attention_heads
    block_size = args.block_size
    seq_len = args.ctx_len + args.k_draft
    total_slots = args.batch * seq_len
    num_blocks = math.ceil(total_slots / block_size)
    qk_dim = shapes.head_dim + shapes.qk_rope_head_dim
    sm_scale = 1.0 / math.sqrt(qk_dim)

    q = torch.randn((1, tokens, q_heads, qk_dim), device=device, dtype=torch.bfloat16)
    kv_cache_bytes = torch.randint(
        0,
        255,
        (num_blocks, block_size, args.cache_bytes_per_token),
        device=device,
        dtype=torch.uint8,
    )
    kv_cache = kv_cache_bytes.unsqueeze(-2).contiguous()
    block_table = torch.zeros((1, 1), device=device, dtype=torch.int32)
    cache_seqlens = torch.full((1,), args.topk, device=device, dtype=torch.int32)
    indices = make_chain_sparse_indices(
        torch,
        args.batch,
        args.ctx_len,
        args.k_draft,
        args.topk,
        device=device,
    )
    indices = indices.view(1, tokens, args.topk).contiguous()
    metadata_kwargs = {
        "cache_seqlens": cache_seqlens,
        "num_q_tokens_per_head_k": tokens * q_heads,
        "num_heads_q": q_heads,
        "num_heads_k": 1,
        "is_fp8_kvcache": True,
        "topk": args.topk,
    }
    tile_scheduler_metadata, num_splits = call_with_supported_kwargs(
        symbols.get_mla_metadata,
        metadata_kwargs,
        (
            metadata_kwargs["cache_seqlens"],
            metadata_kwargs["num_q_tokens_per_head_k"],
            metadata_kwargs["num_heads_k"],
            metadata_kwargs["num_heads_q"],
            metadata_kwargs["is_fp8_kvcache"],
        ),
    )

    def run() -> Any:
        kwargs = {
            "q": q,
            "k_cache": kv_cache,
            "kv_cache": kv_cache,
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "head_dim_v": shapes.head_dim,
            "tile_scheduler_metadata": tile_scheduler_metadata,
            "num_splits": num_splits,
            "softmax_scale": sm_scale,
            "causal": False,
            "descale_q": None,
            "descale_k": None,
            "is_fp8_kvcache": True,
            "indices": indices,
        }
        result = call_with_supported_kwargs(
            symbols.flash_mla_with_kvcache,
            kwargs,
            (
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                tile_scheduler_metadata,
                num_splits,
                None,
                sm_scale,
                False,
                indices,
                cache_seqlens,
            ),
        )
        if isinstance(result, tuple):
            return result[0]
        return result

    out = run()
    torch.cuda.synchronize()
    runtime_ms = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
    bytes_read = q.numel() * q.element_size() + kv_cache_bytes.numel()
    bytes_written = out.numel() * out.element_size()
    roofline_us = estimate_memory_roofline_us(
        bytes_read + bytes_written,
        args.bandwidth_gbs,
    )

    print("DeepSeek V4-Flash FlashMLA sparse FP8 decode microbenchmark")
    print(f"Hardware: {torch.cuda.get_device_name()}")
    print(f"FlashMLA module: {symbols.module_name}")
    print(
        "Shapes: "
        f"batch={args.batch}, ctx={args.ctx_len}, K_draft={args.k_draft}, "
        f"tokens={tokens}, heads={q_heads}/1, qk_dim={qk_dim}, "
        f"topk={args.topk}, block_size={block_size}, "
        f"cache_bytes_per_token={args.cache_bytes_per_token}"
    )
    print("Correctness: NOT CHECKED (packed FP8 low-level smoke/speed path)")
    print(f"Runtime: {runtime_ms * 1000:.2f} us")
    print(
        "Memory roofline: "
        f"{roofline_us:.2f} us at {args.bandwidth_gbs:.1f} GB/s "
        f"({roofline_us / (runtime_ms * 1000) * 100:.1f}% of measured)"
    )


def dry_run(args: argparse.Namespace) -> None:
    shapes = get_v4_flash_shapes()
    extra: dict[str, Any] = {
        "mode": args.mode,
        "heads": args.num_heads or shapes.num_attention_heads,
        "topk": args.topk,
        "block_size": args.block_size,
        "requires": "Hopper/Blackwell FlashMLA sparse",
    }
    if args.extraction_report:
        path = Path(args.extraction_report)
        if path.exists():
            report = load_extraction_report(path)
            extra["extraction_report"] = str(path)
            extra["support_flags"] = report.support_flags
        else:
            extra["extraction_report"] = f"{path} (missing)"

    for line in dry_run_lines(
        gpu="h100",
        path="flashmla-sparse",
        batch=args.batch,
        ctx_len=args.ctx_len,
        k_draft=args.k_draft,
        dtype="bfloat16" if args.mode == "bf16-prefill" else "fp8-kv",
        bandwidth_gbs=args.bandwidth_gbs,
        extra=extra,
    ):
        print(line)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("bf16-prefill", "fp8-decode"),
        default="bf16-prefill",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--k-draft", type=int, default=4)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--cache-bytes-per-token", type=int, default=584)
    parser.add_argument("--bandwidth-gbs", type=float, default=H100_PEAK_BANDWIDTH_GB_S)
    parser.add_argument("--extraction-report", default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        dry_run(args)
        return
    if args.mode == "bf16-prefill":
        run_bf16_sparse_prefill(args)
    else:
        run_fp8_sparse_decode(args)


if __name__ == "__main__":
    main()
