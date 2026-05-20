"""Stage 0 Triton RMSNorm warmup benchmark."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from typing import Any

from bench.shapes import get_v4_flash_shapes


def import_runtime_modules() -> tuple[Any, Any, Any]:
    import torch
    import triton
    import triton.language as tl

    return torch, triton, tl


def make_rmsnorm_kernel(triton: Any, tl: Any) -> Any:
    @triton.jit
    def _rmsnorm_kernel(
        x_ptr,
        weight_ptr,
        y_ptr,
        hidden_size: tl.constexpr,
        eps: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size

        x = tl.load(x_ptr + row_id * hidden_size + offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        mean_square = tl.sum(x_f32 * x_f32, axis=0) / hidden_size
        inv_rms = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = x_f32 * inv_rms * weight
        tl.store(y_ptr + row_id * hidden_size + offsets, y, mask=mask)

    return _rmsnorm_kernel


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def rmsnorm_torch(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    x_f32 = x.float()
    inv_rms = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * inv_rms * weight.float()).to(x.dtype)


def rmsnorm_triton(torch: Any, kernel: Any, x: Any, weight: Any, eps: float) -> Any:
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [tokens, hidden], got {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    tokens, hidden_size = x.shape
    if weight.numel() != hidden_size:
        raise ValueError(
            f"weight length {weight.numel()} must match hidden_size {hidden_size}"
        )

    block_size = _next_power_of_2(hidden_size)
    y = torch.empty_like(x)
    kernel[(tokens,)](
        x,
        weight,
        y,
        hidden_size,
        eps,
        block_size,
        num_warps=8,
    )
    return y


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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    shapes = get_v4_flash_shapes()
    hidden_size = args.hidden_size or shapes.hidden_size
    if args.dry_run:
        print("RMSNorm dry run")
        print(f"  tokens: {args.tokens}")
        print(f"  hidden_size: {hidden_size}")
        print(f"  dtype: {args.dtype}")
        return

    torch, triton, tl = import_runtime_modules()
    kernel = make_rmsnorm_kernel(triton, tl)
    assert_cuda_ready(torch)
    dtype = parse_dtype(torch, args.dtype)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    x = torch.randn((args.tokens, hidden_size), device=device, dtype=dtype)
    weight = torch.randn((hidden_size,), device=device, dtype=dtype)
    eps = shapes.rms_norm_eps

    y_ref = rmsnorm_torch(torch, x, weight, eps)
    y_tri = rmsnorm_triton(torch, kernel, x, weight, eps)
    torch.cuda.synchronize()

    atol = 1e-2
    rtol = 1e-2
    passed = torch.allclose(y_tri, y_ref, atol=atol, rtol=rtol)
    max_abs = (y_tri - y_ref).abs().max().item()

    if not passed:
        raise AssertionError(
            f"RMSNorm mismatch: max_abs={max_abs:.6g}, atol={atol}, rtol={rtol}"
        )

    torch_ms = triton.testing.do_bench(
        lambda: rmsnorm_torch(torch, x, weight, eps),
        warmup=args.warmup,
        rep=args.rep,
    )
    triton_ms = triton.testing.do_bench(
        lambda: rmsnorm_triton(torch, kernel, x, weight, eps),
        warmup=args.warmup,
        rep=args.rep,
    )

    speedup = torch_ms / triton_ms
    total_bytes = args.tokens * hidden_size * x.element_size() * 2
    total_bytes += hidden_size * weight.element_size()
    bandwidth_gbs = total_bytes / (triton_ms / 1e3) / 1e9

    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print("DeepSeek V4-Flash RMSNorm microbenchmark")
    print(f"Hardware: {device_name}, SM{capability[0]}{capability[1]}")
    print(
        "Shapes: "
        f"tokens={args.tokens}, hidden_size={hidden_size}, dtype={args.dtype}"
    )
    print(f"PyTorch eager runtime: {torch_ms * 1000:.2f} us")
    print(f"Triton runtime: {triton_ms * 1000:.2f} us")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Approx Triton memory bandwidth: {bandwidth_gbs:.1f} GB/s")
    print(
        "Correctness: PASS "
        f"(allclose atol={atol}, rtol={rtol}, max_abs={max_abs:.6g})"
    )

    if not math.isfinite(speedup) or speedup < 2.0:
        print("Target: FAIL (wanted >=2.00x over PyTorch eager)")
    else:
        print("Target: PASS (>=2.00x over PyTorch eager)")


if __name__ == "__main__":
    main()

