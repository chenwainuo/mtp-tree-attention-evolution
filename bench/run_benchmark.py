"""Unified benchmark entry point.

This wrapper keeps the benchmark target explicit:

- 3090 and 4090 run the dense FlashInfer custom-mask proxy benchmark.
- H100 runs the FlashMLA sparse benchmark scaffold.

The proxy path is useful for local development and semantic validation, but it
is not a production FlashMLA result. The H100 path is the first production-path
adapter because vLLM FlashMLA sparse is Hopper/Blackwell-oriented.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def proxy_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--gpu",
        args.gpu,
        "--batch",
        str(args.batch),
        "--ctx-len",
        str(args.ctx_len),
        "--k-draft",
        str(args.k_draft),
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
    ]
    if args.dtype is not None:
        result.extend(["--dtype", args.dtype])
    if args.num_q_heads is not None:
        result.extend(["--num-q-heads", str(args.num_q_heads)])
    if args.num_kv_heads is not None:
        result.extend(["--num-kv-heads", str(args.num_kv_heads)])
    if args.head_dim is not None:
        result.extend(["--head-dim", str(args.head_dim)])
    if args.bandwidth_gbs is not None:
        result.extend(["--bandwidth-gbs", str(args.bandwidth_gbs)])
    result.extend(["--workspace-mb", str(args.workspace_mb)])
    if args.dry_run:
        result.append("--dry-run")
    return result


def flashmla_args(args: argparse.Namespace) -> list[str]:
    result = [
        "--mode",
        args.flashmla_mode,
        "--impl",
        args.flashmla_impl,
        "--batch",
        str(args.batch),
        "--ctx-len",
        str(args.ctx_len),
        "--k-draft",
        str(args.k_draft),
        "--topk",
        str(args.topk),
        "--block-size",
        str(args.block_size),
        "--cache-bytes-per-token",
        str(args.cache_bytes_per_token),
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--triton-block-k",
        str(args.triton_block_k),
        "--triton-block-d",
        str(args.triton_block_d),
        "--triton-block-v",
        str(args.triton_block_v),
        "--triton-warps",
        str(args.triton_warps),
    ]
    if args.num_q_heads is not None:
        result.extend(["--num-heads", str(args.num_q_heads)])
    if args.bandwidth_gbs is not None:
        result.extend(["--bandwidth-gbs", str(args.bandwidth_gbs)])
    if args.extraction_report is not None:
        result.extend(["--extraction-report", args.extraction_report])
    if args.dry_run:
        result.append("--dry-run")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        choices=("3090", "4090", "h100"),
        required=True,
        help="Benchmark target. 3090/4090 use proxy path; h100 uses FlashMLA path.",
    )
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--ctx-len", type=int, default=8192)
    parser.add_argument("--k-draft", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default=None)
    parser.add_argument("--num-q-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--cache-bytes-per-token", type=int, default=656)
    parser.add_argument("--extraction-report", default=None)
    parser.add_argument(
        "--flashmla-mode",
        choices=("bf16-prefill", "fp8-decode"),
        default="bf16-prefill",
    )
    parser.add_argument(
        "--flashmla-impl",
        choices=("flashmla", "triton"),
        default="flashmla",
        help="H100 BF16 prefill implementation. FlashMLA is the baseline.",
    )
    parser.add_argument("--bandwidth-gbs", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--workspace-mb", type=int, default=2048)
    parser.add_argument("--triton-block-k", type=int, default=32)
    parser.add_argument("--triton-block-d", type=int, default=64)
    parser.add_argument("--triton-block-v", type=int, default=64)
    parser.add_argument("--triton-warps", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.gpu in ("3090", "4090"):
        from bench import bench_tree_attention

        print(
            "Dispatch: dense FlashInfer MTP chain proxy. "
            "This is the 3090/4090 development baseline, not FlashMLA sparse."
        )
        bench_tree_attention.main(proxy_args(args))
        return

    from bench import bench_flashmla_sparse

    print(
        "Dispatch: FlashMLA sparse path. Run `python -m tools.extract_flashmla "
        "--out-dir artifacts` first on this host."
    )
    bench_flashmla_sparse.main(flashmla_args(args))


if __name__ == "__main__":
    main(sys.argv[1:])
