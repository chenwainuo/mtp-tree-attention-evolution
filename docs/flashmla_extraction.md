# FlashMLA Extraction Plan

The dense FlashInfer custom-mask benchmark is a useful proxy, but it is not the
production DeepSeek V4-Flash path. V4 routes through vLLM's FlashMLA sparse
attention backend, so Stage 1 needs a second benchmark layer that extracts and
tests that path directly.

## Production Path To Mirror

Source anchors:

- `vllm/model_executor/layers/deepseek_v4_attention.py`
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `vllm/v1/attention/ops/flashmla.py`
- `vllm/attention/ops/flashmla.py`

The relevant flow is:

1. DeepSeek V4 attention wrapper builds/updates the compressed V4 KV cache.
2. vLLM chooses the MLA sparse backend when `is_flashmla_sparse_supported()` is
   true. In current docs, this is a Hopper/Blackwell path, so H100/H200 are the
   real validation targets.
3. The BF16 sparse prefill path calls `flash_mla_sparse_fwd`.
4. The FP8 sparse decode path calls `flash_mla_with_kvcache` with:
   - query tensor including latent + rope dimensions;
   - packed V4 FP8 KV cache, currently 584 bytes per token;
   - top-k sparse indices;
   - tile scheduler metadata from `get_mla_metadata`.

## Why This Matters

Tree attention in MTP is not just a dense causal mask. For V4-Flash, the
attention path is sparse and compressed. A dense FlashInfer custom-mask result
can validate the tree/chain semantics, but a performance win there may not
transfer to V4's sparse FlashMLA production path.

## Remote Extraction Command

After installing the remote CUDA environment, run:

```bash
python -m tools.extract_flashmla --out-dir artifacts
```

This writes:

- `artifacts/flashmla_extraction.json`
- `artifacts/flashmla_extraction.md`

Those files record installed versions, CUDA visibility, FlashMLA support flags,
module paths, symbol signatures, and short source excerpts. Use them before
modifying the benchmark. If the signatures differ from the scaffold, patch the
benchmark to the installed vLLM version rather than guessing.

## Remote Benchmark Commands

Unified proxy command for 3090:

```bash
python -m bench.run_benchmark --gpu 3090
```

Unified proxy command for 4090:

```bash
python -m bench.run_benchmark --gpu 4090
```

These run dense FlashInfer custom-mask proxies, not FlashMLA sparse. The 3090
path uses FP16 K/V. The 4090 path uses FP8 K/V plus explicit scales and a
dequantized PyTorch reference. Use them for local MTP chain semantics,
correctness harness work, and candidate screening.

BF16 sparse prefill smoke/speed path:

```bash
python -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill
python -m bench.bench_flashmla_sparse --mode bf16-prefill
```

FP8 packed sparse decode smoke/speed path:

```bash
python -m bench.run_benchmark --gpu h100 --flashmla-mode fp8-decode
python -m bench.bench_flashmla_sparse --mode fp8-decode
```

The FP8 mode is intentionally marked as a smoke/speed benchmark until we build a
numeric PyTorch reference for V4's packed KV-cache format. The correctness
benchmark still starts with `bench_tree_attention.py`, which uses explicit
PyTorch dense chain attention and FlashInfer custom masks.

## Open Items Before Claiming A Real Baseline

- Confirm the installed vLLM version exposes the same FlashMLA symbol signatures
  used by `bench_flashmla_sparse.py`.
- Confirm the packed V4 FP8 cache byte layout in the installed vLLM source.
- Replace synthetic sparse top-k indices with the same top-k construction used
  by V4's sparse attention metadata path.
- Add a correctness reference for BF16 sparse prefill first. Do not claim FP8
  correctness until packed-cache dequantization is reproduced in Python.
- Measure on H100/H200 for FlashMLA. A 3090 cannot validate this path because
  FlashMLA sparse support is not expected on SM86.
