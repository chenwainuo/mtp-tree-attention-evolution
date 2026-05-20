# MTP Tree-Attention Kernel Evolution

This repository starts with benchmark extraction for DeepSeek V4-Flash MTP
verification attention. Stage 0 is a Triton RMSNorm warmup; Stage 1 is the first
MTP chain-attention microbenchmark.

## Current Status

- Local GPU validation is deferred until WSL2/Linux CUDA or a rented GPU is
  available.
- `bench/shapes.py` extracts DeepSeek V4-Flash shapes from `config.json` and
  includes an offline fallback snapshot.
- `bench/bench_rmsnorm.py` is the Stage 0 Triton warmup benchmark.
- `bench/bench_tree_attention.py` is the initial MTP chain-attention benchmark
  scaffold using FlashInfer ragged prefill with `custom_mask`.
- `python -m bench.run_benchmark --gpu 4090` uses a distinct FP8 K/V proxy path,
  not just 4090 roofline numbers.
- `tools/extract_flashmla.py` records the installed vLLM FlashMLA path on the
  remote CUDA host.
- `bench/bench_flashmla_sparse.py` is the low-level FlashMLA sparse benchmark
  scaffold for BF16 sparse prefill and FP8 sparse decode.

## Environment

Use Linux, WSL2 Ubuntu, or a CUDA rental instance. Do not use Cygwin/MSYS for
Triton/FlashInfer work.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Smoke test:

```bash
python -c "import torch; import triton; import flashinfer; print(torch.cuda.is_available(), triton.__version__, flashinfer.__version__)"
```

Download only the model config, not weights:

```bash
mkdir -p configs
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash config.json --local-dir configs --local-dir-use-symlinks False
mv configs/config.json configs/deepseek-v4-flash-config.json
python -m bench.shapes
```

## Stage 0 RMSNorm

```bash
python -m bench.bench_rmsnorm
```

Expected output format:

```text
DeepSeek V4-Flash RMSNorm microbenchmark
Hardware: ...
Shapes: tokens=4096, hidden_size=4096, dtype=float16
PyTorch eager runtime: ...
Triton runtime: ...
Speedup: ...
Correctness: PASS ...
```

## Stage 1 MTP Chain Attention

Unified 3090 proxy entry point:

```bash
python -m bench.run_benchmark --gpu 3090 --dry-run
python -m bench.run_benchmark --gpu 3090
```

Unified 4090 proxy entry point:

```bash
python -m bench.run_benchmark --gpu 4090 --dry-run
python -m bench.run_benchmark --gpu 4090
```

Direct proxy benchmark:

```bash
python -m bench.bench_tree_attention
```

The first topology is chain MTP: draft token `i` attends to all context tokens
and draft tokens `<= i`. This is the simplest tree and gives us a correctness
harness before implementing arbitrary branching topologies.

Important caveat: DeepSeek V4 production attention in vLLM routes through
FlashMLA sparse attention. The 3090 entry point is an FP16 dense proxy. The 4090
entry point is an FP8 K/V dense proxy with a dequantized PyTorch reference. Both
are valid for MTP chain semantics and local iteration, but neither is a
production FlashMLA baseline. By default the dense proxy uses V4's
`index_head_dim = 128` rather than the production MLA latent `head_dim = 512`,
because the plain dense FlashInfer ragged-prefill kernel does not support the
512-wide latent shape on the 3090/4090 proxy path.

## Stage 1 FlashMLA Extraction

On the remote CUDA host, first capture the installed vLLM/FlashMLA API:

```bash
python -m tools.extract_flashmla --out-dir artifacts
```

Then run the low-level FlashMLA sparse smoke/speed benchmarks:

```bash
python -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill
python -m bench.run_benchmark --gpu h100 --flashmla-mode fp8-decode
python -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill --dry-run
python -m bench.bench_flashmla_sparse --mode bf16-prefill
python -m bench.bench_flashmla_sparse --mode fp8-decode
```

Read [docs/flashmla_extraction.md](docs/flashmla_extraction.md) before treating
these as baselines. The FP8 path mirrors the production packed-cache call more
closely, but it is not a correctness benchmark until we reproduce the V4 packed
KV-cache dequantization reference.

## Local Validation Without CUDA

```bash
python -m py_compile bench/*.py tools/*.py tests/*.py
python -m bench.run_benchmark --gpu 3090 --dry-run
python -m bench.run_benchmark --gpu 4090 --dry-run
python -m bench.run_benchmark --gpu h100 --dry-run
python -m unittest discover tests
pytest  # after installing requirements.txt
```

## RunPod Remote Benchmark

`tools/runpod_benchmark.py` launches a RunPod pod, clones this public repo,
runs validation, runs the benchmark, and polls `report.json` from the pod's HTTP
artifact endpoint. The API key is read from `RUNPOD_API_KEY` or `RUNPOD` in
`.env`.

Preview the pod payload without creating anything:

```bash
python3 tools/runpod_benchmark.py --local-dry-run --gpu 4090 --remote-dry-run
```

Run the H100 FlashMLA path and save `report.json` / `output.log` under
`artifacts/runpod/`:

```bash
python3 tools/runpod_benchmark.py --gpu h100
```

Useful variants:

```bash
python3 tools/runpod_benchmark.py --gpu 4090
python3 tools/runpod_benchmark.py --gpu h100 --flashmla-mode fp8-decode
python3 tools/runpod_benchmark.py --gpu h100 --ref main --terminate-on-complete
python3 tools/runpod_benchmark.py --gpu h100 --benchmark-command "python3 -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill --rep 20"
```

If the selected RunPod image already has the exact CUDA stack installed, use
`--skip-install` or add image-specific setup with `--extra-setup-command`.

The launcher defaults to `--install-profile auto`:

- `3090` / `4090`: uses `runpod-pytorch`, which keeps the PyTorch build already
  present in the RunPod CUDA 12.8 image and installs only the benchmark extras
  from `requirements-runpod.txt`.
- `h100`: uses `runpod-vllm`, which installs vLLM with `uv pip install --system
  vllm --torch-backend=auto` before running the FlashMLA path.

Use `--install-profile pinned` to install the full local `requirements.txt`
including `torch==2.12.0`.
