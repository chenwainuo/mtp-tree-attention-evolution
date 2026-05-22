# Real-Time Inference Benchmark Plan

## Goal

Measure whether the FlashMLA sparse prefill optimization improves real-time
serving, not just the low-level BF16 sparse prefill microbenchmark.

The current benchmark proves that a specific FlashMLA kernel call can get
faster while preserving numeric correctness. It does not prove end-to-end
serving improvement. Real-time inference only improves if this kernel is on the
request critical path and accounts for a meaningful share of latency.

Expected impact:

```text
end-to-end gain ~= fraction_of_latency_spent_in_sparse_prefill * kernel_speedup
```

For example, a 12% kernel speedup becomes roughly a 2.4% TTFT speedup if sparse
prefill is 20% of TTFT. Decode-heavy workloads may show little or no gain.

## What Needs To Be Implemented

### 1. Strict Kernel Dispatch Guard

The optimized kernel must only run when its assumptions hold:

- `topk_length == nullptr`
- sparse indices are known valid for every selected KV position
- `topk` and shape match the specialized path, currently the H100 BF16 sparse
  prefill shape with `d_qk = 576` and full `topk = 512`
- the caller does not use padded, sentinel, or dynamically truncated sparse
  rows

The generic FlashMLA path must remain available for all other cases.

This is the key PR-readiness requirement. The optimization is defensible only if
the dispatch condition prevents it from handling inputs where the removed mask
logic is still needed.

### 2. Source-Build A/B Support

The existing source-build loop already supports a no-op FlashMLA overlay and a
patched FlashMLA overlay. For real-time inference, extend that same A/B pattern
to a vLLM server run:

1. Build and install the no-op FlashMLA source overlay.
2. Run the existing low-level correctness/speed benchmark as a sanity check.
3. Start a vLLM OpenAI-compatible server with the target model.
4. Run the serving benchmark and save results.
5. Stop the server.
6. Build and install the patched FlashMLA overlay.
7. Repeat the same sanity check and serving benchmark.

Both halves must run on the same pod, same GPU type, same model, same vLLM
version, same request corpus, same generation settings, and same concurrency.

### 3. Real Serving Benchmark Client

Add a benchmark client that can hit a running vLLM OpenAI-compatible endpoint.
Suggested file:

```text
tools/bench_vllm_realtime.py
```

Required inputs:

- `--base-url`, for example `http://127.0.0.1:8000/v1`
- `--model`
- `--prompts-file`
- `--endpoint`, either `chat.completions` or `completions`
- `--concurrency`
- `--max-tokens`
- `--temperature`, usually `0`
- `--warmup-requests`
- `--output-json`

Required metrics:

- success and error counts
- input tokens, output tokens, and total tokens when returned by the server
- time to first token, measured from request start to first streamed token
- end-to-end request latency
- inter-token latency distribution for streamed responses
- aggregate output tokens/sec
- request throughput
- p50, p90, p95, and p99 for TTFT and end-to-end latency

The client should use streaming responses. TTFT cannot be measured accurately
from non-streaming responses.

### 4. Prompt Corpus

Add a fixed prompt corpus that stresses the suspected win area. Suggested file:

```text
bench/prompts/realtime_prefill.jsonl
```

Include at least three workload classes:

- long-prefill / short-output: `8k-16k` input tokens, `16-64` output tokens
- interactive mixed: `1k-4k` input tokens, `64-256` output tokens
- decode-heavy control: `256-1k` input tokens, `512+` output tokens

The long-prefill class is where the optimization is most likely to show up in
TTFT. The decode-heavy class is a control case; it should not be expected to
improve much if this is truly a prefill-only optimization.

### 5. RunPod Orchestration

Extend the remote workflow with a real-time mode. This can be added to the
existing source-build loop or as a separate tool.

Suggested options:

```text
--serving-benchmark
--model-path <path-or-hf-id>
--served-model-name <name>
--prompts-file bench/prompts/realtime_prefill.jsonl
--concurrency 1,4,8,16
--max-tokens 64
--server-port 8000
```

The orchestration should save:

- no-op microbenchmark report
- patched microbenchmark report
- no-op serving benchmark report
- patched serving benchmark report
- vLLM server logs for both runs
- environment metadata: GPU, driver, CUDA, vLLM version, FlashMLA source ref,
  patch name, commit SHA, model ID, generation settings

## How To Test

### Local Tests Without CUDA

These tests verify the benchmark harness and orchestration plumbing:

```bash
python3 -m py_compile bench/*.py tools/*.py tests/*.py
python3 -m unittest discover tests
```

If `tools/bench_vllm_realtime.py` is added, include unit tests with a tiny mock
SSE server that emits OpenAI-compatible streaming chunks. Test:

- TTFT is measured from request start to first chunk
- inter-token gaps are recorded correctly
- failed requests are counted and do not corrupt percentiles
- JSON output has a stable schema
- concurrency limit is respected

### Remote H100 Sanity Test

Before claiming serving impact, each A/B run must first prove the low-level path
is installed and still correct:

```bash
python3 -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill
```

Required result:

- no-op source overlay correctness: `PASS`
- patched overlay correctness: `PASS`
- patched microbenchmark is faster than no-op source baseline in the same pod

If the microbenchmark does not reproduce in the serving pod, the serving result
is not attributable to the patch.

### Remote Serving A/B Test

Run the serving matrix on the same pod:

```text
variants: no-op source overlay, patched source overlay
concurrency: 1, 4, 8, 16
workloads: long-prefill, mixed interactive, decode-heavy control
repeats: at least 3 per cell
```

For each cell:

1. Start vLLM.
2. Send warmup requests and discard their metrics.
3. Run the fixed prompt set.
4. Save JSON metrics and server logs.
5. Stop vLLM cleanly before switching variants.

Use the median of repeated runs for the headline number. Keep p95/p99 visible
because serving variance can hide small kernel wins.

### Acceptance Criteria

A real-time inference improvement is credible if all of the following hold:

- patched FlashMLA passes the BF16 sparse prefill correctness benchmark
- same-pod microbenchmark shows a reproducible speedup versus no-op source
- long-prefill TTFT improves in the serving benchmark
- error rate does not increase
- output token counts and finish reasons are comparable
- decode-heavy control does not show unexplained large movement
- p95 latency does not regress even if p50 improves
- results reproduce across at least three same-shape runs

For a PR, the strongest case would include both:

- kernel-level speedup with correctness and dispatch guards
- serving-level TTFT improvement on a real vLLM model path

Without serving evidence, the change can still be proposed as a specialized
kernel optimization, but acceptance odds are lower because maintainers will ask
whether the microbenchmark win transfers to production inference.

## What Would Disprove The Optimization

The optimization should not be claimed as a real-time inference win if:

- TTFT does not improve on long-prefill workloads
- only tokens/sec changes while TTFT does not, without a clear explanation
- the serving benchmark improves but the same-pod microbenchmark does not
- output behavior changes, request errors increase, or p95/p99 latency regresses
- the optimized path is triggered for inputs with invalid indices or dynamic
  `topk_length`

In those cases, the microbenchmark may still be valid, but it is not evidence of
a production serving improvement.

## Implementation Status - 2026-05-22

The realtime harness is implemented on branch
`codex/realtime-serving-benchmark`.

Implemented files:

```text
tools/bench_vllm_realtime.py
bench/prompts/realtime_prefill.jsonl
patches/flashmla/bf16_prefill/sm90_prefill_guarded_no_topklen_assume_valid_sync_order.patch
tests/test_bench_vllm_realtime.py
```

The source-build loop now supports:

```text
--serving-benchmark
--model-path
--served-model-name
--prompts-file
--endpoint
--concurrency
--max-tokens
--server-port
--warmup-requests
--serving-repeats
--temperature
--vllm-arg
--vllm-env
```

`--vllm-env` is intentionally applied only to the vLLM server subprocess. This
keeps the microbenchmark environment unchanged while still allowing server-only
workarounds such as `VLLM_USE_DEEP_GEMM=0`.

Local validation:

```text
python3 -m py_compile bench/*.py tools/*.py tests/*.py
python3 -m unittest discover tests
```

Current local test count is `45`.

Latest H100 serving smoke:

```text
Artifact:
artifacts/evolve_flashmla/evolve-flashmla-20260522-001306/runpod/runpod-shia6syjmvo1ig-20260522-002718/

Model:
Qwen/Qwen2.5-0.5B-Instruct

Serving config:
concurrency=1
serving_repeats=1
warmup_requests=0
server_port=8001
VLLM_USE_DEEP_GEMM=0 for the vLLM server process
```

Results:

```text
source-noop:
  microbenchmark correctness: PASS
  microbenchmark runtime: 23.35 us
  serving reports: produced for decode-heavy, mixed, and long-prefill workloads

sm90_prefill_guarded_no_topklen_assume_valid_sync_order:
  microbenchmark correctness: PASS
  microbenchmark runtime: 23.05 us
  speedup vs source no-op: 1.285%
  serving reports: produced for decode-heavy, mixed, and long-prefill workloads
```

The source loop ended with status `exhausted` because the candidate did not
clear the configured `2%` microbenchmark speedup gate. The serving harness did
run for both variants. The small Qwen smoke model is useful for validating the
orchestration and streaming client, but it is not a production DeepSeek V4
serving claim. One long-prefill row exceeded the Qwen 32k context limit in both
variants, so that smoke should not be used as acceptance evidence for
long-prefill TTFT.
