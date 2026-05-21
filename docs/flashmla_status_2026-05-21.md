# FlashMLA Optimization Status - 2026-05-21

## Goal

The project goal is to make the real DeepSeek V4 production FlashMLA path faster. The current optimization target is vLLM's FlashMLA sparse backend on H100, not Triton comparison kernels and not 3090/4090 proxy paths.

The current correctness-bearing target is H100 BF16 sparse prefill:

```bash
python3 -m bench.run_benchmark --gpu h100 --flashmla-mode bf16-prefill
```

This target calls `flash_mla_sparse_fwd` through:

```text
vllm.v1.attention.backends.mla.flashmla_sparse
```

FP8 decode exists and is closer to the V4 decode/cache production shape, but it is still smoke/speed-only until packed-cache correctness is implemented.

## Current Baselines

Wheel baseline, previously verified on H100:

```text
Mode: H100 BF16 sparse prefill
Runtime: 23.29 us
Correctness: PASS
FlashMLA module: vllm.v1.attention.backends.mla.flashmla_sparse
```

FP8 decode baseline, previously verified on H100:

```text
Mode: H100 FP8 decode
Runtime: 23.01 us
Correctness: NOT CHECKED
Reason: packed FP8 low-level path is smoke/speed-only today
```

Latest source-build no-op baseline:

```text
Pod: ggv776z744sazo
Commit: 95dcac331c36dafbc0d0a5982e0f52b5fa543d02
Runtime: 23.64 us
Correctness: PASS (allclose atol=0.03, rtol=0.03, max_abs=0.00195312)
Drift vs wheel baseline: 1.50% slower
Status: acceptable source-build drift, within 20% guardrail
```

## Source-Build Loop

The project now has a working FlashMLA source-build agent loop:

```bash
python3 tools/evolve_flashmla.py \
  --ref <pushed-git-sha> \
  --baseline-us 23.29 \
  --source-ref v0.21.0 \
  --candidate patches/flashmla/bf16_prefill/<candidate>.patch \
  --terminate-on-complete
```

Primary files:

```text
tools/evolve_flashmla.py
tools/flashmla_source_loop.py
tools/source_build_flashmla.py
tools/runpod_benchmark.py
tools/extract_flashmla.py
```

The loop does the following:

1. Starts a RunPod H100 pod.
2. Installs vLLM wheel/dependencies with the `runpod-vllm-source` profile.
3. Runs local static validation on the remote clone.
4. Builds a no-op FlashMLA source overlay.
5. Runs BF16 sparse prefill correctness and timing.
6. Applies one candidate patch to FlashMLA source.
7. Builds the patched FlashMLA overlay.
8. Runs BF16 sparse prefill correctness and timing again.
9. Accepts a candidate only if correctness passes and speedup vs no-op source baseline is at least `2%`.
10. Deletes the pod when complete and saves artifacts locally.

The source-build path intentionally keeps the installed vLLM wheel as the production package. It builds only FlashMLA extension targets, then overlays these artifacts into the installed wheel:

```text
vllm/_flashmla_C*.so
vllm/_flashmla_extension_C*.so
vllm/third_party/flashmla/flash_mla_interface.py
```

This avoids replacing the entire installed vLLM package with an editable source install. It also fixes the earlier issue where vLLM source build skipped FlashMLA on CUDA 12.8 because vLLM's setup logic only enabled FlashMLA extensions when `nvcc >= 12.9`.

## Validation

Local validation currently passes:

```bash
python3 -m py_compile bench/*.py tools/*.py tests/*.py
python3 -m unittest discover tests
```

Current unit-test count:

```text
36 tests passing
```

Remote validation:

```text
No-op FlashMLA source overlay: correctness PASS
Patched FlashMLA overlay: correctness gate works
Artifact collection: working
Pod cleanup: working
```

## Candidate Results

### `sm90_btopk128`

Patch:

```text
patches/flashmla/bf16_prefill/sm90_btopk128.patch
```

Intent:

```text
Change SM90 sparse prefill B_TOPK from 64 to 128 only for D_QK == 576 and no topk_length.
```

Result:

```text
Build: PASS
Runtime: FAIL
Error: CUDA invalid argument in phase1.cuh:626
Correctness: not reached
Status: rejected
```

Likely cause:

```text
Increasing B_TOPK increases shared-memory requirements. The failure occurs at/near cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size), so the patched kernel likely exceeds launchable dynamic shared memory for this configuration.
```

### `sm90_prefill_evict_first`

Patch:

```text
patches/flashmla/bf16_prefill/sm90_prefill_evict_first.patch
```

Intent:

```text
Change producer KV cp.async L2 cache policy from evict_last to evict_first for one-pass sparse top-k loads.
```

Latest result:

```text
Pod: ggv776z744sazo
Commit: 95dcac331c36dafbc0d0a5982e0f52b5fa543d02
No-op runtime: 23.64 us
Candidate runtime: 23.62 us
Correctness: PASS
Speedup vs source no-op: 0.085%
Speedup vs wheel baseline: -1.42%
Status: rejected, below 2% acceptance threshold
```

This candidate is correctness-preserving but not materially faster.

## Important Artifacts

Latest complete candidate run:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-153150/runpod/runpod-ggv776z744sazo-20260521-154312/
```

Most important file:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-153150/runpod/runpod-ggv776z744sazo-20260521-154312/candidate_summary.json
```

Useful supporting artifacts:

```text
output.log
report.json
build_source-noop.log
build_sm90_prefill_evict_first.log
source_provenance_source-noop.json
source_provenance_sm90_prefill_evict_first.json
source_overlay_source-noop.json
source_overlay_sm90_prefill_evict_first.json
flashmla_extraction_source-noop.json
flashmla_extraction_sm90_prefill_evict_first.json
```

Earlier source-build infrastructure runs:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-041543/
artifacts/evolve_flashmla/evolve-flashmla-20260521-043844/
artifacts/evolve_flashmla/evolve-flashmla-20260521-044639/
artifacts/evolve_flashmla/evolve-flashmla-20260521-051242/
```

## Known Issues

The loop works, but it is still too slow for high-throughput agent search.

Current pain points:

```text
Each remote run rebuilds the no-op source overlay.
Each candidate reclones vLLM and FlashMLA.
Each candidate reruns full CMake configure.
The CMake configure still processes the broader vLLM tree even though the build target is FlashMLA-only.
```

The latest successful run took about 11 minutes wall-clock for no-op plus one candidate. Earlier runs were longer when the loop did full editable vLLM install.

There is also still no FP8 packed-cache correctness gate. That means BF16 sparse prefill is the only correctness-bearing optimization target today, even though FP8 decode is important for the final V4 production path.

## Next Engineering Steps

Recommended next loop-infrastructure work:

1. Reuse one cloned vLLM/FlashMLA source tree inside a pod.
2. Build no-op once, then reset/apply candidates in the same source tree.
3. Reuse CMake build directories where possible.
4. Avoid rerunning no-op for every single candidate when multiple candidates are evaluated in one pod.
5. Add `--max-candidates > 1` workflow that actually amortizes setup/build cost.
6. Add clearer patch validation before remote execution, including `git apply --check` against the pinned FlashMLA commit.

Recommended next kernel-candidate directions:

1. Avoid changing `B_TOPK` upward without first computing shared-memory size and launch limits.
2. Look for lower-risk changes inside the producer warpgroup: load ordering, prefetch distance, cache policy by tile range, or validity-mask timing.
3. Consider specializing only the D_QK=576/no-topk-length instantiation without increasing shared memory.
4. Investigate whether the SM100 sources contain useful scheduling ideas that can be safely backported to SM90 prefill.
5. Add instrumentation or static reporting for `sizeof(SharedMemoryPlan)` by candidate to reject impossible kernels locally before remote runtime.

Recommended correctness work:

```text
Implement FP8 decode correctness for the packed-cache layout so the loop can optimize the more production-relevant V4 decode path, not only BF16 prefill.
```

## Current Bottom Line

The source-build FlashMLA optimization loop is functional and validated on H100. The current best source-built kernel remains the no-op FlashMLA overlay. No candidate has beaten the acceptance threshold yet.

The immediate bottleneck is no longer missing infrastructure. The next gains require better FlashMLA kernel candidates and faster candidate iteration inside one pod.
