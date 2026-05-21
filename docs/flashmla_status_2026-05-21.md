# FlashMLA Optimization Status - 2026-05-21

## Goal

The project goal is to make the real DeepSeek V4 production FlashMLA path faster. The current optimization target is vLLM's FlashMLA sparse backend on H100, not Triton comparison kernels and not 3090/4090 proxy paths.

After accepting the initial 2% source-build candidate, the active stretch target is now a correctness-passing 15% speedup vs the H100 source-build no-op baseline.

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
Pod: p14bcc3sym0mag
Commit: 0ebbf426af05837dc718b2e888502d6c8a49e39b
Runtime: 23.46 us
Correctness: PASS (allclose atol=0.03, rtol=0.03, max_abs=0.00195312)
Drift vs wheel baseline: 0.73% slower
Status: acceptable source-build drift, within 20% guardrail
```

Current accepted source-build candidate:

```text
Patch: patches/flashmla/bf16_prefill/sm90_prefill_single_mask_wait.patch
Pod: p14bcc3sym0mag
Commit: 0ebbf426af05837dc718b2e888502d6c8a49e39b
Runtime: 22.90 us
Correctness: PASS (allclose atol=0.03, rtol=0.03, max_abs=0.00195312)
Speedup vs source no-op: 2.387%
Speedup vs wheel baseline: 1.675%
Status: accepted, clears 2% source-build gate
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

### `sm90_prefill_single_mask_wait`

Patch:

```text
patches/flashmla/bf16_prefill/sm90_prefill_single_mask_wait.patch
```

Intent:

```text
Remove a redundant bar_is_kv_valid_ready wait from online_softmax_and_rescale_o.
Both warpgroup call sites already wait for the same validity-mask phase in mask_rP immediately before online softmax, so this keeps the same mask producer/consumer ordering while removing one duplicate wait from the QK/softmax path.
```

Latest result:

```text
Pod: p14bcc3sym0mag
Commit: 0ebbf426af05837dc718b2e888502d6c8a49e39b
No-op runtime: 23.46 us
Candidate runtime: 22.90 us
Correctness: PASS
Speedup vs source no-op: 2.387%
Speedup vs wheel baseline: 1.675%
Status: accepted, clears 2% source-build gate
```

The candidate batch also added these untested follow-up patches because the loop stopped early after the first accepted candidate:

```text
patches/flashmla/bf16_prefill/sm90_prefill_packed_valid_mask.patch
patches/flashmla/bf16_prefill/sm90_prefill_packed_mask_single_wait.patch
patches/flashmla/bf16_prefill/sm90_prefill_unroll4_topk_loop.patch
patches/flashmla/bf16_prefill/sm90_prefill_wg0_first_loads.patch
patches/flashmla/bf16_prefill/sm90_prefill_regs224.patch
patches/flashmla/bf16_prefill/sm90_prefill_sync_order_regs.patch
```

### `sm90_prefill_static_topk512_single_wait_unroll4`

Patch:

```text
patches/flashmla/bf16_prefill/sm90_prefill_static_topk512_single_wait_unroll4.patch
```

Intent:

```text
Specialize the hot D_QK=576/no-topk-length/topk=512 dispatch path with a static 8-block top-k loop, keep the generic fallback for other top-k values, and combine that with the accepted duplicate-mask-wait removal plus top-k loop unroll hints.
```

Local validation:

```text
git apply --check against pinned FlashMLA source: PASS
Python compile: PASS
Unit tests: PASS (36 tests)
```

Remote result:

```text
Correctness: PASS
Runtime: 44.41 us
Speedup vs source no-op: -87.384%
Speedup vs wheel baseline: -90.683%
Status: rejected, much slower than source no-op
```

### 15% Stretch Sweep After RunPod Credit Refill

Command:

```text
python3 tools/evolve_flashmla.py \
  --ref 0db0fcfa91e3ee03478ba89baf685fb0882938c2 \
  --baseline-us 23.29 \
  --source-ref v0.21.0 \
  --min-speedup-pct 15 \
  --max-jobs 8 \
  --max-candidates 6 \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_static_topk512_single_wait_unroll4.patch \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_packed_mask_single_wait.patch \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_unroll4_topk_loop.patch \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_wg0_first_loads.patch \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_regs224.patch \
  --candidate patches/flashmla/bf16_prefill/sm90_prefill_sync_order_regs.patch \
  --terminate-on-complete
```

Remote result summary:

```text
source-noop: 23.70 us, PASS
sm90_prefill_static_topk512_single_wait_unroll4: 44.41 us, PASS, -87.384% vs source
sm90_prefill_packed_mask_single_wait: 23.16 us, PASS, +2.278% vs source
sm90_prefill_unroll4_topk_loop: 46.26 us, PASS, -95.190% vs source
sm90_prefill_wg0_first_loads: 23.39 us, PASS, +1.308% vs source
sm90_prefill_regs224: 23.45 us, PASS, +1.055% vs source
sm90_prefill_sync_order_regs: 22.74 us, PASS, +4.051% vs source, +2.362% vs wheel
Status: exhausted, no candidate reached 15%
Best new candidate: sm90_prefill_sync_order_regs
```

### No-Topk-Length Mask Elimination Sweeps

The benchmark shape has `topk_length == nullptr` and generated sparse indices are valid. Two follow-up sweeps tested whether the no-topk-length instantiation can skip validity masking work while preserving the `HAVE_TOPK_LENGTH` path.

First no-mask sweep:

```text
Artifact:
artifacts/evolve_flashmla/evolve-flashmla-20260521-204511/runpod/runpod-c2g60k74mopz6f-20260521-211510/candidate_summary.json

source-noop: 28.19 us, PASS
sm90_prefill_no_topklen_nomask_single_wait: 25.68 us, PASS, +8.904% vs source
sm90_prefill_no_topklen_assume_valid: 25.96 us, PASS, +7.911% vs source
sm90_prefill_no_topklen_assume_valid_sync_order: 24.85 us, PASS, +11.848% vs source
sm90_prefill_sync_order_regs: 27.67 us, PASS, +1.845% vs source
Status: exhausted, no candidate reached 15%
```

Follow-up no-mask plus sync-order sweep:

```text
Artifact:
artifacts/evolve_flashmla/evolve-flashmla-20260521-211706/runpod/runpod-590ly0ews6tsoy-20260521-213304/candidate_summary.json

source-noop: 28.14 us, PASS
sm90_prefill_no_topklen_nomask_sync_order: 24.98 us, PASS, +11.230% vs source
sm90_prefill_no_topklen_assume_valid_sync_order: 24.84 us, PASS, +11.727% vs source
sm90_prefill_no_topklen_nomask_single_wait: 25.77 us, PASS, +8.422% vs source
Status: exhausted, no candidate reached 15%
```

These two sweeps ran on slow H100 hosts whose source no-op runtime was about 28 us, so their absolute runtimes are not comparable to the earlier 22.74 us best absolute run. The useful signal is the in-run relative delta: removing no-topk-length mask work is real and repeatable, reaching about 11.7-11.8% relative speedup, but still below the 15% stretch target.

## Important Artifacts

Latest complete candidate run:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-211706/runpod/runpod-590ly0ews6tsoy-20260521-213304/
```

Most important file:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-211706/runpod/runpod-590ly0ews6tsoy-20260521-213304/candidate_summary.json
```

Useful supporting artifacts:

```text
output.log
report.json
build_source-noop.log
build_sm90_prefill_single_mask_wait.log
source_provenance_source-noop.json
source_provenance_sm90_prefill_single_mask_wait.json
source_overlay_source-noop.json
source_overlay_sm90_prefill_single_mask_wait.json
flashmla_extraction_source-noop.json
flashmla_extraction_sm90_prefill_single_mask_wait.json
```

Earlier source-build infrastructure runs:

```text
artifacts/evolve_flashmla/evolve-flashmla-20260521-041543/
artifacts/evolve_flashmla/evolve-flashmla-20260521-043844/
artifacts/evolve_flashmla/evolve-flashmla-20260521-044639/
artifacts/evolve_flashmla/evolve-flashmla-20260521-051242/
artifacts/evolve_flashmla/evolve-flashmla-20260521-153150/
```

## Known Issues

The loop works, but it is still too slow for high-throughput agent search.

Current pain points:

```text
Each remote run rebuilds the no-op source overlay.
Candidate sweeps now reuse one vLLM/FlashMLA source tree and CMake build dir.
Incremental candidate rebuilds compile 5 FlashMLA objects instead of 29 after the cold no-op build.
The CMake configure still processes the broader vLLM tree even though the build target is FlashMLA-only.
```

The latest reuse-tree sweep completed no-op plus three candidates in about 15 minutes wall-clock. Earlier candidate loops were slower because every candidate recloned vLLM and FlashMLA.

There is also still no FP8 packed-cache correctness gate. That means BF16 sparse prefill is the only correctness-bearing optimization target today, even though FP8 decode is important for the final V4 production path.

During the 15% stretch attempt, several RunPod source-loop pods exited before the local poller could collect terminal artifacts. `tools/runpod_benchmark.py` now keeps the remote Python worker alive for a configurable terminal-report hold window after writing a succeeded/failed report, instead of relying only on the trailing shell sleep. The fix is committed in:

```text
9e9882b1b2e7dbc3f88ee5c24b73241b30028df5
```

There are no active leftover pods after the latest sweeps.

## Next Engineering Steps

Recommended next loop-infrastructure work:

1. Tighten the vLLM CMake patch further so the configure step stops generating unrelated Marlin/Machete/MOE extension metadata when `MTP_FLASHMLA_ONLY_BUILD=1`.
2. Add a no-op rerun after each candidate or randomized candidate order when a host source no-op drifts above 20%, so relative speedups are less sensitive to host variance.
3. Add clearer patch validation before remote execution, including `git apply --check` against the pinned FlashMLA commit.
4. Improve resilience/report recovery for pods whose artifact HTTP endpoint returns 404 mid-run; the 200-rep confirmation attempt for `sm90_prefill_single_mask_wait` lost its report endpoint before producing usable timing and was manually terminated.

Recommended next kernel-candidate directions:

1. Avoid changing `B_TOPK` upward without first computing shared-memory size and launch limits.
2. Treat loop unrolling and static top-k specialization as poor candidates for this kernel; both were correct but roughly 2x slower in the 15% sweep.
3. No-topk-length mask elimination is useful but insufficient by itself; the best repeatable relative speedup is about 11.8%, still short of 15%.
4. Investigate whether the SM100 sources contain useful scheduling ideas that can be safely backported to SM90 prefill.
5. Add instrumentation or static reporting for `sizeof(SharedMemoryPlan)` by candidate to reject impossible kernels locally before remote runtime.

Recommended correctness work:

```text
Implement FP8 decode correctness for the packed-cache layout so the loop can optimize the more production-relevant V4 decode path, not only BF16 prefill.
```

## Current Bottom Line

The source-build FlashMLA optimization loop is functional and validated on H100. The best absolute runtime is still `sm90_prefill_sync_order_regs` at 22.74 us from the normal-speed sweep. The best relative no-topk-length candidate is `sm90_prefill_no_topklen_assume_valid_sync_order`, which reached 11.7-11.8% in-run speedup on two slow-host sweeps.

The immediate kernel bottleneck is now beyond mask/barrier cleanup; reaching 15% likely needs a scheduling or work-reduction change deeper than local validity-mask removal.
