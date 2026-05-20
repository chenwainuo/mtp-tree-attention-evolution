# Source Notes

These are the current primary-source anchors for the benchmark scaffold.

## DeepSeek V4-Flash Config

Official Hugging Face config:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json

Extracted defaults:

- `hidden_size = 4096`
- `num_attention_heads = 64`
- `num_key_value_heads = 1`
- `head_dim = 512`
- `num_nextn_predict_layers = 1`
- `max_position_embeddings = 1048576`
- `sliding_window = 128`
- `rms_norm_eps = 1e-6`
- attention quantization config reports FP8 `e4m3`
- instruct model `expert_dtype = fp4`; base model reports `expert_dtype = fp8`

## vLLM V4 Attention Path

Relevant vLLM PRs:

- https://github.com/vllm-project/vllm/pull/40871
- https://github.com/vllm-project/vllm/pull/41136
- https://github.com/vllm-project/vllm/pull/41312

Current vLLM source anchors:

- `vllm/model_executor/layers/deepseek_v4_attention.py`
- `vllm/v1/attention/ops/flashmla.py`
- `vllm/v1/attention/backends/mla/flashmla_sparse.py`
- `vllm/attention/ops/flashmla.py`

Important implementation detail: current vLLM DeepSeek V4 attention routes through
FlashMLA sparse attention for the production path, not plain dense FlashInfer
prefill. vLLM's FlashMLA sparse availability check is exposed from different ops
modules across releases and says the kernel is a Hopper/Blackwell path, so the
3090/4090 path uses a proxy benchmark. Current V4 packed FP8 KV cache entries
are 584 bytes per token.

FlashMLA extraction artifacts:

- `tools/extract_flashmla.py`
- `bench/bench_flashmla_sparse.py`
- `docs/flashmla_extraction.md`

The extraction script must be run on the remote CUDA host after installing vLLM.
It records the exact installed module paths, signatures, support flags, and
source excerpts so the benchmark can be patched to the real API if upstream has
changed.

## FlashInfer Custom-Mask Prefill

FlashInfer API docs:

- https://docs.flashinfer.ai/api/attention.html
- https://docs.flashinfer.ai/generated/flashinfer.prefill.single_prefill_with_kv_cache_return_lse.html

FlashInfer test anchor:

- https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_batch_prefill_kernels.py

The Stage 1 proxy starts with `BatchPrefillWithRaggedKVCacheWrapper` plus a
custom boolean mask for chain MTP. The 3090 target uses FP16 K/V. The 4090 target
uses FP8 K/V with explicit scales and validates against dequantized K/V. Neither
proxy is the full DeepSeek V4 FlashMLA sparse metadata path. The dense proxy
defaults to `index_head_dim = 128` instead of the V4 MLA latent `head_dim = 512`,
because the plain dense FlashInfer ragged-prefill kernel rejects the 512-wide
proxy configuration on 3090.
