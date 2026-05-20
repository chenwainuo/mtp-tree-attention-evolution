"""DeepSeek V4-Flash benchmark shape extraction.

This module intentionally has no third-party dependencies. It can run on a
non-GPU machine, which lets us prepare benchmark code before the CUDA box is
available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "deepseek-v4-flash-config.json"

# Snapshot from:
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/config.json
# Keep this as a fallback so benchmarks remain runnable after cloning, even
# before `huggingface-cli download ... config.json` has been run.
DEFAULT_DEEPSEEK_V4_FLASH_CONFIG: dict[str, Any] = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "expert_dtype": "fp4",
    "hc_eps": 1e-6,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "index_head_dim": 128,
    "index_n_heads": 64,
    "index_topk": 512,
    "initializer_range": 0.02,
    "max_position_embeddings": 1048576,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 1,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-6,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    },
    "rope_theta": 10000,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "torch_dtype": "bfloat16",
    "transformers_version": "4.57.1",
    "use_cache": True,
    "vocab_size": 129280,
    "compress_rope_theta": 160000,
    "compress_ratios": [
        0,
        0,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        128,
        4,
        0,
    ],
}


@dataclass(frozen=True)
class V4FlashShapes:
    model_id: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    max_position_embeddings: int
    sliding_window: int
    qk_rope_head_dim: int
    index_head_dim: int
    index_n_heads: int
    index_topk: int
    compress_ratios: tuple[int, ...]
    rms_norm_eps: float
    expert_dtype: str
    attention_quant_method: str
    attention_quant_format: str
    torch_dtype: str

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return DEFAULT_DEEPSEEK_V4_FLASH_CONFIG


def extract_shapes(config: dict[str, Any]) -> V4FlashShapes:
    quant = config.get("quantization_config", {})
    return V4FlashShapes(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        hidden_size=int(config["hidden_size"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=int(config["head_dim"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_nextn_predict_layers=int(config.get("num_nextn_predict_layers", 1)),
        max_position_embeddings=int(config["max_position_embeddings"]),
        sliding_window=int(config["sliding_window"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        index_head_dim=int(config["index_head_dim"]),
        index_n_heads=int(config["index_n_heads"]),
        index_topk=int(config["index_topk"]),
        compress_ratios=tuple(int(x) for x in config["compress_ratios"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        expert_dtype=str(config.get("expert_dtype", "unknown")),
        attention_quant_method=str(quant.get("quant_method", "unknown")),
        attention_quant_format=str(quant.get("fmt", "unknown")),
        torch_dtype=str(config.get("torch_dtype", "unknown")),
    )


def get_v4_flash_shapes(path: Path = DEFAULT_CONFIG_PATH) -> V4FlashShapes:
    return extract_shapes(load_config(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    shapes = get_v4_flash_shapes(args.config)
    if args.json:
        print(json.dumps(asdict(shapes), indent=2))
        return

    print("DeepSeek V4-Flash shapes")
    print(f"  model_id: {shapes.model_id}")
    print(f"  hidden_size: {shapes.hidden_size}")
    print(
        "  attention heads: "
        f"{shapes.num_attention_heads} query / "
        f"{shapes.num_key_value_heads} KV "
        f"(GQA group size {shapes.gqa_group_size})"
    )
    print(f"  head_dim: {shapes.head_dim}")
    print(f"  layers: {shapes.num_hidden_layers}")
    print(f"  MTP next-token layers: {shapes.num_nextn_predict_layers}")
    print(f"  max_position_embeddings: {shapes.max_position_embeddings}")
    print(f"  sliding_window: {shapes.sliding_window}")
    print(
        "  attention quantization: "
        f"{shapes.attention_quant_method}/{shapes.attention_quant_format}"
    )
    print(f"  expert_dtype: {shapes.expert_dtype}")


if __name__ == "__main__":
    main()

