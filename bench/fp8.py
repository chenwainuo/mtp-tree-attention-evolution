"""FP8 helper functions.

The scalar helpers are dependency-free for local tests. Runtime tensor
quantization imports torch lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E4M3_MAX = 448.0


@dataclass(frozen=True)
class Fp8TensorPair:
    quantized: Any
    dequantized: Any
    scale: float


def scale_from_amax(amax: float, fp8_max: float = E4M3_MAX) -> float:
    if not (amax >= 0.0):
        raise ValueError(f"amax must be finite and non-negative, got {amax!r}")
    if amax == 0.0:
        return 1.0
    scale = amax / fp8_max
    if not (scale > 0.0):
        raise ValueError(f"computed invalid FP8 scale {scale!r}")
    return scale


def dequantize_scalar(q_value: float, scale: float) -> float:
    if not (scale > 0.0):
        raise ValueError(f"scale must be positive, got {scale!r}")
    return q_value * scale


def quantize_fp8_tensor(torch: Any, tensor: Any) -> Fp8TensorPair:
    """Quantize a tensor to e4m3 and return quantized + dequantized views.

    Scale convention: `quantized = clamp(tensor / scale).to(float8_e4m3fn)`;
    dequantized reference uses `quantized.float() * scale`.
    """
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required for the 4090 FP8 proxy")

    amax_tensor = tensor.float().abs().max()
    amax = float(amax_tensor.item())
    scale = scale_from_amax(amax)
    scaled = torch.clamp(tensor.float() / scale, min=-E4M3_MAX, max=E4M3_MAX)
    quantized = scaled.to(torch.float8_e4m3fn)
    dequantized = (quantized.float() * scale).to(tensor.dtype)

    if not torch.isfinite(dequantized.float()).all():
        raise RuntimeError("non-finite value produced by FP8 quant/dequant")
    return Fp8TensorPair(quantized=quantized, dequantized=dequantized, scale=scale)

