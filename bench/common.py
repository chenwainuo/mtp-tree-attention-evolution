"""Shared benchmark helpers with no third-party imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RTX_3090_PEAK_BANDWIDTH_GB_S = 936.0
RTX_4090_PEAK_BANDWIDTH_GB_S = 1008.0
H100_PEAK_BANDWIDTH_GB_S = 3350.0


@dataclass(frozen=True)
class TargetDefaults:
    gpu: str
    path: str
    dtype: str
    bandwidth_gbs: float
    uses_fp8_kv: bool
    requires_flashmla: bool


TARGET_DEFAULTS: dict[str, TargetDefaults] = {
    "3090": TargetDefaults(
        gpu="3090",
        path="dense-flashinfer-fp16-proxy",
        dtype="float16",
        bandwidth_gbs=RTX_3090_PEAK_BANDWIDTH_GB_S,
        uses_fp8_kv=False,
        requires_flashmla=False,
    ),
    "4090": TargetDefaults(
        gpu="4090",
        path="dense-flashinfer-fp8-kv-proxy",
        dtype="float16",
        bandwidth_gbs=RTX_4090_PEAK_BANDWIDTH_GB_S,
        uses_fp8_kv=True,
        requires_flashmla=False,
    ),
    "h100": TargetDefaults(
        gpu="h100",
        path="flashmla-sparse",
        dtype="bfloat16",
        bandwidth_gbs=H100_PEAK_BANDWIDTH_GB_S,
        uses_fp8_kv=True,
        requires_flashmla=True,
    ),
}


def require_target(gpu: str) -> TargetDefaults:
    try:
        return TARGET_DEFAULTS[gpu]
    except KeyError as exc:
        raise ValueError(f"unknown GPU target {gpu!r}") from exc


def estimate_memory_roofline_us(total_bytes: int, peak_bandwidth_gb_s: float) -> float:
    return total_bytes / (peak_bandwidth_gb_s * 1e9) * 1e6


def normalize_support_flag(value: Any) -> tuple[bool, str | None]:
    """Normalize bool or tuple-style vLLM support-return values.

    vLLM helpers have used both `bool` and `(bool, reason)` shapes. The adapter
    treats unknown truthy objects as supported but records no reason.
    """
    if isinstance(value, tuple):
        if not value:
            return False, "empty support tuple"
        supported = bool(value[0])
        reason = None if len(value) < 2 or value[1] is None else str(value[1])
        return supported, reason
    if isinstance(value, bool):
        return value, None
    return bool(value), None


def format_pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def dry_run_lines(
    *,
    gpu: str,
    path: str,
    batch: int,
    ctx_len: int,
    k_draft: int,
    dtype: str,
    bandwidth_gbs: float,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    lines = [
        "Benchmark dry run",
        f"  target_gpu: {gpu}",
        f"  path: {path}",
        f"  batch: {batch}",
        f"  ctx_len: {ctx_len}",
        f"  K_draft: {k_draft}",
        f"  dtype: {dtype}",
        f"  bandwidth_gbs: {bandwidth_gbs:.1f}",
    ]
    for key, value in (extra or {}).items():
        lines.append(f"  {key}: {value}")
    return lines

