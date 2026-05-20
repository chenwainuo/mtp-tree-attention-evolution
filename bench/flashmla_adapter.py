"""Signature-adaptive vLLM FlashMLA adapter helpers."""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from bench.common import normalize_support_flag


FLASHMLA_MODULE_CANDIDATES = (
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.v1.attention.ops.flashmla",
)


@dataclass(frozen=True)
class FlashMLASymbols:
    flash_mla_sparse_fwd: Any | None
    flash_mla_with_kvcache: Any | None
    get_mla_metadata: Any | None
    module_name: str
    module: ModuleType


@dataclass(frozen=True)
class FlashMLAExtraction:
    path: Path
    report: dict[str, Any]

    @property
    def support_flags(self) -> dict[str, Any]:
        return dict(self.report.get("support_flags", {}))


def load_extraction_report(path: Path) -> FlashMLAExtraction:
    return FlashMLAExtraction(path=path, report=json.loads(path.read_text()))


def import_flashmla_symbols() -> FlashMLASymbols:
    errors: list[str] = []
    for module_name in FLASHMLA_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - environment probe.
            errors.append(f"{module_name}: {exc!r}")
            continue

        return FlashMLASymbols(
            flash_mla_sparse_fwd=getattr(module, "flash_mla_sparse_fwd", None),
            flash_mla_with_kvcache=getattr(module, "flash_mla_with_kvcache", None),
            get_mla_metadata=getattr(module, "get_mla_metadata", None),
            module_name=module_name,
            module=module,
        )

    raise RuntimeError(
        "Could not import FlashMLA symbols from vLLM. Run "
        "`python -m tools.extract_flashmla --out-dir artifacts` first.\n"
        + "\n".join(errors)
    )


def flashmla_support_status(module: ModuleType) -> tuple[bool, str | None]:
    checker = getattr(module, "is_flashmla_sparse_supported", None)
    if checker is None:
        checker = getattr(module, "is_flashmla_supported", None)
    if checker is None:
        return False, "no FlashMLA support checker found"
    return normalize_support_flag(checker())


def assert_hopper_or_blackwell(torch: Any) -> None:
    capability = torch.cuda.get_device_capability()
    major = int(capability[0])
    if major < 9:
        raise SystemExit(
            "FlashMLA sparse is a Hopper/Blackwell path. "
            f"Detected SM{capability[0]}{capability[1]}; use --gpu 3090/4090 "
            "proxy paths on Ampere/Ada."
        )


def signature_accepts(fn: Any, required: set[str]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(param.kind == param.VAR_KEYWORD for param in params.values()):
        return True
    return required.issubset(params.keys())


def ensure_sparse_prefill_signature(fn: Any) -> None:
    try:
        arity = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return
    if arity < 4:
        raise RuntimeError(
            "`flash_mla_sparse_fwd` signature is not supported by this scaffold; "
            f"expected at least 4 parameters, got {arity}"
        )


def ensure_decode_signature(fn: Any) -> None:
    try:
        arity = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        return
    if arity < 10:
        raise RuntimeError(
            "`flash_mla_with_kvcache` signature is not supported by this scaffold; "
            f"expected at least 10 parameters, got {arity}"
        )

