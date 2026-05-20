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
    "vllm.attention.ops.flashmla",
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
    imported: list[FlashMLASymbols] = []
    for module_name in FLASHMLA_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - environment probe.
            errors.append(f"{module_name}: {exc!r}")
            continue

        symbols = FlashMLASymbols(
            flash_mla_sparse_fwd=getattr(module, "flash_mla_sparse_fwd", None),
            flash_mla_with_kvcache=getattr(module, "flash_mla_with_kvcache", None),
            get_mla_metadata=getattr(module, "get_mla_metadata", None),
            module_name=module_name,
            module=module,
        )
        imported.append(symbols)
        if any(
            symbol is not None
            for symbol in (
                symbols.flash_mla_sparse_fwd,
                symbols.flash_mla_with_kvcache,
                symbols.get_mla_metadata,
            )
        ):
            return symbols

    if imported:
        return imported[0]

    raise RuntimeError(
        "Could not import FlashMLA symbols from vLLM. Run "
        "`python -m tools.extract_flashmla --out-dir artifacts` first.\n"
        + "\n".join(errors)
    )


def flashmla_support_checker(module: ModuleType) -> Any | None:
    for name in (
        "is_flashmla_sparse_supported",
        "is_flashmla_supported",
        "is_flashmla_dense_supported",
    ):
        checker = getattr(module, name, None)
        if checker is not None:
            return checker
    return None


def flashmla_support_status(module: ModuleType) -> tuple[bool, str | None]:
    checker = flashmla_support_checker(module)
    errors: list[str] = []
    if checker is None:
        for module_name in FLASHMLA_MODULE_CANDIDATES:
            try:
                candidate = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - environment probe.
                errors.append(f"{module_name}: {exc!r}")
                continue
            checker = flashmla_support_checker(candidate)
            if checker is not None:
                break

    if checker is None:
        details = "" if not errors else "; " + "; ".join(errors)
        return False, "no FlashMLA support checker found" + details
    try:
        return normalize_support_flag(checker())
    except Exception as exc:  # noqa: BLE001 - support probes should be reportable.
        checker_name = getattr(checker, "__name__", type(checker).__name__)
        checker_module = getattr(checker, "__module__", module.__name__)
        return False, f"{checker_module}.{checker_name} raised {exc!r}"


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
