"""Extract installed vLLM FlashMLA source details on a remote CUDA host.

Run this after installing the target vLLM/FlashMLA stack. It does not benchmark;
it records which modules, symbols, signatures, support flags, and source files
are actually present. That prevents us from writing benchmarks against a stale
or imagined API.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from bench.common import normalize_support_flag


DEFAULT_MODULES = (
    "vllm.model_executor.layers.deepseek_v4_attention",
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.v1.attention.ops.flashmla",
    "vllm.attention.ops.flashmla",
)

DEFAULT_SYMBOLS = {
    "vllm.model_executor.layers.deepseek_v4_attention": (
        "DeepseekV4MLAAttention",
        "DeepseekV4MultiHeadLatentAttentionWrapper",
        "DeepseekV4Attention",
    ),
    "vllm.v1.attention.backends.mla.flashmla_sparse": (
        "FlashMLASparseBackend",
        "FlashMLASparseImpl",
        "FlashMLASparseMetadata",
        "FlashMLASparseMetadataBuilder",
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
        "is_flashmla_sparse_supported",
    ),
    "vllm.v1.attention.ops.flashmla": (
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
        "get_flashmla_version",
        "is_flashmla_supported",
        "is_flashmla_dense_supported",
        "is_flashmla_sparse_supported",
        "is_flashmla_v2",
    ),
    "vllm.attention.ops.flashmla": (
        "flash_mla_sparse_fwd",
        "flash_mla_with_kvcache",
        "get_mla_metadata",
        "get_flashmla_version",
        "is_flashmla_supported",
        "is_flashmla_dense_supported",
        "is_flashmla_sparse_supported",
        "is_flashmla_v2",
    ),
}

SUPPORT_MODULES = (
    "vllm.v1.attention.backends.mla.flashmla_sparse",
    "vllm.v1.attention.ops.flashmla",
    "vllm.attention.ops.flashmla",
)

SUPPORT_SYMBOLS = (
    "is_flashmla_sparse_supported",
    "is_flashmla_dense_supported",
    "is_flashmla_supported",
    "is_flashmla_v2",
    "get_flashmla_version",
)


@dataclass(frozen=True)
class SymbolReport:
    name: str
    present: bool
    kind: str | None = None
    signature: str | None = None
    source_file: str | None = None
    source_start_line: int | None = None
    source_excerpt: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ModuleReport:
    name: str
    present: bool
    file: str | None
    symbols: list[SymbolReport]
    error: str | None = None


def version_of(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def safe_import(module_name: str) -> tuple[ModuleType | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001 - this is an environment probe.
        return None, repr(exc)


def source_excerpt(obj: Any, max_lines: int) -> tuple[str | None, int | None, str | None]:
    try:
        lines, start_line = inspect.getsourcelines(obj)
    except Exception as exc:  # noqa: BLE001
        return None, None, repr(exc)
    excerpt = "".join(lines[:max_lines])
    return excerpt, start_line, None


def symbol_report(module: ModuleType, name: str, max_lines: int) -> SymbolReport:
    if not hasattr(module, name):
        return SymbolReport(name=name, present=False)

    obj = getattr(module, name)
    try:
        signature = str(inspect.signature(obj))
    except Exception:
        signature = None

    try:
        source_file = inspect.getsourcefile(obj) or inspect.getfile(obj)
    except Exception:
        source_file = None

    excerpt, start_line, error = source_excerpt(obj, max_lines)
    return SymbolReport(
        name=name,
        present=True,
        kind=type(obj).__name__,
        signature=signature,
        source_file=source_file,
        source_start_line=start_line,
        source_excerpt=excerpt,
        error=error,
    )


def module_report(module_name: str, max_lines: int) -> ModuleReport:
    module, error = safe_import(module_name)
    if module is None:
        return ModuleReport(
            name=module_name,
            present=False,
            file=None,
            symbols=[],
            error=error,
        )

    symbols = [
        symbol_report(module, symbol_name, max_lines)
        for symbol_name in DEFAULT_SYMBOLS.get(module_name, ())
    ]
    return ModuleReport(
        name=module_name,
        present=True,
        file=getattr(module, "__file__", None),
        symbols=symbols,
    )


def collect_support_flags() -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for module_name in SUPPORT_MODULES:
        module, error = safe_import(module_name)
        if module is None:
            flags[f"{module_name}.__import__"] = {
                "present": False,
                "error": error,
            }
            continue

        for name in SUPPORT_SYMBOLS:
            key = f"{module_name}.{name}"
            if not hasattr(module, name):
                flags[key] = {"present": False}
                continue
            try:
                raw_value = getattr(module, name)()
                if name.startswith("is_"):
                    supported, reason = normalize_support_flag(raw_value)
                    flags[key] = {
                        "present": True,
                        "value": supported,
                        "reason": reason,
                    }
                else:
                    flags[key] = {"present": True, "value": raw_value}
            except Exception as exc:  # noqa: BLE001
                flags[key] = {"present": True, "error": repr(exc)}

    return flags


def collect_torch_cuda() -> dict[str, Any]:
    torch, error = safe_import("torch")
    if torch is None:
        return {"torch_import_error": error}

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return {"torch_version": getattr(torch, "__version__", None), "cuda": None}

    result: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_available": bool(cuda.is_available()),
    }
    if result["cuda_available"]:
        result["device_name"] = cuda.get_device_name()
        result["device_capability"] = cuda.get_device_capability()
    return result


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = ["# FlashMLA Extraction Report", ""]
    lines.append("## Environment")
    for key, value in report["versions"].items():
        lines.append(f"- `{key}`: `{value}`")
    for key, value in report["torch_cuda"].items():
        lines.append(f"- `torch_cuda.{key}`: `{value}`")
    lines.append("")

    lines.append("## Support Flags")
    for key, value in report["support_flags"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")

    lines.append("## Modules And Symbols")
    for module in report["modules"]:
        lines.append(f"### `{module['name']}`")
        lines.append(f"- present: `{module['present']}`")
        lines.append(f"- file: `{module['file']}`")
        if module.get("error"):
            lines.append(f"- error: `{module['error']}`")
        for symbol in module["symbols"]:
            lines.append(f"- `{symbol['name']}` present: `{symbol['present']}`")
            if symbol.get("signature"):
                lines.append(f"  signature: `{symbol['signature']}`")
            if symbol.get("source_file"):
                lines.append(
                    "  source: "
                    f"`{symbol['source_file']}:{symbol['source_start_line']}`"
                )
        lines.append("")
    output_path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-lines", type=int, default=80)
    args = parser.parse_args(argv)

    modules = [module_report(name, args.max_lines) for name in DEFAULT_MODULES]
    report = {
        "versions": {
            "torch": version_of("torch"),
            "vllm": version_of("vllm"),
            "flashinfer-python": version_of("flashinfer-python"),
            "flashmla": version_of("flashmla"),
        },
        "torch_cuda": collect_torch_cuda(),
        "support_flags": collect_support_flags(),
        "modules": [asdict(module) for module in modules],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "flashmla_extraction.json"
    md_path = args.out_dir / "flashmla_extraction.md"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    write_markdown(report, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
