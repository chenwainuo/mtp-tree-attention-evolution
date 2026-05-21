"""Clone, patch, and build vLLM with a local FlashMLA source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_VLLM_REPO = "https://github.com/vllm-project/vllm.git"
DEFAULT_FLASHMLA_REPO = "https://github.com/vllm-project/FlashMLA.git"
FLASHMLA_CONFIG_PATH = Path("csrc/sm90/prefill/sparse/config.h")
EXPECTED_B_TOPK = "static constexpr int B_TOPK = 64;    // TopK block size"
EXPECTED_TEMPLATE = "template<int D_QK, bool HAVE_TOPK_LENGTH"
FLASHMLA_TAG_RE = re.compile(r"GIT_TAG\s+([0-9a-fA-F]{8,40}|[^\s)]+)")
FLASHMLA_EXTENSION_NAMES = ("_flashmla_C", "_flashmla_extension_C")
SETUP_FLASHMLA_CONDITION = """if envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9")
    ):"""
SETUP_FORCED_FLASHMLA_CONDITION = """if os.getenv("MTP_FORCE_FLASHMLA_EXTENSIONS") == "1" or envs.VLLM_USE_PRECOMPILED or (
        CUDA_HOME and get_nvcc_cuda_version() >= Version("12.9")
    ):"""


def run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    display = " ".join(command)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="replace") as log:
            log.write(f"$ {display}\n")
    print(f"$ {display}", flush=True)
    proc = subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(line)
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"{display} failed with exit code {returncode}")


def git_clone_ref(repo: str, ref: str, dst: Path, *, log_path: Path | None = None) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    run_command(["git", "clone", "--depth", "1", repo, str(dst)], log_path=log_path)
    run_command(["git", "fetch", "--depth", "1", "origin", ref], cwd=dst, log_path=log_path)
    run_command(["git", "checkout", "FETCH_HEAD"], cwd=dst, log_path=log_path)
    run_command(["git", "submodule", "update", "--init", "--recursive"], cwd=dst, log_path=log_path)


def reset_repo_tree(repo_dir: Path, *, clean_untracked: bool, log_path: Path | None = None) -> None:
    run_command(["git", "reset", "--hard"], cwd=repo_dir, log_path=log_path)
    if clean_untracked:
        run_command(["git", "clean", "-fd"], cwd=repo_dir, log_path=log_path)


def prepare_repo_tree(
    repo: str,
    ref: str,
    dst: Path,
    *,
    reuse_existing_tree: bool,
    clean_untracked: bool,
    log_path: Path | None = None,
) -> None:
    if reuse_existing_tree and (dst / ".git").exists():
        reset_repo_tree(dst, clean_untracked=clean_untracked, log_path=log_path)
        return
    git_clone_ref(repo, ref, dst, log_path=log_path)


def git_rev_parse(repo_dir: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def parse_flashmla_ref(vllm_dir: Path) -> str:
    cmake_path = vllm_dir / "cmake" / "external_projects" / "flashmla.cmake"
    content = cmake_path.read_text()
    match = FLASHMLA_TAG_RE.search(content)
    if not match:
        raise RuntimeError(f"could not find FlashMLA GIT_TAG in {cmake_path}")
    return match.group(1)


def validate_flashmla_source(flashmla_dir: Path, *, require_original_btopk: bool = True) -> str:
    config_path = flashmla_dir / FLASHMLA_CONFIG_PATH
    if not config_path.exists():
        raise RuntimeError(f"source mismatch: missing {FLASHMLA_CONFIG_PATH}")
    content = config_path.read_text()
    expected = [EXPECTED_TEMPLATE]
    if require_original_btopk:
        expected.append(EXPECTED_B_TOPK)
    missing = [marker for marker in expected if marker not in content]
    if missing:
        raise RuntimeError(
            "source mismatch: expected SM90 sparse prefill config markers missing: "
            + ", ".join(missing)
        )
    return content


def source_excerpt(content: str, marker: str, *, context: int = 8) -> str:
    lines = content.splitlines()
    marker_line = next((i for i, line in enumerate(lines) if marker in line), -1)
    if marker_line < 0:
        return ""
    start = max(0, marker_line - context)
    end = min(len(lines), marker_line + context + 1)
    return "\n".join(lines[start:end])


def artifact_path(artifacts_dir: Path, stem: str, suffix: str, label: str | None) -> Path:
    label_suffix = "" if label is None else f"_{label}"
    return artifacts_dir / f"{stem}{label_suffix}{suffix}"


def apply_candidate_patch(
    flashmla_dir: Path,
    patch_path: Path,
    artifacts_dir: Path,
    label: str | None,
) -> str:
    patch_text = patch_path.read_text()
    patch_sha = hashlib.sha256(patch_text.encode()).hexdigest()
    artifact_path(artifacts_dir, "applied_patch", ".patch", label).write_text(patch_text)
    (artifacts_dir / "applied_patch.patch").write_text(patch_text)
    run_command(["git", "apply", "--check", str(patch_path)], cwd=flashmla_dir)
    run_command(["git", "apply", str(patch_path)], cwd=flashmla_dir)
    validate_flashmla_source(flashmla_dir, require_original_btopk=False)
    return patch_sha


def patch_vllm_setup_for_flashmla_overlay(vllm_dir: Path) -> None:
    setup_path = vllm_dir / "setup.py"
    content = setup_path.read_text()
    if SETUP_FORCED_FLASHMLA_CONDITION in content:
        pass
    elif SETUP_FLASHMLA_CONDITION in content:
        content = content.replace(SETUP_FLASHMLA_CONDITION, SETUP_FORCED_FLASHMLA_CONDITION, 1)
    else:
        raise RuntimeError("source mismatch: vLLM setup.py FlashMLA condition not found")

    filter_marker = "if _no_device():\n    ext_modules = []\n"
    filter_block = (
        'if os.getenv("MTP_FLASHMLA_ONLY_BUILD") == "1":\n'
        '    flashmla_targets = {"vllm._flashmla_C", "vllm._flashmla_extension_C"}\n'
        "    ext_modules = [ext for ext in ext_modules if ext.name in flashmla_targets]\n\n"
    )
    if filter_block in content:
        pass
    elif filter_marker in content:
        content = content.replace(filter_marker, filter_block + filter_marker, 1)
    else:
        raise RuntimeError("source mismatch: vLLM setup.py extension filter marker not found")

    triton_copy_marker = "        if _is_cuda() or _is_hip():\n"
    triton_copy_replacement = (
        '        if (_is_cuda() or _is_hip()) and os.getenv("MTP_FLASHMLA_ONLY_BUILD") != "1":\n'
    )
    if triton_copy_replacement in content:
        pass
    elif triton_copy_marker in content:
        content = content.replace(triton_copy_marker, triton_copy_replacement, 1)
    else:
        raise RuntimeError("source mismatch: vLLM setup.py triton copy marker not found")

    deep_gemm_copy_marker = "        if _is_cuda():\n            # copy vendored deep_gemm package"
    deep_gemm_copy_replacement = (
        '        if _is_cuda() and os.getenv("MTP_FLASHMLA_ONLY_BUILD") != "1":\n'
        "            # copy vendored deep_gemm package"
    )
    if deep_gemm_copy_replacement in content:
        pass
    elif deep_gemm_copy_marker in content:
        content = content.replace(deep_gemm_copy_marker, deep_gemm_copy_replacement, 1)
    else:
        raise RuntimeError("source mismatch: vLLM setup.py deep_gemm copy marker not found")
    setup_path.write_text(content)

    cmake_path = vllm_dir / "CMakeLists.txt"
    cmake_content = cmake_path.read_text()
    cmake_external_marker = """# For CUDA and HIP builds also build the triton_kernels external package.
if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
    include(cmake/external_projects/triton_kernels.cmake)
endif()

# For CUDA we also build and ship some external projects.
if (VLLM_GPU_LANG STREQUAL "CUDA")
    include(cmake/external_projects/deepgemm.cmake)
    include(cmake/external_projects/flashmla.cmake)
    include(cmake/external_projects/qutlass.cmake)

    # vllm-flash-attn should be last as it overwrites some CMake functions
    include(cmake/external_projects/vllm_flash_attn.cmake)
endif ()
"""
    cmake_external_replacement = """if(DEFINED ENV{MTP_FLASHMLA_ONLY_BUILD} AND "$ENV{MTP_FLASHMLA_ONLY_BUILD}" STREQUAL "1")
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        include(cmake/external_projects/flashmla.cmake)
    endif()
else()
    # For CUDA and HIP builds also build the triton_kernels external package.
    if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")
        include(cmake/external_projects/triton_kernels.cmake)
    endif()

    # For CUDA we also build and ship some external projects.
    if (VLLM_GPU_LANG STREQUAL "CUDA")
        include(cmake/external_projects/deepgemm.cmake)
        include(cmake/external_projects/flashmla.cmake)
        include(cmake/external_projects/qutlass.cmake)

        # vllm-flash-attn should be last as it overwrites some CMake functions
        include(cmake/external_projects/vllm_flash_attn.cmake)
    endif ()
endif()
"""
    if cmake_external_replacement in cmake_content:
        pass
    elif cmake_external_marker in cmake_content:
        cmake_content = cmake_content.replace(
            cmake_external_marker, cmake_external_replacement, 1
        )
    else:
        raise RuntimeError("source mismatch: vLLM CMake external-project block not found")
    cmake_path.write_text(cmake_content)


def installed_vllm_package_dir(python: str) -> Path:
    result = subprocess.run(
        [
            python,
            "-c",
            "from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def copy_flashmla_overlay(vllm_dir: Path, package_dir: Path) -> list[str]:
    copied: list[str] = []
    for extension_name in FLASHMLA_EXTENSION_NAMES:
        matches = sorted((vllm_dir / "vllm").glob(f"{extension_name}*.so"))
        if not matches:
            raise RuntimeError(f"build did not produce vllm/{extension_name}*.so")
        src = matches[-1]
        dst = package_dir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))

    src_interface = vllm_dir / "vllm" / "third_party" / "flashmla" / "flash_mla_interface.py"
    if not src_interface.exists():
        raise RuntimeError("build did not produce vllm/third_party/flashmla/flash_mla_interface.py")
    dst_interface = package_dir / "third_party" / "flashmla" / "flash_mla_interface.py"
    dst_interface.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_interface, dst_interface)
    copied.append(str(dst_interface))
    return copied


def build_vllm(
    vllm_dir: Path,
    flashmla_dir: Path,
    *,
    python: str,
    artifacts_dir: Path,
    max_jobs: int,
    label: str | None,
) -> dict[str, Any]:
    patch_vllm_setup_for_flashmla_overlay(vllm_dir)
    package_dir = installed_vllm_package_dir(python)
    env = os.environ.copy()
    env.update(
        {
            "FLASH_MLA_SRC_DIR": str(flashmla_dir),
            "VLLM_TARGET_DEVICE": "cuda",
            "MAX_JOBS": str(max_jobs),
            "MTP_FORCE_FLASHMLA_EXTENSIONS": "1",
            "MTP_FLASHMLA_ONLY_BUILD": "1",
        }
    )
    build_log = artifact_path(artifacts_dir, "build", ".log", label)
    run_command(
        [python, "setup.py", "build_ext", "--inplace"],
        cwd=vllm_dir,
        env=env,
        log_path=build_log,
    )
    copied = copy_flashmla_overlay(vllm_dir, package_dir)
    overlay = {
        "mode": "flashmla-extension-overlay",
        "installed_vllm_package_dir": str(package_dir),
        "copied": copied,
    }
    write_json(artifact_path(artifacts_dir, "source_overlay", ".json", label), overlay)
    write_json(artifacts_dir / "source_overlay.json", overlay)
    return overlay


def verify_flashmla_overlay(python: str) -> None:
    subprocess.run(
        [
            python,
            "-c",
            "import vllm._flashmla_C, vllm._flashmla_extension_C; "
            "print(vllm._flashmla_C.__file__); print(vllm._flashmla_extension_C.__file__)",
        ],
        check=True,
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--vllm-repo", default=DEFAULT_VLLM_REPO)
    parser.add_argument("--vllm-ref", default="releases/v0.21.0")
    parser.add_argument("--flashmla-repo", default=DEFAULT_FLASHMLA_REPO)
    parser.add_argument("--flashmla-ref", default="auto")
    parser.add_argument("--candidate-patch", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/flashmla-source-build"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/workspace/mtp-runpod-artifacts"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--reuse-existing-tree", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--local-dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.candidate_patch is not None and not args.candidate_patch.is_absolute():
        args.candidate_patch = Path.cwd() / args.candidate_patch

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    vllm_dir = args.work_dir / "vllm"
    flashmla_dir = args.work_dir / "FlashMLA"

    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "vllm_repo": args.vllm_repo,
        "vllm_ref": args.vllm_ref,
        "flashmla_repo": args.flashmla_repo,
        "candidate_patch": None if args.candidate_patch is None else str(args.candidate_patch),
        "status": "planned" if args.local_dry_run else "running",
    }
    summary_path = artifact_path(args.artifacts_dir, "source_build_summary", ".json", args.label)
    write_json(summary_path, summary)
    write_json(args.artifacts_dir / "source_build_summary.json", summary)

    if args.local_dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        prepare_repo_tree(
            args.vllm_repo,
            args.vllm_ref,
            vllm_dir,
            reuse_existing_tree=args.reuse_existing_tree,
            clean_untracked=False,
        )
        flashmla_ref = parse_flashmla_ref(vllm_dir) if args.flashmla_ref == "auto" else args.flashmla_ref
        prepare_repo_tree(
            args.flashmla_repo,
            flashmla_ref,
            flashmla_dir,
            reuse_existing_tree=args.reuse_existing_tree,
            clean_untracked=True,
        )
        before_content = validate_flashmla_source(flashmla_dir)

        patch_sha = None
        if args.candidate_patch is not None:
            patch_sha = apply_candidate_patch(
                flashmla_dir,
                args.candidate_patch,
                args.artifacts_dir,
                args.label,
            )

        after_content = (flashmla_dir / FLASHMLA_CONFIG_PATH).read_text()
        provenance = {
            "vllm_repo": args.vllm_repo,
            "vllm_ref": args.vllm_ref,
            "vllm_commit": git_rev_parse(vllm_dir),
            "flashmla_repo": args.flashmla_repo,
            "flashmla_ref": flashmla_ref,
            "flashmla_commit": git_rev_parse(flashmla_dir),
            "flashmla_config_path": str(FLASHMLA_CONFIG_PATH),
            "candidate_patch": None if args.candidate_patch is None else str(args.candidate_patch),
            "candidate_patch_sha256": patch_sha,
            "expected_b_topk_marker": EXPECTED_B_TOPK,
            "before_excerpt": source_excerpt(before_content, EXPECTED_B_TOPK),
            "after_excerpt": source_excerpt(after_content, "B_TOPK"),
        }
        write_json(artifact_path(args.artifacts_dir, "source_provenance", ".json", args.label), provenance)
        write_json(args.artifacts_dir / "source_provenance.json", provenance)

        overlay = None
        if not args.skip_build:
            overlay = build_vllm(
                vllm_dir,
                flashmla_dir,
                python=args.python,
                artifacts_dir=args.artifacts_dir,
                max_jobs=args.max_jobs,
                label=args.label,
            )
            verify_flashmla_overlay(args.python)

        summary.update(
            {
                "status": "succeeded",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "source_provenance": provenance,
                "overlay": overlay,
            }
        )
        write_json(summary_path, summary)
        write_json(args.artifacts_dir / "source_build_summary.json", summary)
        return 0
    except Exception as exc:  # noqa: BLE001 - remote artifact should capture detail.
        summary.update(
            {
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        write_json(summary_path, summary)
        write_json(args.artifacts_dir / "source_build_summary.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
