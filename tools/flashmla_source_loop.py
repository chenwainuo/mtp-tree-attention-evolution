"""Run no-op and patched FlashMLA source builds, then benchmark each."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNTIME_RE = re.compile(r"^Runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*us\s*$", re.MULTILINE)
CORRECTNESS_RE = re.compile(r"^Correctness:\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class FlashMLARun:
    name: str
    status: str
    runtime_us: float | None = None
    correctness: str | None = None
    speedup_vs_source_pct: float | None = None
    speedup_vs_wheel_pct: float | None = None
    command: str | None = None
    build_command: str | None = None
    error: str | None = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def parse_benchmark_output(output: str) -> tuple[float | None, str | None]:
    runtime_match = RUNTIME_RE.search(output)
    correctness_match = CORRECTNESS_RE.search(output)
    return (
        float(runtime_match.group(1)) if runtime_match else None,
        correctness_match.group(1) if correctness_match else None,
    )


def correctness_passed(correctness: str | None) -> bool:
    return correctness is not None and correctness.startswith("PASS")


def drift_pct(baseline_us: float, runtime_us: float) -> float:
    return abs(runtime_us - baseline_us) / baseline_us * 100.0


def run_capture(command: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    display = " ".join(command)
    print(f"$ {display}", flush=True)
    proc = subprocess.run(
        command,
        cwd=None if cwd is None else str(cwd),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = proc.stdout + proc.stderr
    print(output, end="" if output.endswith("\n") else "\n", flush=True)
    return proc.returncode, output


def run_stream(command: list[str], *, cwd: Path | None = None) -> int:
    display = " ".join(command)
    print(f"$ {display}", flush=True)
    return subprocess.run(command, cwd=None if cwd is None else str(cwd), check=False).returncode


def source_build_command(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    patch: Path | None,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "tools.source_build_flashmla",
        "--python",
        args.python,
        "--vllm-ref",
        args.source_ref,
        "--flashmla-ref",
        args.flashmla_ref,
        "--work-dir",
        str(work_dir),
        "--artifacts-dir",
        str(args.artifacts_dir),
        "--label",
        patch.stem if patch is not None else "source-noop",
        "--max-jobs",
        str(args.max_jobs),
    ]
    if patch is not None:
        command.extend(["--candidate-patch", str(patch)])
    return command


def benchmark_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python,
        "-m",
        "bench.run_benchmark",
        "--gpu",
        "h100",
        "--flashmla-mode",
        args.mode,
        "--flashmla-impl",
        "flashmla",
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
    ]


def extraction_command(args: argparse.Namespace, label: str) -> list[str]:
    return [
        args.python,
        "-m",
        "tools.extract_flashmla",
        "--out-dir",
        str(args.artifacts_dir),
        "--max-lines",
        "220",
        "--label",
        label,
    ]


def build_and_benchmark(
    name: str,
    args: argparse.Namespace,
    *,
    work_dir: Path,
    patch: Path | None,
    source_runtime_us: float | None,
) -> FlashMLARun:
    build_cmd = source_build_command(args, work_dir=work_dir, patch=patch)
    result = FlashMLARun(
        name=name,
        status="failed",
        build_command=" ".join(build_cmd),
        command=" ".join(benchmark_command(args)),
    )
    build_rc = run_stream(build_cmd)
    if build_rc != 0:
        result.error = f"source build returned {build_rc}"
        return result

    extract_rc = run_stream(extraction_command(args, name))
    if extract_rc != 0:
        result.error = f"FlashMLA extraction returned {extract_rc}"
        return result

    bench_rc, output = run_capture(benchmark_command(args))
    runtime_us, correctness = parse_benchmark_output(output)
    result.runtime_us = runtime_us
    result.correctness = correctness
    if runtime_us is not None:
        result.speedup_vs_wheel_pct = (args.baseline_us - runtime_us) / args.baseline_us * 100.0
        if source_runtime_us:
            result.speedup_vs_source_pct = (source_runtime_us - runtime_us) / source_runtime_us * 100.0
    if bench_rc != 0:
        result.error = f"benchmark returned {bench_rc}"
        return result
    if runtime_us is None:
        result.error = "benchmark output did not include Runtime"
        return result
    if not correctness_passed(correctness):
        result.error = f"correctness did not pass: {correctness}"
        return result
    result.status = "succeeded"
    return result


def reset_work_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--mode", choices=("bf16-prefill",), default="bf16-prefill")
    parser.add_argument("--baseline-us", type=float, default=23.29)
    parser.add_argument("--min-speedup-pct", type=float, default=2.0)
    parser.add_argument("--source-baseline-max-drift-pct", type=float, default=20.0)
    parser.add_argument("--source-ref", default="releases/v0.21.0")
    parser.add_argument("--flashmla-ref", default="auto")
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--work-dir", type=Path, default=Path("/workspace/flashmla-source-loop"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("/workspace/mtp-runpod-artifacts"))
    args = parser.parse_args(argv)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    selected = args.candidate[: args.max_candidates]
    summary: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "mode": args.mode,
        "baseline_us": args.baseline_us,
        "source_baseline_max_drift_pct": args.source_baseline_max_drift_pct,
        "min_speedup_pct": args.min_speedup_pct,
        "source_ref": args.source_ref,
        "flashmla_ref": args.flashmla_ref,
        "candidates": [str(path) for path in selected],
        "results": [],
    }
    write_json(args.artifacts_dir / "candidate_summary.json", summary)

    reset_work_dir(args.work_dir)
    noop_result = build_and_benchmark(
        "source-noop",
        args,
        work_dir=args.work_dir / "noop",
        patch=None,
        source_runtime_us=None,
    )
    summary["source_noop"] = asdict(noop_result)
    summary["results"].append(asdict(noop_result))
    write_json(args.artifacts_dir / "candidate_summary.json", summary)
    if noop_result.status != "succeeded" or noop_result.runtime_us is None:
        summary["status"] = "failed"
        summary["error"] = "source no-op build/benchmark failed"
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        return 0
    source_drift_pct = drift_pct(args.baseline_us, noop_result.runtime_us)
    summary["source_noop_drift_pct"] = source_drift_pct
    if source_drift_pct > args.source_baseline_max_drift_pct:
        summary["status"] = "failed"
        summary["error"] = (
            "source no-op runtime drifted "
            f"{source_drift_pct:.2f}% from wheel baseline; "
            f"limit is {args.source_baseline_max_drift_pct:.2f}%"
        )
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        return 0

    if not selected:
        summary["status"] = "smoke_succeeded"
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        return 0

    for patch in selected:
        reset_work_dir(args.work_dir / patch.stem)
        candidate_result = build_and_benchmark(
            patch.stem,
            args,
            work_dir=args.work_dir / patch.stem,
            patch=patch,
            source_runtime_us=noop_result.runtime_us,
        )
        summary["results"].append(asdict(candidate_result))
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        if (
            candidate_result.status == "succeeded"
            and candidate_result.speedup_vs_source_pct is not None
            and candidate_result.speedup_vs_source_pct >= args.min_speedup_pct
        ):
            summary["status"] = "improved"
            summary["winner"] = asdict(candidate_result)
            write_json(args.artifacts_dir / "candidate_summary.json", summary)
            return 0

    summary["status"] = "exhausted"
    write_json(args.artifacts_dir / "candidate_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
