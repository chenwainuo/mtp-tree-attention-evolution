"""Run a bounded H100 sparse-prefill evolution loop on RunPod.

The loop is intentionally conservative:

- BF16 sparse prefill is the correctness gate.
- FlashMLA remains the baseline/source of truth.
- Triton candidates are opt-in experiment variants.
- The loop stops on the first correctness-passing candidate that clears the
  requested speedup threshold, or when the candidate quota is exhausted.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_US = 23.29
DEFAULT_CANDIDATES = (
    (16, 64, 64, 4),
    (32, 64, 64, 4),
    (64, 64, 64, 4),
    (32, 128, 64, 4),
    (32, 64, 128, 8),
)

RUNTIME_RE = re.compile(r"^Runtime:\s*([0-9]+(?:\.[0-9]+)?)\s*us\s*$", re.MULTILINE)
CORRECTNESS_RE = re.compile(r"^Correctness:\s*(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Candidate:
    name: str
    block_k: int
    block_d: int
    block_v: int
    warps: int


@dataclass
class CandidateResult:
    name: str
    status: str
    command: list[str]
    benchmark_command: str
    runtime_us: float | None = None
    speedup_pct: float | None = None
    correctness: str | None = None
    run_dir: str | None = None
    returncode: int | None = None
    reason: str | None = None


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def candidate_name(block_k: int, block_d: int, block_v: int, warps: int) -> str:
    return f"triton-k{block_k}-d{block_d}-v{block_v}-w{warps}"


def parse_candidate_spec(spec: str) -> Candidate:
    values: dict[str, str] = {}
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        key, separator, value = part.partition("=")
        if not separator:
            raise argparse.ArgumentTypeError(
                "candidate specs must look like k=32,d=64,v=64,warps=4"
            )
        values[key.strip().lower()] = value.strip()

    aliases = {"block_k": "k", "block_d": "d", "block_v": "v", "w": "warps"}
    normalized = {aliases.get(key, key): value for key, value in values.items()}
    missing = {"k", "d", "v", "warps"} - normalized.keys()
    if missing:
        raise argparse.ArgumentTypeError(f"candidate spec missing {sorted(missing)}")

    try:
        block_k = int(normalized["k"])
        block_d = int(normalized["d"])
        block_v = int(normalized["v"])
        warps = int(normalized["warps"])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"candidate values must be integers: {spec}") from exc

    for label, value in (("k", block_k), ("d", block_d), ("v", block_v)):
        if not is_power_of_two(value):
            raise argparse.ArgumentTypeError(f"candidate {label} must be a power of two")
    if warps <= 0:
        raise argparse.ArgumentTypeError("candidate warps must be positive")

    name = normalized.get("name") or candidate_name(block_k, block_d, block_v, warps)
    return Candidate(name=name, block_k=block_k, block_d=block_d, block_v=block_v, warps=warps)


def default_candidates() -> list[Candidate]:
    return [
        Candidate(
            name=candidate_name(block_k, block_d, block_v, warps),
            block_k=block_k,
            block_d=block_d,
            block_v=block_v,
            warps=warps,
        )
        for block_k, block_d, block_v, warps in DEFAULT_CANDIDATES
    ]


def parse_benchmark_output(output: str) -> tuple[float | None, str | None]:
    runtime_match = RUNTIME_RE.search(output)
    correctness_match = CORRECTNESS_RE.search(output)
    runtime_us = float(runtime_match.group(1)) if runtime_match else None
    correctness = correctness_match.group(1) if correctness_match else None
    return runtime_us, correctness


def correctness_passed(correctness: str | None) -> bool:
    return correctness is not None and correctness.startswith("PASS")


def benchmark_command_for_candidate(
    candidate: Candidate,
    *,
    python: str,
    warmup: int,
    rep: int,
) -> str:
    parts = [
        python,
        "-m",
        "bench.run_benchmark",
        "--gpu",
        "h100",
        "--flashmla-mode",
        "bf16-prefill",
        "--flashmla-impl",
        "triton",
        "--triton-block-k",
        str(candidate.block_k),
        "--triton-block-d",
        str(candidate.block_d),
        "--triton-block-v",
        str(candidate.block_v),
        "--triton-warps",
        str(candidate.warps),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
    ]
    return " ".join(parts)


def build_runpod_command(
    candidate: Candidate,
    args: argparse.Namespace,
    candidate_output_dir: Path,
) -> tuple[list[str], str]:
    benchmark_command = benchmark_command_for_candidate(
        candidate,
        python=args.python,
        warmup=args.warmup,
        rep=args.rep,
    )
    command = [
        args.python,
        str(args.repo_root / "tools" / "runpod_benchmark.py"),
        "--gpu",
        "h100",
        "--flashmla-mode",
        "bf16-prefill",
        "--flashmla-impl",
        "triton",
        "--ref",
        args.ref,
        "--repo-url",
        args.repo_url,
        "--env-file",
        str(args.env_file),
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--poll-seconds",
        str(args.poll_seconds),
        "--output-dir",
        str(candidate_output_dir),
        "--name",
        f"mtp-evolve-{candidate.name}",
        "--benchmark-command",
        benchmark_command,
    ]
    if not args.keep_pods:
        command.append("--terminate-on-complete")
    return command, benchmark_command


def candidate_spec(candidate: Candidate) -> str:
    return (
        f"name={candidate.name},k={candidate.block_k},d={candidate.block_d},"
        f"v={candidate.block_v},warps={candidate.warps}"
    )


def sweep_benchmark_command(
    candidates: list[Candidate],
    *,
    python: str,
    baseline_us: float,
    min_speedup_pct: float,
    warmup: int,
    rep: int,
    max_candidates: int,
) -> str:
    parts = [
        python,
        "-m",
        "tools.h100_candidate_sweep",
        "--baseline-us",
        f"{baseline_us:.6g}",
        "--min-speedup-pct",
        f"{min_speedup_pct:.6g}",
        "--max-candidates",
        str(max_candidates),
        "--warmup",
        str(warmup),
        "--rep",
        str(rep),
    ]
    for candidate in candidates:
        parts.extend(["--candidate", candidate_spec(candidate)])
    return shlex.join(parts)


def build_sweep_runpod_command(
    candidates: list[Candidate],
    args: argparse.Namespace,
    sweep_output_dir: Path,
) -> tuple[list[str], str]:
    benchmark_command = sweep_benchmark_command(
        candidates,
        python=args.python,
        baseline_us=args.baseline_us,
        min_speedup_pct=args.min_speedup_pct,
        warmup=args.warmup,
        rep=args.rep,
        max_candidates=args.max_candidates,
    )
    command = [
        args.python,
        str(args.repo_root / "tools" / "runpod_benchmark.py"),
        "--gpu",
        "h100",
        "--flashmla-mode",
        "bf16-prefill",
        "--ref",
        args.ref,
        "--repo-url",
        args.repo_url,
        "--env-file",
        str(args.env_file),
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--poll-seconds",
        str(args.poll_seconds),
        "--output-dir",
        str(sweep_output_dir),
        "--name",
        "mtp-evolve-h100-sweep",
        "--benchmark-command",
        benchmark_command,
    ]
    if not args.keep_pods:
        command.append("--terminate-on-complete")
    return command, benchmark_command


def latest_run_dir(output_dir: Path) -> Path | None:
    candidates = [path for path in output_dir.glob("runpod-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_result_from_artifacts(
    candidate: Candidate,
    command: list[str],
    benchmark_command: str,
    candidate_output_dir: Path,
    *,
    baseline_us: float,
    returncode: int,
) -> CandidateResult:
    run_dir = latest_run_dir(candidate_output_dir)
    result = CandidateResult(
        name=candidate.name,
        status="failed",
        command=command,
        benchmark_command=benchmark_command,
        run_dir=None if run_dir is None else str(run_dir),
        returncode=returncode,
    )
    if run_dir is None:
        result.reason = "no RunPod artifact directory was created"
        return result

    report_path = run_dir / "report.json"
    output_path = run_dir / "output.log"
    output = output_path.read_text(encoding="utf-8", errors="replace") if output_path.exists() else ""
    runtime_us, correctness = parse_benchmark_output(output)
    result.runtime_us = runtime_us
    result.correctness = correctness
    if runtime_us is not None:
        result.speedup_pct = (baseline_us - runtime_us) / baseline_us * 100.0

    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text())

    if returncode != 0 or report.get("status") != "succeeded":
        result.reason = report.get("error") or f"runpod launcher returned {returncode}"
        return result
    if runtime_us is None:
        result.reason = "benchmark output did not include a runtime"
        return result
    if not correctness_passed(correctness):
        result.reason = f"correctness did not pass: {correctness}"
        return result

    result.status = "succeeded"
    return result


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_loop(args: argparse.Namespace) -> int:
    if not args.per_candidate_pods:
        return run_sweep(args)

    selected = args.candidate or default_candidates()
    selected = selected[: args.max_candidates]
    session_dir = args.output_dir / datetime.now(timezone.utc).strftime("evolve-h100-%Y%m%d-%H%M%S")
    summary_path = session_dir / "summary.json"
    session_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "baseline_us": args.baseline_us,
        "min_speedup_pct": args.min_speedup_pct,
        "ref": args.ref,
        "status": "running",
        "results": [],
    }
    write_summary(summary_path, payload)

    print(f"Evolution session: {session_dir}")
    print(f"Baseline: {args.baseline_us:.2f} us; target speedup: {args.min_speedup_pct:.2f}%")

    for candidate in selected:
        candidate_output_dir = session_dir / candidate.name
        candidate_output_dir.mkdir(parents=True, exist_ok=True)
        command, benchmark_command = build_runpod_command(candidate, args, candidate_output_dir)
        print(f"\nCandidate: {candidate.name}")
        print(f"Benchmark command: {benchmark_command}")

        if args.local_dry_run:
            result = CandidateResult(
                name=candidate.name,
                status="planned",
                command=command,
                benchmark_command=benchmark_command,
            )
            payload["results"].append(asdict(result))
            continue

        completed = subprocess.run(command, cwd=args.repo_root, check=False)
        result = load_result_from_artifacts(
            candidate,
            command,
            benchmark_command,
            candidate_output_dir,
            baseline_us=args.baseline_us,
            returncode=completed.returncode,
        )
        payload["results"].append(asdict(result))
        write_summary(summary_path, payload)

        if result.status == "succeeded":
            assert result.runtime_us is not None
            assert result.speedup_pct is not None
            print(
                f"Candidate result: {result.runtime_us:.2f} us "
                f"({result.speedup_pct:.2f}% vs baseline)"
            )
            if result.speedup_pct >= args.min_speedup_pct:
                payload["status"] = "improved"
                payload["winner"] = asdict(result)
                write_summary(summary_path, payload)
                print(f"Improvement found: {candidate.name}")
                return 0
        else:
            print(f"Candidate failed: {result.reason}")

    payload["status"] = "exhausted" if not args.local_dry_run else "planned"
    write_summary(summary_path, payload)
    print(f"Summary: {summary_path}")
    return 0 if args.local_dry_run else 1


def run_sweep(args: argparse.Namespace) -> int:
    selected = (args.candidate or default_candidates())[: args.max_candidates]
    session_dir = args.output_dir / datetime.now(timezone.utc).strftime("evolve-h100-%Y%m%d-%H%M%S")
    sweep_output_dir = session_dir / "sweep"
    summary_path = session_dir / "summary.json"
    session_dir.mkdir(parents=True, exist_ok=True)
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    command, benchmark_command = build_sweep_runpod_command(selected, args, sweep_output_dir)
    payload: dict[str, Any] = {
        "baseline_us": args.baseline_us,
        "min_speedup_pct": args.min_speedup_pct,
        "ref": args.ref,
        "mode": "single-pod-sweep",
        "status": "planned" if args.local_dry_run else "running",
        "command": command,
        "benchmark_command": benchmark_command,
    }
    write_summary(summary_path, payload)

    print(f"Evolution session: {session_dir}")
    print(f"Baseline: {args.baseline_us:.2f} us; target speedup: {args.min_speedup_pct:.2f}%")
    print(f"Candidates in one H100 pod: {len(selected)}")
    print(f"Benchmark command: {benchmark_command}")

    if args.local_dry_run:
        print(f"Summary: {summary_path}")
        return 0

    completed = subprocess.run(command, cwd=args.repo_root, check=False)
    payload["returncode"] = completed.returncode
    run_dir = latest_run_dir(sweep_output_dir)
    payload["run_dir"] = None if run_dir is None else str(run_dir)

    if run_dir is None:
        payload["status"] = "failed"
        payload["reason"] = "no RunPod artifact directory was created"
        write_summary(summary_path, payload)
        print(f"Summary: {summary_path}")
        return 1

    report_path = run_dir / "report.json"
    remote_summary_path = run_dir / "candidate_summary.json"
    if report_path.exists():
        payload["report"] = json.loads(report_path.read_text())
    if remote_summary_path.exists():
        remote_summary = json.loads(remote_summary_path.read_text())
        payload["remote_summary"] = remote_summary
        payload["status"] = remote_summary.get("status", "unknown")
        if remote_summary.get("winner"):
            payload["winner"] = remote_summary["winner"]
    else:
        payload["status"] = "failed"
        payload["reason"] = "candidate_summary.json was not collected"

    write_summary(summary_path, payload)
    print(f"Summary: {summary_path}")
    return 0 if payload.get("status") == "improved" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo-url", default="https://github.com/chenwainuo/mtp-tree-attention-evolution.git")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--python", default="python3")
    parser.add_argument("--baseline-us", type=float, default=DEFAULT_BASELINE_US)
    parser.add_argument("--min-speedup-pct", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=len(DEFAULT_CANDIDATES))
    parser.add_argument("--candidate", action="append", type=parse_candidate_spec, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evolve_h100"))
    parser.add_argument("--keep-pods", action="store_true")
    parser.add_argument(
        "--per-candidate-pods",
        action="store_true",
        help="Create a fresh pod for each candidate. Default is one pod for the sweep.",
    )
    parser.add_argument("--local-dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.env_file = args.env_file if args.env_file.is_absolute() else args.repo_root / args.env_file
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    if args.max_candidates <= 0:
        raise SystemExit("--max-candidates must be positive")
    if args.baseline_us <= 0:
        raise SystemExit("--baseline-us must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    return run_loop(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
