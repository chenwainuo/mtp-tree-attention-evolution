"""Launch a bounded FlashMLA source-build optimization loop on RunPod."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.runpod_benchmark import DEFAULT_REPO_URL


def normalize_source_ref(ref: str) -> str:
    if ref.startswith("v") and ref[1:].replace(".", "").isdigit():
        return f"releases/{ref}"
    return ref


def flashmla_source_loop_command(args: argparse.Namespace) -> str:
    parts = [
        args.python,
        "-m",
        "tools.flashmla_source_loop",
        "--python",
        args.python,
        "--mode",
        args.mode,
        "--baseline-us",
        f"{args.baseline_us:.6g}",
        "--min-speedup-pct",
        f"{args.min_speedup_pct:.6g}",
        "--source-baseline-max-drift-pct",
        f"{args.source_baseline_max_drift_pct:.6g}",
        "--source-ref",
        normalize_source_ref(args.source_ref),
        "--flashmla-ref",
        args.flashmla_ref,
        "--max-candidates",
        str(args.max_candidates),
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--max-jobs",
        str(args.max_jobs),
    ]
    for candidate in args.candidate:
        parts.extend(["--candidate", str(candidate)])
    return shlex.join(parts)


def build_runpod_command(args: argparse.Namespace, session_dir: Path) -> tuple[list[str], str]:
    benchmark_command = flashmla_source_loop_command(args)
    command = [
        args.python,
        str(args.repo_root / "tools" / "runpod_benchmark.py"),
        "--gpu",
        "h100",
        "--flashmla-mode",
        args.mode,
        "--flashmla-impl",
        "flashmla",
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
        "--pod-create-retries",
        str(getattr(args, "pod_create_retries", 1)),
        "--pod-create-retry-seconds",
        str(getattr(args, "pod_create_retry_seconds", 60)),
        "--install-profile",
        "runpod-vllm-source",
        "--output-dir",
        str(session_dir / "runpod"),
        "--name",
        "mtp-flashmla-source-loop",
        "--skip-extract-flashmla",
        "--benchmark-command",
        benchmark_command,
    ]
    if args.terminate_on_complete:
        command.append("--terminate-on-complete")
    return command, benchmark_command


def latest_run_dir(output_dir: Path) -> Path | None:
    candidates = [path for path in output_dir.glob("runpod-*") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_remote_summary(run_dir: Path | None) -> dict[str, Any] | None:
    if run_dir is None:
        return None
    path = run_dir / "candidate_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_loop(args: argparse.Namespace) -> int:
    session_dir = args.output_dir / datetime.now(timezone.utc).strftime(
        "evolve-flashmla-%Y%m%d-%H%M%S"
    )
    command, benchmark_command = build_runpod_command(args, session_dir)
    payload: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "planned" if args.local_dry_run else "running",
        "mode": args.mode,
        "baseline_us": args.baseline_us,
        "min_speedup_pct": args.min_speedup_pct,
        "source_baseline_max_drift_pct": args.source_baseline_max_drift_pct,
        "source_ref": normalize_source_ref(args.source_ref),
        "flashmla_ref": args.flashmla_ref,
        "candidates": [str(path) for path in args.candidate],
        "command": command,
        "benchmark_command": benchmark_command,
    }
    summary_path = session_dir / "summary.json"
    write_summary(summary_path, payload)

    print(f"FlashMLA evolution session: {session_dir}")
    print(f"Benchmark command: {benchmark_command}")
    if args.local_dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    completed = subprocess.run(command, cwd=args.repo_root, check=False)
    payload["returncode"] = completed.returncode
    run_dir = latest_run_dir(session_dir / "runpod")
    payload["run_dir"] = None if run_dir is None else str(run_dir)
    remote_summary = load_remote_summary(run_dir)
    if remote_summary is None:
        payload["status"] = "failed"
        payload["error"] = "candidate_summary.json was not collected"
    else:
        payload["remote_summary"] = remote_summary
        payload["status"] = remote_summary.get("status", "unknown")
    write_summary(summary_path, payload)
    print(f"Summary: {summary_path}")
    return 0 if payload["status"] in {"improved", "smoke_succeeded"} else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default="main")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--python", default="python3")
    parser.add_argument("--mode", choices=("bf16-prefill",), default="bf16-prefill")
    parser.add_argument("--baseline-us", type=float, default=23.29)
    parser.add_argument("--source-ref", default="v0.21.0")
    parser.add_argument("--flashmla-ref", default="auto")
    parser.add_argument("--candidate", action="append", type=Path, default=[])
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--min-speedup-pct", type=float, default=2.0)
    parser.add_argument("--source-baseline-max-drift-pct", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--max-jobs", type=int, default=8)
    parser.add_argument("--timeout-minutes", type=int, default=240)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--pod-create-retries", type=int, default=1)
    parser.add_argument("--pod-create-retry-seconds", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evolve_flashmla"))
    parser.add_argument("--terminate-on-complete", action="store_true")
    parser.add_argument("--local-dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.env_file = args.env_file if args.env_file.is_absolute() else args.repo_root / args.env_file
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else args.repo_root / args.output_dir
    args.candidate = [
        candidate if candidate.is_absolute() else Path(candidate)
        for candidate in args.candidate
    ][: args.max_candidates]
    if args.baseline_us <= 0:
        raise SystemExit("--baseline-us must be positive")
    if args.max_candidates < 0:
        raise SystemExit("--max-candidates must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    return run_loop(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
