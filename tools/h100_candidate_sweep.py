"""Run multiple H100 Triton sparse-prefill candidates inside one pod."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.evolve_h100 import (
    DEFAULT_BASELINE_US,
    CandidateResult,
    benchmark_command_for_candidate,
    correctness_passed,
    default_candidates,
    parse_benchmark_output,
    parse_candidate_spec,
)


def run_candidate(candidate: Any, args: argparse.Namespace) -> CandidateResult:
    benchmark_command = benchmark_command_for_candidate(
        candidate,
        python=args.python,
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"=== Candidate: {candidate.name} ===", flush=True)
    print(f"$ {benchmark_command}", flush=True)
    completed = subprocess.run(
        shlex.split(benchmark_command),
        check=False,
        text=True,
        capture_output=True,
    )
    output = completed.stdout + completed.stderr
    print(output, end="" if output.endswith("\n") else "\n", flush=True)

    runtime_us, correctness = parse_benchmark_output(output)
    result = CandidateResult(
        name=candidate.name,
        status="failed",
        command=shlex.split(benchmark_command),
        benchmark_command=benchmark_command,
        runtime_us=runtime_us,
        correctness=correctness,
        returncode=completed.returncode,
    )
    if runtime_us is not None:
        result.speedup_pct = (args.baseline_us - runtime_us) / args.baseline_us * 100.0

    if completed.returncode != 0:
        result.reason = f"candidate command returned {completed.returncode}"
        return result
    if runtime_us is None:
        result.reason = "candidate output did not include Runtime"
        return result
    if not correctness_passed(correctness):
        result.reason = f"correctness did not pass: {correctness}"
        return result

    result.status = "succeeded"
    return result


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--baseline-us", type=float, default=DEFAULT_BASELINE_US)
    parser.add_argument("--min-speedup-pct", type=float, default=2.0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--candidate", action="append", type=parse_candidate_spec, default=None)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("/workspace/mtp-runpod-artifacts/candidate_summary.json"),
    )
    args = parser.parse_args(argv)

    candidates = (args.candidate or default_candidates())[: args.max_candidates]
    summary: dict[str, Any] = {
        "baseline_us": args.baseline_us,
        "min_speedup_pct": args.min_speedup_pct,
        "status": "running",
        "results": [],
    }
    write_summary(args.summary_path, summary)

    for candidate in candidates:
        result = run_candidate(candidate, args)
        summary["results"].append(asdict(result))
        write_summary(args.summary_path, summary)
        if result.status == "succeeded":
            assert result.speedup_pct is not None
            print(
                f"Candidate summary: runtime={result.runtime_us:.2f} us, "
                f"speedup={result.speedup_pct:.2f}%",
                flush=True,
            )
            if result.speedup_pct >= args.min_speedup_pct:
                summary["status"] = "improved"
                summary["winner"] = asdict(result)
                write_summary(args.summary_path, summary)
                print(f"Improvement found: {candidate.name}", flush=True)
                return 0
        else:
            print(f"Candidate failed: {result.reason}", flush=True)

    summary["status"] = "exhausted"
    write_summary(args.summary_path, summary)
    print("No qualifying improvement found.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
