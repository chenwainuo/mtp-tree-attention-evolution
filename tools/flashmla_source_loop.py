"""Run no-op and patched FlashMLA source builds, then benchmark each."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
    environment_metadata: str | None = None
    serving_reports: list[str] | None = None
    vllm_log: str | None = None
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


def run_text(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "returncode": None,
            "output": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "returncode": proc.returncode,
        "output": (proc.stdout + proc.stderr)[-8000:],
    }


def safe_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def parse_concurrency_values(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("--concurrency values must be positive")
        values.append(value)
    if not values:
        raise ValueError("--concurrency must include at least one value")
    return values


def parse_env_overrides(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError("--vllm-env entries must use KEY=VALUE")
        parsed[key] = item
    return parsed


def prompt_workloads(path: Path) -> list[str]:
    workloads: set[str] = set()
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        workload = payload.get("workload")
        if workload:
            workloads.add(str(workload))
    if not workloads:
        raise ValueError(f"no workloads found in {path}")
    return sorted(workloads)


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
    if args.reuse_source_tree:
        command.append("--reuse-existing-tree")
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


def served_model_name(args: argparse.Namespace) -> str:
    if args.served_model_name:
        return args.served_model_name
    name = Path(str(args.model_path)).name
    return name or str(args.model_path)


def serving_base_url(args: argparse.Namespace) -> str:
    return f"http://127.0.0.1:{args.server_port}/v1"


def collect_environment_metadata(
    name: str,
    args: argparse.Namespace,
    *,
    patch: Path | None,
) -> Path:
    label = safe_label(name)
    provenance_path = args.artifacts_dir / f"source_provenance_{label}.json"
    provenance = None
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())

    version_probe = (
        "import json, platform\n"
        "data = {'python': platform.python_version()}\n"
        "try:\n"
        "    import torch\n"
        "    data['torch'] = torch.__version__\n"
        "    data['cuda_available'] = torch.cuda.is_available()\n"
        "    data['torch_cuda'] = getattr(torch.version, 'cuda', None)\n"
        "except Exception as exc:\n"
        "    data['torch_error'] = f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        "    import vllm\n"
        "    data['vllm'] = getattr(vllm, '__version__', None)\n"
        "except Exception as exc:\n"
        "    data['vllm_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(data, sort_keys=True))\n"
    )
    metadata = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "variant": name,
        "candidate_patch": None if patch is None else str(patch),
        "repo_commit": run_text(["git", "rev-parse", "HEAD"])["output"].strip(),
        "gpu": run_text(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "cuda": run_text(["nvidia-smi"]),
        "python_torch_vllm": run_text([args.python, "-c", version_probe]),
        "source_provenance": provenance,
        "serving": {
            "enabled": args.serving_benchmark,
            "model_path": None if args.model_path is None else str(args.model_path),
            "served_model_name": served_model_name(args) if args.model_path else None,
            "endpoint": args.endpoint,
            "prompts_file": str(args.prompts_file),
            "concurrency": parse_concurrency_values(args.concurrency),
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "warmup_requests": args.warmup_requests,
            "serving_repeats": args.serving_repeats,
            "server_port": args.server_port,
            "vllm_args": args.vllm_arg,
            "vllm_env": parse_env_overrides(args.vllm_env),
        },
    }
    path = args.artifacts_dir / f"environment_{label}.json"
    write_json(path, metadata)
    return path


def wait_for_vllm_server(base_url: str, proc: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    models_url = f"{base_url.rstrip('/')}/models"
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(models_url, timeout=5) as response:
                if 200 <= response.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"timed out waiting for vLLM server at {models_url}: {last_error}")


def start_vllm_server(
    name: str,
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[Any], Any, Path]:
    label = safe_label(name)
    log_path = args.artifacts_dir / f"vllm_{label}.log"
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    command = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(args.model_path),
        "--served-model-name",
        served_model_name(args),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.server_port),
    ]
    command.extend(args.vllm_arg)
    print(f"$ {' '.join(command)}", flush=True)
    log_handle.write(f"$ {' '.join(command)}\n")
    log_handle.flush()
    proc = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **parse_env_overrides(args.vllm_env)},
    )
    try:
        wait_for_vllm_server(serving_base_url(args), proc, timeout_s=args.server_startup_timeout_s)
    except Exception:
        stop_vllm_server(proc, log_handle)
        raise
    return proc, log_handle, log_path


def stop_vllm_server(proc: subprocess.Popen[Any], log_handle: Any) -> None:
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
    finally:
        log_handle.close()


def serving_benchmark_command(
    args: argparse.Namespace,
    *,
    output_json: Path,
    workload: str,
    concurrency: int,
) -> list[str]:
    return [
        args.python,
        "-m",
        "tools.bench_vllm_realtime",
        "--base-url",
        serving_base_url(args),
        "--model",
        served_model_name(args),
        "--prompts-file",
        str(args.prompts_file),
        "--endpoint",
        args.endpoint,
        "--concurrency",
        str(concurrency),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--warmup-requests",
        str(args.warmup_requests),
        "--output-json",
        str(output_json),
        "--workload",
        workload,
    ]


def run_serving_benchmarks(name: str, args: argparse.Namespace) -> tuple[list[str], str]:
    proc, log_handle, log_path = start_vllm_server(name, args)
    reports: list[str] = []
    try:
        label = safe_label(name)
        workloads = prompt_workloads(args.prompts_file)
        for repeat_index in range(1, args.serving_repeats + 1):
            for workload in workloads:
                workload_label = safe_label(workload)
                for concurrency in parse_concurrency_values(args.concurrency):
                    output_json = (
                        args.artifacts_dir
                        / f"serving_{label}_{workload_label}_c{concurrency}_r{repeat_index}.json"
                    )
                    rc, _ = run_capture(
                        serving_benchmark_command(
                            args,
                            output_json=output_json,
                            workload=workload,
                            concurrency=concurrency,
                        )
                    )
                    if rc != 0:
                        raise RuntimeError(
                            "serving benchmark failed: "
                            f"variant={name} workload={workload} "
                            f"concurrency={concurrency} repeat={repeat_index}"
                        )
                    reports.append(str(output_json))
    finally:
        stop_vllm_server(proc, log_handle)
    return reports, str(log_path)


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

    environment_path = collect_environment_metadata(name, args, patch=patch)
    result.environment_metadata = str(environment_path)
    result.status = "succeeded"
    return result


def attach_serving_benchmarks(result: FlashMLARun, args: argparse.Namespace) -> FlashMLARun:
    if not args.serving_benchmark or result.status != "succeeded":
        return result
    try:
        reports, log_path = run_serving_benchmarks(result.name, args)
    except Exception as exc:  # noqa: BLE001 - keep remote summary useful.
        result.status = "failed"
        result.error = f"serving benchmark failed: {type(exc).__name__}: {exc}"
        return result
    result.serving_reports = reports
    result.vllm_log = log_path
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
    parser.add_argument("--reuse-source-tree", action="store_true")
    parser.add_argument("--serving-benchmark", action="store_true")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--prompts-file", type=Path, default=Path("bench/prompts/realtime_prefill.jsonl"))
    parser.add_argument("--endpoint", choices=("chat.completions", "completions"), default="chat.completions")
    parser.add_argument("--concurrency", default="1,4,8,16")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--server-port", type=int, default=8001)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--serving-repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm-arg", action="append", default=[])
    parser.add_argument("--vllm-env", action="append", default=[])
    parser.add_argument("--server-startup-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)

    if args.serving_benchmark and not args.model_path:
        raise SystemExit("--serving-benchmark requires --model-path")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if args.serving_repeats <= 0:
        raise SystemExit("--serving-repeats must be positive")
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be non-negative")
    parse_concurrency_values(args.concurrency)
    parse_env_overrides(args.vllm_env)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    selected = args.candidate[: args.max_candidates]
    shared_work_dir = args.work_dir / "shared" if args.reuse_source_tree else None
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
        "serving_benchmark": args.serving_benchmark,
        "serving_config": {
            "model_path": args.model_path,
            "served_model_name": served_model_name(args) if args.model_path else None,
            "prompts_file": str(args.prompts_file),
            "endpoint": args.endpoint,
            "concurrency": parse_concurrency_values(args.concurrency),
            "max_tokens": args.max_tokens,
            "server_port": args.server_port,
            "warmup_requests": args.warmup_requests,
            "serving_repeats": args.serving_repeats,
            "temperature": args.temperature,
            "vllm_args": args.vllm_arg,
            "vllm_env": parse_env_overrides(args.vllm_env),
        },
        "results": [],
    }
    write_json(args.artifacts_dir / "candidate_summary.json", summary)

    reset_work_dir(args.work_dir)
    noop_result = build_and_benchmark(
        "source-noop",
        args,
        work_dir=shared_work_dir or args.work_dir / "noop",
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

    noop_result = attach_serving_benchmarks(noop_result, args)
    summary["source_noop"] = asdict(noop_result)
    summary["results"][0] = asdict(noop_result)
    write_json(args.artifacts_dir / "candidate_summary.json", summary)
    if noop_result.status != "succeeded":
        summary["status"] = "failed"
        summary["error"] = "source no-op serving benchmark failed"
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        return 0

    if not selected:
        summary["status"] = "smoke_succeeded"
        write_json(args.artifacts_dir / "candidate_summary.json", summary)
        return 0

    for patch in selected:
        candidate_result = build_and_benchmark(
            patch.stem,
            args,
            work_dir=shared_work_dir or args.work_dir / patch.stem,
            patch=patch,
            source_runtime_us=noop_result.runtime_us,
        )
        candidate_result = attach_serving_benchmarks(candidate_result, args)
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
