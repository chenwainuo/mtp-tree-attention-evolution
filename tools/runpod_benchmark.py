"""Launch the benchmark on RunPod and collect a JSON report.

The launcher uses only the Python standard library so it can run before project
dependencies are installed locally. It creates a RunPod pod whose start command:

1. Starts a tiny HTTP file server for artifacts.
2. Clones the public benchmark repository.
3. Installs dependencies.
4. Runs local validation and the selected benchmark command.
5. Writes report.json and output.log for this launcher to poll.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNPOD_API_BASE = "https://rest.runpod.io/v1"
DEFAULT_REPO_URL = "https://github.com/chenwainuo/mtp-tree-attention-evolution.git"
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
DEFAULT_REPORT_PORT = 8000
TERMINAL_STATUSES = {"succeeded", "failed"}
PROXY_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

DEFAULT_GPU_TYPES = {
    "3090": ["NVIDIA GeForce RTX 3090"],
    "4090": ["NVIDIA GeForce RTX 4090"],
    "h100": ["NVIDIA H100 80GB HBM3"],
}

INSTALL_PROFILES = ("auto", "pinned", "runpod-pytorch", "runpod-vllm")


REMOTE_WORKER = r"""
import base64
import collections
import json
import os
import pathlib
import shutil
import shlex
import subprocess
from datetime import datetime, timezone


cfg = json.loads(base64.b64decode(os.environ["MTP_RUNPOD_CONFIG_B64"]).decode())
artifacts_dir = pathlib.Path(cfg["artifacts_dir"])
repo_dir = pathlib.Path(cfg["repo_dir"])
report_path = artifacts_dir / "report.json"
output_path = artifacts_dir / "output.log"
started_at = datetime.now(timezone.utc).isoformat()
commands = []
tail = collections.deque(maxlen=120)


class StepFailed(RuntimeError):
    def __init__(self, name, exit_code):
        super().__init__(f"{name} failed with exit code {exit_code}")
        self.name = name
        self.exit_code = exit_code


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def write_report(status, phase, **extra):
    payload = {
        "status": status,
        "phase": phase,
        "started_at": started_at,
        "updated_at": utc_now(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "repo_url": cfg["repo_url"],
        "ref": cfg.get("ref") or None,
        "benchmark_command": cfg["benchmark_command"],
        "install_profile": cfg.get("install_profile"),
        "preflight_command": cfg.get("preflight_command") or None,
        "install_command": cfg.get("install_command") or None,
        "artifacts": {
            "report": "report.json",
            "output": "output.log",
            "http_log": "http.log",
        },
        "commands": commands,
        "tail": list(tail),
    }
    payload.update(extra)
    atomic_write_json(report_path, payload)


def append_log(line=""):
    with output_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(line + "\n")
    tail.append(line[-2000:])


def command_display(command):
    if isinstance(command, str):
        return command
    return shlex.join(str(part) for part in command)


def run_step(name, command, *, cwd=None, shell=False):
    record = {
        "name": name,
        "command": command_display(command),
        "cwd": None if cwd is None else str(cwd),
        "started_at": utc_now(),
        "exit_code": None,
    }
    commands.append(record)
    write_report("running", f"running {name}", current_command=record)
    append_log(f"$ {record['command']}")
    proc = subprocess.Popen(
        command,
        cwd=None if cwd is None else str(cwd),
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        append_log(raw_line.rstrip("\n"))
    exit_code = proc.wait()
    record["exit_code"] = exit_code
    record["finished_at"] = utc_now()
    write_report("running", f"completed {name}", current_command=record)
    if exit_code != 0:
        raise StepFailed(name, exit_code)


def main():
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    write_report("running", "starting")

    try:
        run_step("python version", [cfg["python"], "--version"])
        run_step("git version", ["git", "--version"])

        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        run_step("clone repo", ["git", "clone", "--depth", "1", cfg["repo_url"], str(repo_dir)])

        if cfg.get("ref"):
            run_step("fetch ref", ["git", "fetch", "--depth", "1", "origin", cfg["ref"]], cwd=repo_dir)
            run_step("checkout ref", ["git", "checkout", "FETCH_HEAD"], cwd=repo_dir)

        if cfg.get("install_command"):
            run_step("install dependencies", cfg["install_command"], cwd=repo_dir, shell=True)

        for index, command in enumerate(cfg.get("extra_setup_commands") or [], start=1):
            run_step(f"extra setup {index}", command, cwd=repo_dir, shell=True)

        if cfg.get("preflight_command"):
            run_step("preflight", cfg["preflight_command"], cwd=repo_dir, shell=True)

        run_step("benchmark", cfg["benchmark_command"], cwd=repo_dir, shell=True)
    except StepFailed as exc:
        write_report("failed", f"failed {exc.name}", exit_code=exc.exit_code, error=str(exc))
        return 0
    except Exception as exc:
        write_report("failed", "failed", exit_code=1, error=f"{type(exc).__name__}: {exc}")
        return 0

    write_report("succeeded", "complete", exit_code=0)
    return 0


raise SystemExit(main())
"""


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def resolve_api_key(env_file: Path) -> str:
    for key_name in ("RUNPOD_API_KEY", "RUNPOD"):
        value = os.environ.get(key_name)
        if value:
            return value
    file_values = parse_env_file(env_file)
    for key_name in ("RUNPOD_API_KEY", "RUNPOD"):
        value = file_values.get(key_name)
        if value:
            return value
    raise SystemExit(
        f"RunPod API key not found. Set RUNPOD_API_KEY or RUNPOD in {env_file}."
    )


def default_benchmark_command(args: argparse.Namespace) -> str:
    command = [
        args.python,
        "-m",
        "bench.run_benchmark",
        "--gpu",
        args.gpu,
    ]
    if args.gpu == "h100":
        command.extend(["--flashmla-mode", args.flashmla_mode])
    if args.remote_dry_run:
        command.append("--dry-run")
    command.extend(args.benchmark_extra)
    return shlex.join(command)


def resolve_install_profile(args: argparse.Namespace) -> str:
    if args.install_profile != "auto":
        return args.install_profile
    if args.gpu == "h100":
        return "runpod-vllm"
    return "runpod-pytorch"


def build_install_command(args: argparse.Namespace, python: str) -> str | None:
    if args.skip_install:
        return None

    quoted_python = shlex.quote(python)
    profile = resolve_install_profile(args)
    if profile == "pinned":
        return (
            f"{quoted_python} -m pip install --upgrade pip && "
            f"{quoted_python} -m pip install -r requirements.txt"
        )
    if profile == "runpod-pytorch":
        return (
            f"{quoted_python} -m pip install --upgrade pip && "
            f"{quoted_python} -m pip install -r requirements-runpod.txt"
        )
    if profile == "runpod-vllm":
        return (
            f"{quoted_python} -m pip install --upgrade pip uv && "
            "uv pip install --system vllm --torch-backend=auto"
        )
    raise ValueError(f"unknown install profile {profile!r}")


def build_remote_config(args: argparse.Namespace) -> dict[str, Any]:
    python = args.python
    install_command = build_install_command(args, python)

    preflight_command = None
    if not args.skip_preflight:
        quoted_python = shlex.quote(python)
        preflight_command = (
            f"{quoted_python} -m py_compile bench/*.py tools/*.py tests/*.py && "
            f"{quoted_python} -m unittest discover tests"
        )

    return {
        "repo_url": args.repo_url,
        "ref": args.ref,
        "python": python,
        "artifacts_dir": "/workspace/mtp-runpod-artifacts",
        "repo_dir": "/workspace/mtp-benchmark",
        "install_command": install_command,
        "install_profile": resolve_install_profile(args),
        "extra_setup_commands": args.extra_setup_command,
        "preflight_command": preflight_command,
        "benchmark_command": args.benchmark_command or default_benchmark_command(args),
    }


def build_remote_start_command(remote_config: dict[str, Any], report_port: int) -> str:
    encoded_config = base64.b64encode(json.dumps(remote_config).encode()).decode()
    artifacts_dir = shlex.quote(remote_config["artifacts_dir"])
    python = shlex.quote(remote_config["python"])
    worker = REMOTE_WORKER.strip()
    return (
        "set -u\n"
        f"mkdir -p {artifacts_dir}\n"
        f"{python} -m http.server {report_port} --bind 0.0.0.0 --directory "
        f"{artifacts_dir} > {artifacts_dir}/http.log 2>&1 &\n"
        f"export MTP_RUNPOD_CONFIG_B64={shlex.quote(encoded_config)}\n"
        f"{python} - <<'PY'\n"
        f"{worker}\n"
        "PY\n"
        "sleep infinity\n"
    )


def build_pod_payload(args: argparse.Namespace, start_command: str) -> dict[str, Any]:
    gpu_types = args.gpu_type or DEFAULT_GPU_TYPES[args.gpu]
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    payload: dict[str, Any] = {
        "name": args.name or f"mtp-benchmark-{args.gpu}-{now}",
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": args.gpu_count,
        "imageName": args.image,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": "/workspace",
        "ports": [f"{args.report_port}/http"],
        "supportPublicIp": args.support_public_ip,
        "interruptible": args.interruptible,
        "dockerEntrypoint": ["bash", "-lc"],
        "dockerStartCmd": [start_command],
    }
    if args.allowed_cuda:
        payload["allowedCudaVersions"] = args.allowed_cuda
    if args.data_center:
        payload["dataCenterIds"] = args.data_center
        payload["dataCenterPriority"] = "availability"
    return payload


def request_json(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": PROXY_USER_AGENT}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    if api_key is not None:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    if not body:
        return {}
    return json.loads(body.decode())


def fetch_bytes(url: str, *, timeout: int = 60) -> bytes:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--fail-with-body",
        "--max-time",
        str(timeout),
        "--user-agent",
        PROXY_USER_AGENT,
        "--header",
        "Accept: */*",
        "--header",
        "Accept-Language: en-US,en;q=0.9",
        url,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True)
    except FileNotFoundError:
        result = None

    if result is not None:
        if result.returncode == 0:
            return result.stdout
        detail = result.stderr.decode("utf-8", errors="replace")
        body = result.stdout.decode("utf-8", errors="replace")
        raise RuntimeError(f"curl failed with exit {result.returncode}: {detail}{body}")

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": PROXY_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as exc:
        raise RuntimeError(f"GET {url} failed: {exc}") from exc


def fetch_report_json(url: str, *, timeout: int = 60) -> dict[str, Any]:
    try:
        return json.loads(fetch_bytes(url, timeout=timeout).decode())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"GET {url} did not return JSON: {exc}") from exc


def report_url_for_pod(pod_id: str, report_port: int) -> str:
    return f"https://{pod_id}-{report_port}.proxy.runpod.net/report.json"


def artifacts_base_url(pod_id: str, report_port: int) -> str:
    return f"https://{pod_id}-{report_port}.proxy.runpod.net"


def poll_report(report_url: str, *, timeout_seconds: int, interval_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_marker: tuple[str | None, str | None] | None = None
    last_error_at = 0.0
    while time.monotonic() < deadline:
        try:
            report = fetch_report_json(report_url, timeout=20)
        except RuntimeError as exc:
            now = time.monotonic()
            if now - last_error_at > max(interval_seconds, 30):
                print(f"Waiting for report endpoint: {exc}", file=sys.stderr)
                last_error_at = now
            time.sleep(interval_seconds)
            continue

        marker = (report.get("status"), report.get("phase"))
        if marker != last_marker:
            print(f"Remote status: {marker[0]} - {marker[1]}", file=sys.stderr)
            last_marker = marker
        if report.get("status") in TERMINAL_STATUSES:
            return report
        time.sleep(interval_seconds)

    raise TimeoutError(f"Timed out waiting for {report_url}")


def save_artifacts(
    report: dict[str, Any],
    *,
    pod_id: str,
    report_port: int,
    output_dir: Path,
) -> Path:
    run_dir = output_dir / f"runpod-{pod_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    base_url = artifacts_base_url(pod_id, report_port)
    for artifact_name in ("output.log", "http.log"):
        try:
            data = fetch_bytes(f"{base_url}/{artifact_name}", timeout=60)
        except Exception as exc:  # best effort; report.json is the source of truth
            (run_dir / f"{artifact_name}.fetch_error.txt").write_text(str(exc))
            continue
        (run_dir / artifact_name).write_bytes(data)
    return run_dir


def delete_pod(api_base: str, api_key: str, pod_id: str) -> None:
    request_json("DELETE", f"{api_base}/pods/{pod_id}", api_key=api_key, timeout=60)


def print_summary(report: dict[str, Any], run_dir: Path | None, pod_id: str, report_port: int) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
    if run_dir is not None:
        print(f"\nSaved artifacts: {run_dir}")
    print(f"Report URL: {report_url_for_pod(pod_id, report_port)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-base", default=RUNPOD_API_BASE)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--ref", default=None, help="Optional branch, tag, or commit to fetch after cloning.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--gpu", choices=("3090", "4090", "h100"), default="h100")
    parser.add_argument("--gpu-type", action="append", default=None)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"), default="SECURE")
    parser.add_argument("--interruptible", action="store_true")
    parser.add_argument("--support-public-ip", action="store_true")
    parser.add_argument("--container-disk-gb", type=int, default=80)
    parser.add_argument("--volume-gb", type=int, default=20)
    parser.add_argument("--allowed-cuda", action="append", default=None)
    parser.add_argument("--data-center", action="append", default=None)
    parser.add_argument("--report-port", type=int, default=DEFAULT_REPORT_PORT)
    parser.add_argument("--install-profile", choices=INSTALL_PROFILES, default="auto")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--extra-setup-command", action="append", default=[])
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--remote-dry-run", action="store_true")
    parser.add_argument("--flashmla-mode", choices=("bf16-prefill", "fp8-decode"), default="bf16-prefill")
    parser.add_argument("--benchmark-extra", action="append", default=[])
    parser.add_argument("--benchmark-command", default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runpod"))
    parser.add_argument("--terminate-on-complete", action="store_true")
    parser.add_argument(
        "--local-dry-run",
        action="store_true",
        help="Print the RunPod payload without creating a pod.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    remote_config = build_remote_config(args)
    start_command = build_remote_start_command(remote_config, args.report_port)
    payload = build_pod_payload(args, start_command)

    if args.local_dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    api_key = resolve_api_key(args.env_file)
    pod = request_json("POST", f"{args.api_base}/pods", api_key=api_key, payload=payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise SystemExit(f"RunPod create response did not include a pod id: {pod}")

    print(f"Created RunPod pod: {pod_id}", file=sys.stderr)
    print(f"Report URL: {report_url_for_pod(pod_id, args.report_port)}", file=sys.stderr)

    if args.no_wait:
        print(json.dumps({"pod_id": pod_id, "report_url": report_url_for_pod(pod_id, args.report_port)}))
        return 0

    report: dict[str, Any] | None = None
    run_dir: Path | None = None
    try:
        report = poll_report(
            report_url_for_pod(pod_id, args.report_port),
            timeout_seconds=args.timeout_minutes * 60,
            interval_seconds=args.poll_seconds,
        )
        run_dir = save_artifacts(
            report,
            pod_id=pod_id,
            report_port=args.report_port,
            output_dir=args.output_dir,
        )
        print_summary(report, run_dir, pod_id, args.report_port)
        return 0 if report.get("status") == "succeeded" else 1
    finally:
        if args.terminate_on_complete:
            try:
                delete_pod(args.api_base, api_key, pod_id)
                print(f"Deleted RunPod pod: {pod_id}", file=sys.stderr)
            except Exception as exc:
                print(f"Failed to delete RunPod pod {pod_id}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
