from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools import runpod_benchmark


class RunpodBenchmarkLauncherTests(unittest.TestCase):
    def test_parse_env_file_supports_runpod_key_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("RUNPOD='secret-value'\n")
            self.assertEqual(runpod_benchmark.parse_env_file(path)["RUNPOD"], "secret-value")

    def test_default_h100_command_uses_flashmla_mode(self) -> None:
        args = argparse.Namespace(
            python="python3",
            gpu="h100",
            flashmla_mode="fp8-decode",
            remote_dry_run=False,
            benchmark_extra=[],
        )
        command = runpod_benchmark.default_benchmark_command(args)
        self.assertIn("--gpu h100", command)
        self.assertIn("--flashmla-mode fp8-decode", command)

    def test_auto_install_profile_tracks_gpu_path(self) -> None:
        args = argparse.Namespace(gpu="4090", install_profile="auto")
        self.assertEqual(runpod_benchmark.resolve_install_profile(args), "runpod-pytorch")

        args = argparse.Namespace(gpu="h100", install_profile="auto")
        self.assertEqual(runpod_benchmark.resolve_install_profile(args), "runpod-vllm")

    def test_runpod_install_profile_does_not_reinstall_torch(self) -> None:
        args = argparse.Namespace(
            gpu="4090",
            install_profile="runpod-pytorch",
            skip_install=False,
        )
        command = runpod_benchmark.build_install_command(args, "python3")
        self.assertIsNotNone(command)
        self.assertIn("requirements-runpod.txt", command)
        self.assertNotIn("requirements.txt", command)

    def test_h100_install_profile_installs_vllm(self) -> None:
        args = argparse.Namespace(
            gpu="h100",
            install_profile="runpod-vllm",
            skip_install=False,
        )
        command = runpod_benchmark.build_install_command(args, "python3")
        self.assertIsNotNone(command)
        self.assertIn("uv pip install --system vllm --torch-backend=auto", command)

    def test_h100_remote_config_extracts_flashmla_artifacts(self) -> None:
        args = argparse.Namespace(
            python="python3",
            gpu="h100",
            install_profile="auto",
            skip_install=True,
            skip_preflight=True,
            remote_dry_run=False,
            repo_url="https://github.com/example/repo.git",
            ref=None,
            extra_setup_command=[],
            benchmark_command=None,
            flashmla_mode="bf16-prefill",
            benchmark_extra=[],
        )
        config = runpod_benchmark.build_remote_config(args)
        self.assertIn("tools.extract_flashmla", config["extract_flashmla_command"])
        self.assertIn(
            "/workspace/mtp-runpod-artifacts",
            config["extract_flashmla_command"],
        )

    def test_pod_payload_exposes_report_port_and_gpu_type(self) -> None:
        args = argparse.Namespace(
            gpu="4090",
            gpu_type=None,
            name="test-pod",
            cloud_type="SECURE",
            gpu_count=1,
            image="image",
            container_disk_gb=80,
            volume_gb=20,
            report_port=8000,
            support_public_ip=False,
            interruptible=False,
            allowed_cuda=None,
            data_center=None,
        )
        payload = runpod_benchmark.build_pod_payload(args, "echo hello")
        self.assertEqual(payload["gpuTypeIds"], ["NVIDIA GeForce RTX 4090"])
        self.assertEqual(payload["ports"], ["8000/http"])
        self.assertEqual(payload["dockerEntrypoint"], ["bash", "-lc"])

    def test_remote_start_command_contains_decodable_config(self) -> None:
        config = {
            "repo_url": "https://github.com/example/repo.git",
            "ref": None,
            "python": "python3",
            "artifacts_dir": "/workspace/mtp-runpod-artifacts",
            "repo_dir": "/workspace/mtp-benchmark",
            "install_command": None,
            "extra_setup_commands": [],
            "preflight_command": None,
            "benchmark_command": "python3 -m bench.run_benchmark --gpu 4090",
        }
        command = runpod_benchmark.build_remote_start_command(config, 8000)
        self.assertIn("python3 -m http.server 8000", command)
        prefix = "export MTP_RUNPOD_CONFIG_B64="
        encoded_line = next(line for line in command.splitlines() if line.startswith(prefix))
        encoded = encoded_line[len(prefix) :].strip("'")
        decoded = json.loads(runpod_benchmark.base64.b64decode(encoded).decode())
        self.assertEqual(decoded["repo_url"], config["repo_url"])


if __name__ == "__main__":
    unittest.main()
