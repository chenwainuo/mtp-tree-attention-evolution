from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from tools import evolve_h100


class EvolveH100Tests(unittest.TestCase):
    def test_parse_benchmark_output(self) -> None:
        output = "\nCorrectness: PASS (allclose)\nRuntime: 21.50 us\n"
        runtime_us, correctness = evolve_h100.parse_benchmark_output(output)
        self.assertEqual(runtime_us, 21.50)
        self.assertEqual(correctness, "PASS (allclose)")
        self.assertTrue(evolve_h100.correctness_passed(correctness))

    def test_parse_candidate_spec(self) -> None:
        candidate = evolve_h100.parse_candidate_spec("k=32,d=64,v=128,warps=8")
        self.assertEqual(candidate.name, "triton-k32-d64-v128-w8")
        self.assertEqual(candidate.block_k, 32)
        self.assertEqual(candidate.block_d, 64)
        self.assertEqual(candidate.block_v, 128)
        self.assertEqual(candidate.warps, 8)

    def test_parse_candidate_spec_rejects_non_power_of_two(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            evolve_h100.parse_candidate_spec("k=48,d=64,v=64,warps=4")

    def test_build_runpod_command_includes_candidate_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            args = argparse.Namespace(
                python="python3",
                repo_root=repo_root,
                ref="abc123",
                repo_url="https://github.com/example/repo.git",
                env_file=repo_root / ".env",
                timeout_minutes=12,
                poll_seconds=3,
                keep_pods=False,
                warmup=5,
                rep=7,
            )
            candidate = evolve_h100.Candidate(
                name="triton-k32-d64-v64-w4",
                block_k=32,
                block_d=64,
                block_v=64,
                warps=4,
            )
            command, benchmark_command = evolve_h100.build_runpod_command(
                candidate,
                args,
                repo_root / "out",
            )
        self.assertIn("--terminate-on-complete", command)
        self.assertIn("--benchmark-command", command)
        self.assertIn("--ref", command)
        self.assertIn("abc123", command)
        self.assertIn("--flashmla-impl triton", benchmark_command)
        self.assertIn("--triton-block-k 32", benchmark_command)
        self.assertIn("--rep 7", benchmark_command)


if __name__ == "__main__":
    unittest.main()
