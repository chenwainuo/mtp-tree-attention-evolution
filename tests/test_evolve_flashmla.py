from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from tools import evolve_flashmla, flashmla_source_loop, source_build_flashmla


class EvolveFlashMLATests(unittest.TestCase):
    def test_normalize_source_ref_maps_release_tag(self) -> None:
        self.assertEqual(
            evolve_flashmla.normalize_source_ref("v0.21.0"),
            "releases/v0.21.0",
        )
        self.assertEqual(
            evolve_flashmla.normalize_source_ref("main"),
            "main",
        )

    def test_flashmla_source_loop_command_uses_patch_candidate(self) -> None:
        args = argparse.Namespace(
            python="python3",
            mode="bf16-prefill",
            baseline_us=23.29,
            min_speedup_pct=2.0,
            source_baseline_max_drift_pct=20.0,
            source_ref="v0.21.0",
            flashmla_ref="auto",
            max_candidates=1,
            warmup=5,
            rep=7,
            max_jobs=4,
            candidate=[Path("patches/flashmla/bf16_prefill/sm90_btopk128.patch")],
        )
        command = evolve_flashmla.flashmla_source_loop_command(args)
        self.assertIn("tools.flashmla_source_loop", command)
        self.assertIn("--source-ref releases/v0.21.0", command)
        self.assertIn("--source-baseline-max-drift-pct 20", command)
        self.assertIn("patches/flashmla/bf16_prefill/sm90_btopk128.patch", command)
        self.assertIn("--rep 7", command)

    def test_build_runpod_command_skips_default_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                python="python3",
                repo_root=root,
                repo_url="https://github.com/example/repo.git",
                ref="abc123",
                env_file=root / ".env",
                timeout_minutes=10,
                poll_seconds=2,
                mode="bf16-prefill",
                baseline_us=23.29,
                min_speedup_pct=2.0,
                source_baseline_max_drift_pct=20.0,
                source_ref="v0.21.0",
                flashmla_ref="auto",
                max_candidates=1,
                warmup=5,
                rep=7,
                max_jobs=4,
                candidate=[Path("patches/flashmla/bf16_prefill/sm90_btopk128.patch")],
                terminate_on_complete=True,
            )
            command, benchmark_command = evolve_flashmla.build_runpod_command(
                args,
                root / "session",
            )
        self.assertIn("--skip-extract-flashmla", command)
        self.assertIn("runpod-vllm-source", command)
        self.assertIn("--terminate-on-complete", command)
        self.assertIn("--benchmark-command", command)
        self.assertIn("tools.flashmla_source_loop", benchmark_command)

    def test_parse_benchmark_output(self) -> None:
        output = "Correctness: PASS (allclose)\nRuntime: 22.40 us\n"
        runtime, correctness = flashmla_source_loop.parse_benchmark_output(output)
        self.assertEqual(runtime, 22.40)
        self.assertEqual(correctness, "PASS (allclose)")
        self.assertTrue(flashmla_source_loop.correctness_passed(correctness))

    def test_drift_pct(self) -> None:
        self.assertAlmostEqual(
            flashmla_source_loop.drift_pct(23.29, 25.619),
            10.0,
            places=5,
        )

    def test_source_validation_requires_expected_btopk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / source_build_flashmla.FLASHMLA_CONFIG_PATH
            config.parent.mkdir(parents=True)
            config.write_text(
                "template<int D_QK, bool HAVE_TOPK_LENGTH>\n"
                "static constexpr int B_TOPK = 64;    // TopK block size\n"
            )
            content = source_build_flashmla.validate_flashmla_source(root)
        self.assertIn("B_TOPK = 64", content)

    def test_patch_vllm_setup_for_flashmla_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup = root / "setup.py"
            setup.write_text(
                source_build_flashmla.SETUP_FLASHMLA_CONDITION
                + "\n"
                + "if _no_device():\n"
                + "    ext_modules = []\n"
                + "        if _is_cuda() or _is_hip():\n"
                + "            pass\n"
                + "        if _is_cuda():\n"
                + "            # copy vendored deep_gemm package\n"
                + "            pass\n"
            )
            cmake = root / "CMakeLists.txt"
            cmake.write_text(
                "# For CUDA and HIP builds also build the triton_kernels external package.\n"
                'if(VLLM_GPU_LANG STREQUAL "CUDA" OR VLLM_GPU_LANG STREQUAL "HIP")\n'
                "    include(cmake/external_projects/triton_kernels.cmake)\n"
                "endif()\n"
                "\n"
                "# For CUDA we also build and ship some external projects.\n"
                'if (VLLM_GPU_LANG STREQUAL "CUDA")\n'
                "    include(cmake/external_projects/deepgemm.cmake)\n"
                "    include(cmake/external_projects/flashmla.cmake)\n"
                "    include(cmake/external_projects/qutlass.cmake)\n"
                "\n"
                "    # vllm-flash-attn should be last as it overwrites some CMake functions\n"
                "    include(cmake/external_projects/vllm_flash_attn.cmake)\n"
                "endif ()\n"
            )
            source_build_flashmla.patch_vllm_setup_for_flashmla_overlay(root)
            content = setup.read_text()
            cmake_content = cmake.read_text()

        self.assertIn("MTP_FORCE_FLASHMLA_EXTENSIONS", content)
        self.assertIn("MTP_FLASHMLA_ONLY_BUILD", content)
        self.assertIn('flashmla_targets = {"vllm._flashmla_C"', content)
        self.assertIn('os.getenv("MTP_FLASHMLA_ONLY_BUILD") != "1"', content)
        self.assertIn("MTP_FLASHMLA_ONLY_BUILD", cmake_content)
        self.assertIn("include(cmake/external_projects/flashmla.cmake)", cmake_content)

    def test_copy_flashmla_overlay_copies_extensions_and_interface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_pkg = root / "source" / "vllm"
            package_dir = root / "site" / "vllm"
            source_pkg.mkdir(parents=True)
            package_dir.mkdir(parents=True)
            (source_pkg / "_flashmla_C.abi3.so").write_text("core")
            (source_pkg / "_flashmla_extension_C.abi3.so").write_text("extension")
            interface = source_pkg / "third_party" / "flashmla" / "flash_mla_interface.py"
            interface.parent.mkdir(parents=True)
            interface.write_text("interface")
            copied = source_build_flashmla.copy_flashmla_overlay(root / "source", package_dir)

            self.assertEqual(len(copied), 3)
            self.assertEqual((package_dir / "_flashmla_C.abi3.so").read_text(), "core")
            self.assertEqual(
                (package_dir / "third_party" / "flashmla" / "flash_mla_interface.py").read_text(),
                "interface",
            )

    def test_git_clone_ref_initializes_submodules(self) -> None:
        calls: list[tuple[list[str], Path | None]] = []

        def fake_run_command(
            command: list[str],
            *,
            cwd: Path | None = None,
            log_path: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            del log_path, env
            calls.append((command, cwd))

        original = source_build_flashmla.run_command
        source_build_flashmla.run_command = fake_run_command
        try:
            with tempfile.TemporaryDirectory() as tmp:
                source_build_flashmla.git_clone_ref(
                    "https://example.invalid/repo.git",
                    "releases/v0.21.0",
                    Path(tmp) / "repo",
                )
        finally:
            source_build_flashmla.run_command = original

        self.assertIn(
            (["git", "submodule", "update", "--init", "--recursive"], calls[-1][1]),
            calls,
        )


if __name__ == "__main__":
    unittest.main()
