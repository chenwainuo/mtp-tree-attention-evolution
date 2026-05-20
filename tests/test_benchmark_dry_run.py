from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from bench import bench_flashmla_sparse, bench_tree_attention, run_benchmark
from bench.common import normalize_support_flag, require_target
from bench.flashmla_adapter import (
    ensure_decode_signature,
    ensure_sparse_prefill_signature,
    flashmla_support_status,
    import_flashmla_symbols,
    load_extraction_report,
)
from bench.fp8 import dequantize_scalar, scale_from_amax
from bench.shapes import get_v4_flash_shapes


class BenchmarkDryRunTests(unittest.TestCase):
    def test_target_defaults_are_distinct(self) -> None:
        self.assertFalse(require_target("3090").uses_fp8_kv)
        self.assertTrue(require_target("4090").uses_fp8_kv)
        self.assertTrue(require_target("h100").requires_flashmla)

    def test_chain_mask_semantics(self) -> None:
        ctx_len = 8
        self.assertTrue(bench_tree_attention.chain_allows(ctx_len, 0, 8))
        self.assertFalse(bench_tree_attention.chain_allows(ctx_len, 0, 9))
        self.assertTrue(bench_tree_attention.chain_allows(ctx_len, 3, 11))
        self.assertFalse(bench_tree_attention.chain_allows(ctx_len, 3, 12))

    def test_proxy_head_dim_uses_dense_compatible_shape(self) -> None:
        shapes = get_v4_flash_shapes()
        self.assertEqual(bench_tree_attention.default_proxy_head_dim(shapes), 128)
        self.assertLess(bench_tree_attention.default_proxy_head_dim(shapes), shapes.head_dim)

    def test_fp8_scalar_helpers(self) -> None:
        self.assertAlmostEqual(scale_from_amax(448.0), 1.0)
        self.assertAlmostEqual(dequantize_scalar(10.0, 0.5), 5.0)
        with self.assertRaises(ValueError):
            scale_from_amax(-1.0)

    def test_normalize_support_flag(self) -> None:
        self.assertEqual(normalize_support_flag(True), (True, None))
        self.assertEqual(
            normalize_support_flag((False, "sm_86 unsupported")),
            (False, "sm_86 unsupported"),
        )

    def test_flashmla_support_status_finds_ops_checker(self) -> None:
        backend = ModuleType("fake_backend")
        ops = ModuleType("fake_ops")

        def is_flashmla_sparse_supported():
            return False, "extension missing"

        ops.is_flashmla_sparse_supported = is_flashmla_sparse_supported
        with patch(
            "bench.flashmla_adapter.FLASHMLA_MODULE_CANDIDATES",
            ("fake_backend", "fake_ops"),
        ):
            with patch.dict(sys.modules, {"fake_backend": backend, "fake_ops": ops}):
                self.assertEqual(
                    flashmla_support_status(backend),
                    (False, "extension missing"),
                )

    def test_flashmla_symbol_import_skips_empty_modules(self) -> None:
        backend = ModuleType("fake_backend")
        ops = ModuleType("fake_ops")
        ops.flash_mla_sparse_fwd = lambda q, kv, indices, scale: (q, kv, indices, scale)
        with patch(
            "bench.flashmla_adapter.FLASHMLA_MODULE_CANDIDATES",
            ("fake_backend", "fake_ops"),
        ):
            with patch.dict(sys.modules, {"fake_backend": backend, "fake_ops": ops}):
                symbols = import_flashmla_symbols()
        self.assertEqual(symbols.module_name, "fake_ops")
        self.assertIs(symbols.flash_mla_sparse_fwd, ops.flash_mla_sparse_fwd)

    def test_flashmla_keyword_call_filters_to_signature(self) -> None:
        def fn(q, *, head_dim_v, indices=None):
            return q, head_dim_v, indices

        result = bench_flashmla_sparse.call_with_supported_kwargs(
            fn,
            {"q": "q", "head_dim_v": 512, "indices": "idx", "ignored": "x"},
            ("fallback",),
        )
        self.assertEqual(result, ("q", 512, "idx"))

    def test_primary_tensor_unwraps_tuple_and_list(self) -> None:
        self.assertEqual(bench_flashmla_sparse.primary_tensor(("out", "lse")), "out")
        self.assertEqual(bench_flashmla_sparse.primary_tensor(["out", "lse"]), "out")
        with self.assertRaises(RuntimeError):
            bench_flashmla_sparse.primary_tensor([])

    def test_extraction_report_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flashmla_extraction.json"
            path.write_text(json.dumps({"support_flags": {"supported": True}}))
            report = load_extraction_report(path)
            self.assertEqual(report.support_flags, {"supported": True})

    def test_fake_flashmla_signature_checks(self) -> None:
        def sparse_ok(q, kv, indices, sm_scale):
            return q, kv, indices, sm_scale

        def sparse_bad(q, kv):
            return q, kv

        def decode_ok(a, b, c, d, e, f, g, h, i, j):
            return a, b, c, d, e, f, g, h, i, j

        ensure_sparse_prefill_signature(sparse_ok)
        ensure_decode_signature(decode_ok)
        with self.assertRaises(RuntimeError):
            ensure_sparse_prefill_signature(sparse_bad)

    def test_unified_dry_runs(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream):
            run_benchmark.main(["--gpu", "3090", "--dry-run"])
        output = stream.getvalue()
        self.assertIn("dense-flashinfer-fp16-proxy", output)
        self.assertIn("head_dim: 128", output)

        stream = io.StringIO()
        with redirect_stdout(stream):
            run_benchmark.main(["--gpu", "4090", "--dry-run"])
        output = stream.getvalue()
        self.assertIn("dense-flashinfer-fp8-kv-proxy", output)
        self.assertIn("fp8_kv: True", output)

        stream = io.StringIO()
        with redirect_stdout(stream):
            run_benchmark.main(["--gpu", "h100", "--dry-run"])
        output = stream.getvalue()
        self.assertIn("flashmla-sparse", output)
        self.assertIn("impl: flashmla", output)
        self.assertIn("cache_bytes_per_token: 656", output)

        stream = io.StringIO()
        with redirect_stdout(stream):
            run_benchmark.main(["--gpu", "h100", "--flashmla-impl", "triton", "--dry-run"])
        output = stream.getvalue()
        self.assertIn("impl: triton", output)
        self.assertIn("triton_layout: grouped", output)
        self.assertIn("triton_block_h: 16", output)
        self.assertIn("triton_block_k: 32", output)


if __name__ == "__main__":
    unittest.main()
