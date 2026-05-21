from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tools import bench_vllm_realtime


class MockSSEState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.fail = False
        self.delay_s = 0.0

    def chunks(self, path: str) -> list[dict[str, Any]]:
        if path.endswith("/chat/completions"):
            return [
                {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "A"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "B"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}},
            ]
        return [
            {"choices": [{"text": "X", "finish_reason": None}]},
            {"choices": [{"text": "Y", "finish_reason": "length"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}},
        ]


class MockSSEServer:
    def __init__(self, state: MockSSEState) -> None:
        self.state = state
        state_ref = state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                payload = json.loads(body.decode("utf-8"))
                with state_ref.lock:
                    state_ref.requests.append(payload)
                    state_ref.active += 1
                    state_ref.max_active = max(state_ref.max_active, state_ref.active)
                try:
                    if state_ref.fail:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(b"server error")
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in state_ref.chunks(self.path):
                        time.sleep(state_ref.delay_s)
                        self.wfile.write(
                            f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with state_ref.lock:
                        state_ref.active -= 1

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def write_prompts(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def args_for(base_url: str, prompts_file: Path, output_json: Path, **overrides: Any) -> Any:
    values = {
        "base_url": base_url,
        "model": "mock-model",
        "prompts_file": prompts_file,
        "endpoint": "chat.completions",
        "concurrency": 1,
        "max_tokens": 4,
        "temperature": 0.0,
        "warmup_requests": 0,
        "output_json": output_json,
        "workload": None,
        "timeout_s": 10.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class BenchVllmRealtimeTests(unittest.TestCase):
    def test_chat_streaming_metrics_and_schema(self) -> None:
        state = MockSSEState()
        state.delay_s = 0.002
        with tempfile.TemporaryDirectory() as tmp, MockSSEServer(state) as base_url:
            root = Path(tmp)
            prompts = root / "prompts.jsonl"
            output = root / "out.json"
            write_prompts(
                prompts,
                [{"id": "p1", "workload": "interactive_mixed", "prompt": "hello"}],
            )

            payload = bench_vllm_realtime.run_benchmark(
                args_for(base_url, prompts, output, warmup_requests=1)
            )
            self.assertTrue(output.exists())

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["counts"], {"total": 1, "success": 1, "errors": 0})
        self.assertEqual(payload["usage"]["completion_tokens"], 2)
        request = payload["requests"][0]
        self.assertIsNotNone(request["ttft_s"])
        self.assertEqual(request["streamed_chunks"], 2)
        self.assertEqual(len(request["inter_token_latency_s"]), 1)
        self.assertIsNotNone(payload["metrics"]["ttft_s"]["p50"])

    def test_completions_endpoint_uses_per_prompt_max_tokens(self) -> None:
        state = MockSSEState()
        with tempfile.TemporaryDirectory() as tmp, MockSSEServer(state) as base_url:
            root = Path(tmp)
            prompts = root / "prompts.jsonl"
            output = root / "out.json"
            write_prompts(
                prompts,
                [{"id": "p1", "workload": "decode_heavy_control", "prompt": "continue", "max_tokens": 7}],
            )

            payload = bench_vllm_realtime.run_benchmark(
                args_for(
                    base_url,
                    prompts,
                    output,
                    endpoint="completions",
                    max_tokens=3,
                )
            )

        self.assertEqual(payload["counts"]["success"], 1)
        self.assertEqual(state.requests[0]["max_tokens"], 7)
        self.assertEqual(state.requests[0]["prompt"], "continue")
        self.assertNotIn("messages", state.requests[0])
        self.assertEqual(payload["requests"][0]["finish_reason"], "length")

    def test_failed_requests_are_counted_without_percentiles(self) -> None:
        state = MockSSEState()
        state.fail = True
        with tempfile.TemporaryDirectory() as tmp, MockSSEServer(state) as base_url:
            root = Path(tmp)
            prompts = root / "prompts.jsonl"
            output = root / "out.json"
            write_prompts(prompts, [{"id": "p1", "workload": "w", "prompt": "hello"}])

            payload = bench_vllm_realtime.run_benchmark(args_for(base_url, prompts, output))

        self.assertEqual(payload["counts"], {"total": 1, "success": 0, "errors": 1})
        self.assertIsNone(payload["metrics"]["ttft_s"]["p50"])
        self.assertIn("HTTP 500", payload["requests"][0]["error"])

    def test_concurrency_limit_is_respected(self) -> None:
        state = MockSSEState()
        state.delay_s = 0.02
        with tempfile.TemporaryDirectory() as tmp, MockSSEServer(state) as base_url:
            root = Path(tmp)
            prompts = root / "prompts.jsonl"
            output = root / "out.json"
            write_prompts(
                prompts,
                [
                    {"id": f"p{i}", "workload": "interactive_mixed", "prompt": f"hello {i}"}
                    for i in range(5)
                ],
            )

            payload = bench_vllm_realtime.run_benchmark(
                args_for(base_url, prompts, output, concurrency=2)
            )

        self.assertEqual(payload["counts"]["success"], 5)
        self.assertGreaterEqual(state.max_active, 2)
        self.assertLessEqual(state.max_active, 2)

    def test_read_prompts_filters_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.jsonl"
            write_prompts(
                prompts,
                [
                    {"id": "a", "workload": "long_prefill_short_output", "prompt": "a"},
                    {"id": "b", "workload": "decode_heavy_control", "prompt": "b"},
                ],
            )
            selected = bench_vllm_realtime.read_prompts(prompts, "decode_heavy_control")
        self.assertEqual([prompt.id for prompt in selected], ["b"])


if __name__ == "__main__":
    unittest.main()
