"""Streaming benchmark client for vLLM's OpenAI-compatible server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
ENDPOINT_PATHS = {
    "chat.completions": "chat/completions",
    "completions": "completions",
}


@dataclass(frozen=True)
class PromptRecord:
    id: str
    workload: str
    prompt: str
    max_tokens: int | None = None


@dataclass
class RequestResult:
    prompt_id: str
    workload: str
    request_index: int
    status: str
    max_tokens: int
    http_status: int | None = None
    error: str | None = None
    ttft_s: float | None = None
    latency_s: float | None = None
    inter_token_latency_s: list[float] | None = None
    streamed_chunks: int = 0
    output_chars: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_prompts(path: Path, workload: str | None = None) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        missing = [key for key in ("id", "workload", "prompt") if key not in payload]
        if missing:
            raise ValueError(f"{path}:{line_number}: missing required keys {missing}")
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError(f"{path}:{line_number}: max_tokens must be a positive integer")
        record = PromptRecord(
            id=str(payload["id"]),
            workload=str(payload["workload"]),
            prompt=str(payload["prompt"]),
            max_tokens=max_tokens,
        )
        if workload is None or record.workload == workload:
            prompts.append(record)
    if not prompts:
        detail = "" if workload is None else f" for workload {workload!r}"
        raise ValueError(f"no prompts found in {path}{detail}")
    return prompts


def percentile(values: Iterable[float], pct: int) -> float | None:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    rank = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[min(rank, len(ordered) - 1)]


def percentile_summary(values: Iterable[float]) -> dict[str, float | None]:
    materialized = list(values)
    return {
        "p50": percentile(materialized, 50),
        "p90": percentile(materialized, 90),
        "p95": percentile(materialized, 95),
        "p99": percentile(materialized, 99),
    }


def endpoint_url(base_url: str, endpoint: str) -> str:
    try:
        path = ENDPOINT_PATHS[endpoint]
    except KeyError as exc:
        raise ValueError(f"unsupported endpoint {endpoint!r}") from exc
    return f"{base_url.rstrip('/')}/{path}"


def request_payload(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    base = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if endpoint == "chat.completions":
        base["messages"] = [{"role": "user", "content": prompt}]
    elif endpoint == "completions":
        base["prompt"] = prompt
    else:
        raise ValueError(f"unsupported endpoint {endpoint!r}")
    return base


def iter_sse_data(response: Any) -> Iterable[str]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def extract_content(chunk: dict[str, Any], endpoint: str) -> tuple[str, str | None]:
    choices = chunk.get("choices") or []
    if not choices:
        return "", None
    first = choices[0]
    finish_reason = first.get("finish_reason")
    if endpoint == "chat.completions":
        delta = first.get("delta") or {}
        content = delta.get("content") or ""
    else:
        content = first.get("text") or ""
    return str(content), None if finish_reason is None else str(finish_reason)


def apply_usage(result: RequestResult, usage: dict[str, Any] | None) -> None:
    if not isinstance(usage, dict):
        return
    if isinstance(usage.get("prompt_tokens"), int):
        result.prompt_tokens = usage["prompt_tokens"]
    if isinstance(usage.get("completion_tokens"), int):
        result.completion_tokens = usage["completion_tokens"]
    if isinstance(usage.get("total_tokens"), int):
        result.total_tokens = usage["total_tokens"]


def run_one_request(
    *,
    url: str,
    endpoint: str,
    model: str,
    prompt: PromptRecord,
    request_index: int,
    default_max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> RequestResult:
    max_tokens = prompt.max_tokens or default_max_tokens
    result = RequestResult(
        prompt_id=prompt.id,
        workload=prompt.workload,
        request_index=request_index,
        status="failed",
        max_tokens=max_tokens,
        inter_token_latency_s=[],
    )
    payload = request_payload(
        endpoint=endpoint,
        model=model,
        prompt=prompt.prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    started = time.monotonic()
    previous_content_at: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result.http_status = response.status
            for data in iter_sse_data(response):
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    result.error = f"invalid SSE JSON chunk: {exc}"
                    return result
                apply_usage(result, chunk.get("usage"))
                content, finish_reason = extract_content(chunk, endpoint)
                if finish_reason is not None:
                    result.finish_reason = finish_reason
                if not content:
                    continue
                now = time.monotonic()
                if result.ttft_s is None:
                    result.ttft_s = now - started
                if previous_content_at is not None:
                    assert result.inter_token_latency_s is not None
                    result.inter_token_latency_s.append(now - previous_content_at)
                previous_content_at = now
                result.streamed_chunks += 1
                result.output_chars += len(content)
    except urllib.error.HTTPError as exc:
        result.http_status = exc.code
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        finally:
            exc.close()
        result.error = f"HTTP {exc.code}: {detail[:1000]}"
        result.latency_s = time.monotonic() - started
        return result
    except Exception as exc:  # noqa: BLE001 - benchmark JSON should retain failures.
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = time.monotonic() - started
        return result

    result.latency_s = time.monotonic() - started
    result.status = "succeeded"
    return result


def run_batch(
    *,
    prompts: list[PromptRecord],
    url: str,
    endpoint: str,
    model: str,
    concurrency: int,
    default_max_tokens: int,
    temperature: float,
    timeout_s: float,
) -> list[RequestResult]:
    results_by_index: dict[int, RequestResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one_request,
                url=url,
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                request_index=index,
                default_max_tokens=default_max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            for index, prompt in enumerate(prompts)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results_by_index[result.request_index] = result
    return [results_by_index[index] for index in sorted(results_by_index)]


def warmup_prompts(prompts: list[PromptRecord], count: int) -> list[PromptRecord]:
    if count <= 0:
        return []
    return [prompts[index % len(prompts)] for index in range(count)]


def aggregate_results(
    *,
    results: list[RequestResult],
    duration_s: float,
) -> dict[str, Any]:
    successes = [result for result in results if result.status == "succeeded"]
    errors = [result for result in results if result.status != "succeeded"]
    ttft_values = [result.ttft_s for result in successes if result.ttft_s is not None]
    latency_values = [result.latency_s for result in successes if result.latency_s is not None]
    inter_token_values = [
        gap
        for result in successes
        for gap in (result.inter_token_latency_s or [])
    ]
    usage_completion = [
        result.completion_tokens for result in successes if result.completion_tokens is not None
    ]
    if len(usage_completion) == len(successes) and usage_completion:
        output_units = sum(usage_completion)
        output_unit_source = "usage_completion_tokens"
    elif usage_completion:
        output_units = sum(
            result.completion_tokens
            if result.completion_tokens is not None
            else result.streamed_chunks
            for result in successes
        )
        output_unit_source = "mixed_usage_completion_tokens_and_stream_chunks"
    else:
        output_units = sum(result.streamed_chunks for result in successes)
        output_unit_source = "stream_chunks"

    prompt_tokens = [result.prompt_tokens for result in successes if result.prompt_tokens is not None]
    total_tokens = [result.total_tokens for result in successes if result.total_tokens is not None]

    return {
        "counts": {
            "total": len(results),
            "success": len(successes),
            "errors": len(errors),
        },
        "usage": {
            "prompt_tokens": sum(prompt_tokens) if prompt_tokens else None,
            "completion_tokens": sum(usage_completion) if usage_completion else None,
            "total_tokens": sum(total_tokens) if total_tokens else None,
            "missing_usage_count": len(successes) - len(usage_completion),
        },
        "metrics": {
            "ttft_s": percentile_summary(ttft_values),
            "latency_s": percentile_summary(latency_values),
            "inter_token_latency_s": percentile_summary(inter_token_values),
            "request_throughput_per_s": len(successes) / duration_s if duration_s > 0 else None,
            "output_tokens_per_s": output_units / duration_s if duration_s > 0 else None,
            "output_token_count_source": output_unit_source,
        },
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    prompts = read_prompts(args.prompts_file, workload=args.workload)
    url = endpoint_url(args.base_url, args.endpoint)
    started_at = utc_now()

    warmup_results: list[RequestResult] = []
    warmups = warmup_prompts(prompts, args.warmup_requests)
    if warmups:
        warmup_results = run_batch(
            prompts=warmups,
            url=url,
            endpoint=args.endpoint,
            model=args.model,
            concurrency=args.concurrency,
            default_max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
        )

    started = time.monotonic()
    results = run_batch(
        prompts=prompts,
        url=url,
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        default_max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_s=args.timeout_s,
    )
    duration_s = time.monotonic() - started
    aggregate = aggregate_results(results=results, duration_s=duration_s)
    finished_at = utc_now()

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "config": {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "model": args.model,
            "prompts_file": str(args.prompts_file),
            "workload": args.workload,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "warmup_requests": args.warmup_requests,
            "timeout_s": args.timeout_s,
        },
        "warmup": {
            "total": len(warmup_results),
            "success": sum(1 for result in warmup_results if result.status == "succeeded"),
            "errors": sum(1 for result in warmup_results if result.status != "succeeded"),
        },
        **aggregate,
        "requests": [asdict(result) for result in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-file", required=True, type=Path)
    parser.add_argument("--endpoint", choices=tuple(ENDPOINT_PATHS), required=True)
    parser.add_argument("--concurrency", type=positive_int, required=True)
    parser.add_argument("--max-tokens", type=positive_int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--workload", default=None)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    args = parser.parse_args(argv)
    if args.warmup_requests < 0:
        raise SystemExit("--warmup-requests must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        payload = run_benchmark(parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - CLI should print concise failure.
        print(f"bench_vllm_realtime failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: payload[key] for key in ("counts", "metrics")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
