"""Guarded Jev runner that records canonical, question-level evaluation results."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import json
import math
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings
from script.evaluation_common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TASKS_PATH,
    JsonObject,
    atomic_write_json,
    build_metadata,
    build_request_body,
    failure_records,
    load_evaluation_dataset,
    load_evaluation_tasks,
    result_record,
    safe_error,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "jev_results.json"
Sender = Callable[[str, JsonObject, str, float], tuple[int, Any]]


def _response_body(response: Any) -> Any:
    raw = response.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def send_jev_request(url: str, body: JsonObject, api_key: str, timeout: float) -> tuple[int, Any]:
    """Send one real Jev request. This is reached only through the live guard."""

    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 -- guarded explicit live mode
            return response.status, _response_body(response)
    except HTTPError as error:
        return error.code, _response_body(error)


def _valid_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _parse_answer(question: Mapping[str, Any], answer: Any) -> tuple[Any, float | None, JsonObject | None]:
    """Parse only the native fields observed for the specified Jev question type."""

    if not isinstance(answer, Mapping):
        return None, None, safe_error("invalid_response", "named answer is missing or malformed")

    question_type = question.get("type")
    if answer.get("type") != question_type:
        return None, None, safe_error("invalid_response", "named answer has an unexpected type")

    if question_type == "choice":
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        probabilities = answer.get("probabilities")
        if not isinstance(choice, str) or not _valid_number(confidence) or not isinstance(probabilities, Mapping):
            return None, None, safe_error("invalid_response", "Choice answer is missing native fields")
        return choice, float(confidence), None

    if question_type == "noul":
        probability = answer.get("noul")
        if not _valid_number(probability) or not 0 <= probability <= 1:
            return None, None, safe_error("invalid_response", "Noul answer is missing a valid native probability")
        return probability >= 0.5, None, None

    return None, None, safe_error("invalid_response", "question type is unsupported")


def parse_jev_response(
    example: Mapping[str, Any],
    tasks: Mapping[str, Any],
    response: Any,
    *,
    latency_ms: float,
) -> list[JsonObject]:
    """Convert one Jev HTTP success body into retained result records.

    A malformed answer affects only its named question. A malformed top-level
    response produces explicit failure records for all tasks.
    """

    if not isinstance(response, Mapping):
        return failure_records(
            example, tasks, error=safe_error("invalid_response", "response body is not an object"), latency_ms=latency_ms
        )
    if not isinstance(response.get("model"), str) or "usage" not in response or not isinstance(response.get("answers"), Mapping):
        return failure_records(
            example,
            tasks,
            error=safe_error("invalid_response", "response is missing required top-level fields"),
            latency_ms=latency_ms,
        )

    answers = response["answers"]
    records: list[JsonObject] = []
    for name, question in tasks.items():
        actual, native_confidence, error = _parse_answer(question, answers.get(name))
        records.append(
            result_record(
                example=example,
                question_name=name,
                question=question,
                actual=actual,
                native_confidence=native_confidence,
                raw_answer=answers.get(name) if isinstance(answers.get(name), Mapping) else None,
                latency_ms=latency_ms,
                error=error,
            )
        )
    return records


def _artifact(
    *,
    settings: Settings,
    dataset_path: Path,
    tasks_path: Path,
    run_status: str,
    requested_examples: int,
    completed_examples: int,
    records: list[JsonObject],
) -> JsonObject:
    return {
        "metadata": build_metadata(
            service="jev",
            model=settings.jev_model,
            dataset_path=dataset_path,
            tasks_path=tasks_path,
            run_status=run_status,
            requested_examples=requested_examples,
            completed_examples=completed_examples,
        ),
        "records": records,
    }


def run_evaluation(
    settings: Settings,
    *,
    live: bool = False,
    limit: int | None = None,
    output_path: Path | None = None,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    tasks_path: Path = DEFAULT_TASKS_PATH,
    sender: Sender = send_jev_request,
) -> JsonObject:
    """Run a guarded Jev evaluation or produce a no-network dry-run artifact."""

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if live and os.environ.get("RUN_LIVE_TESTS") != "1":
        raise RuntimeError("live Jev evaluation requires RUN_LIVE_TESTS=1")
    if live and not settings.jev_api_key:
        raise RuntimeError("JEV_API_KEY is required for live Jev evaluation")

    dataset = load_evaluation_dataset(dataset_path)
    tasks = load_evaluation_tasks(tasks_path)
    examples = dataset if limit is None else dataset[:limit]
    target_path = output_path or DEFAULT_OUTPUT_PATH

    if not live:
        planned_requests = []
        for example in examples:
            body = build_request_body(example, tasks, settings.jev_model)
            planned_requests.append(
                {"dataset_id": example["id"], "question_names": list(body["questions"])}
            )
        artifact = _artifact(
            settings=settings,
            dataset_path=dataset_path,
            tasks_path=tasks_path,
            run_status="dry-run",
            requested_examples=len(examples),
            completed_examples=0,
            records=[],
        )
        artifact["planned_requests"] = planned_requests
        atomic_write_json(target_path, artifact)
        return artifact

    records: list[JsonObject] = []
    for completed_examples, example in enumerate(examples, start=1):
        body = build_request_body(example, tasks, settings.jev_model)
        started = perf_counter()
        try:
            status, response = sender(
                f"{settings.jev_base_url}/v1/systemone",
                body,
                settings.jev_api_key or "",
                settings.model_timeout_seconds,
            )
            latency_ms = (perf_counter() - started) * 1000
            if status != 200:
                example_records = failure_records(
                    example,
                    tasks,
                    error=safe_error("http_error", f"Jev returned HTTP {status}"),
                    latency_ms=latency_ms,
                )
            else:
                example_records = parse_jev_response(example, tasks, response, latency_ms=latency_ms)
        except (OSError, TimeoutError, socket.timeout):
            latency_ms = (perf_counter() - started) * 1000
            example_records = failure_records(
                example,
                tasks,
                error=safe_error("transport_error", "Jev request failed"),
                latency_ms=latency_ms,
            )
        except Exception:
            latency_ms = (perf_counter() - started) * 1000
            example_records = failure_records(
                example,
                tasks,
                error=safe_error("runner_error", "Jev request could not be completed"),
                latency_ms=latency_ms,
            )
        records.extend(example_records)
        atomic_write_json(
            target_path,
            _artifact(
                settings=settings,
                dataset_path=dataset_path,
                tasks_path=tasks_path,
                run_status="partial",
                requested_examples=len(examples),
                completed_examples=completed_examples,
                records=records,
            ),
        )

    artifact = _artifact(
        settings=settings,
        dataset_path=dataset_path,
        tasks_path=tasks_path,
        run_status="complete",
        requested_examples=len(examples),
        completed_examples=len(examples),
        records=records,
    )
    atomic_write_json(target_path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="construct plans without sending HTTP requests")
    mode.add_argument("--live", action="store_true", help="send guarded HTTP requests to Jev")
    parser.add_argument("--limit", type=int, help="process at most this many examples")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    artifact = run_evaluation(
        get_settings(), live=args.live, limit=args.limit, output_path=args.output
    )
    print(
        f"Wrote {len(artifact['records'])} records for "
        f"{artifact['metadata']['run_status']} run to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
