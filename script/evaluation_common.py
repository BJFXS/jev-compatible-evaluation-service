"""Shared, provider-neutral utilities for evaluation runners."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "evaluation_dataset.json"
DEFAULT_TASKS_PATH = PROJECT_ROOT / "data" / "evaluation_tasks.json"

JsonObject = dict[str, Any]
SENSITIVE_KEYS = frozenset({"authorization", "api_key", "apikey", "access_token", "token", "secret"})


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON from {path}") from error


def _require_non_blank_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string")
    return value


def _validate_state(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (str, dict, list)):
        raise ValueError("example state must be a string, object, or list")


def load_evaluation_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[JsonObject]:
    """Load and validate the runner-facing shape of an evaluation dataset."""

    data = _load_json(path)
    if not isinstance(data, list) or not data:
        raise ValueError("evaluation dataset must be a non-empty JSON list")

    examples: list[JsonObject] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(data):
        if not isinstance(value, dict):
            raise ValueError(f"dataset example {index} must be an object")
        dataset_id = _require_non_blank_string(value.get("id"), f"dataset example {index} id")
        if dataset_id in seen_ids:
            raise ValueError(f"duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)
        _validate_state(value.get("state"))
        if not isinstance(value.get("expected"), dict):
            raise ValueError(f"dataset example {dataset_id} expected must be an object")
        examples.append(value)
    return examples


def load_evaluation_tasks(path: Path = DEFAULT_TASKS_PATH) -> JsonObject:
    """Load shared Choice/Noul tasks without embedding their labels in code."""

    tasks = _load_json(path)
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("evaluation tasks must be a non-empty JSON object")

    for name, task in tasks.items():
        _require_non_blank_string(name, "task name")
        if not isinstance(task, dict):
            raise ValueError(f"task {name} must be an object")
        question_type = task.get("type")
        if question_type not in {"choice", "noul"}:
            raise ValueError(f"task {name} has unsupported type")
        _require_non_blank_string(task.get("instructions"), f"task {name} instructions")
        criteria = task.get("criteria")
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError(f"task {name} criteria must be a non-empty object")
        for key, description in criteria.items():
            _require_non_blank_string(key, f"task {name} criteria key")
            _require_non_blank_string(description, f"task {name} criteria description")
    return tasks


def build_request_body(example: Mapping[str, Any], tasks: Mapping[str, Any], model: str) -> JsonObject:
    """Build the only body that may leave an evaluation runner.

    Deliberately selecting these three fields prevents dataset metadata and
    ground truth from crossing the provider boundary.
    """

    _require_non_blank_string(model, "model")
    _validate_state(example.get("state"))
    if not isinstance(tasks, Mapping) or not tasks:
        raise ValueError("tasks must be a non-empty mapping")
    return {"state": deepcopy(example["state"]), "model": model, "questions": deepcopy(dict(tasks))}


def sha256_file(path: Path) -> str:
    """Return a stable SHA256 for frozen input files."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_for(example: Mapping[str, Any], question_name: str) -> Any:
    """Extract ground truth only for local result/scoring artifacts."""

    expected = example.get("expected")
    if not isinstance(expected, Mapping) or question_name not in expected:
        raise ValueError(f"example expected answer is missing {question_name}")
    return expected[question_name]


def safe_error(code: str, message: str) -> JsonObject:
    """Construct a serializable error without provider or request details."""

    return {"code": code, "message": message}


def redact_sensitive(value: Any) -> Any:
    """Defensively redact credentials from stored artifacts."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in SENSITIVE_KEYS else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def result_record(
    *,
    example: Mapping[str, Any],
    question_name: str,
    question: Mapping[str, Any],
    actual: Any,
    native_confidence: float | None,
    raw_answer: Mapping[str, Any] | None,
    latency_ms: float | None,
    error: Mapping[str, Any] | None,
) -> JsonObject:
    """Create one canonical question-level evaluation record."""

    return {
        "dataset_id": _require_non_blank_string(example.get("id"), "dataset id"),
        "question_name": question_name,
        "question_type": question.get("type"),
        "expected": expected_for(example, question_name),
        "actual": actual,
        "native_confidence": native_confidence,
        "raw_answer": redact_sensitive(dict(raw_answer)) if raw_answer is not None else None,
        "latency_ms": latency_ms,
        "error": redact_sensitive(dict(error)) if error is not None else None,
    }


def failure_records(
    example: Mapping[str, Any],
    tasks: Mapping[str, Any],
    *,
    error: Mapping[str, Any],
    latency_ms: float | None,
) -> list[JsonObject]:
    """Retain one explicit failure record for every named task."""

    return [
        result_record(
            example=example,
            question_name=name,
            question=question,
            actual=None,
            native_confidence=None,
            raw_answer=None,
            latency_ms=latency_ms,
            error=error,
        )
        for name, question in tasks.items()
    ]


def build_metadata(
    *,
    service: str,
    model: str,
    dataset_path: Path,
    tasks_path: Path,
    run_status: str,
    requested_examples: int,
    completed_examples: int,
) -> JsonObject:
    """Create common artifact metadata for a runner execution."""

    return {
        "service": service,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "dataset_sha256": sha256_file(dataset_path),
        "tasks_sha256": sha256_file(tasks_path),
        "run_status": run_status,
        "requested_examples": requested_examples,
        "completed_examples": completed_examples,
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a JSON artifact, leaving no partially appended JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(redact_sensitive(value), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def score_records(records: list[Mapping[str, Any]]) -> JsonObject:
    """Return simple shared correctness counts without altering predictions."""

    total = len(records)
    valid = [record for record in records if record.get("actual") is not None and record.get("error") is None]
    correct = sum(record.get("actual") == record.get("expected") for record in valid)
    return {"total": total, "valid": len(valid), "correct": correct}
