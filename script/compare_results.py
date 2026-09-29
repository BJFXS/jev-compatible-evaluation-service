"""Pure offline alignment, metrics, and error analysis for Jev/local artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from script.evaluation_common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TASKS_PATH,
    JsonObject,
    atomic_write_json,
    load_evaluation_dataset,
    load_evaluation_tasks,
    sha256_file,
)


DEFAULT_JEV_PATH = PROJECT_ROOT / "results" / "jev_results.json"
DEFAULT_LOCAL_PATH = PROJECT_ROOT / "results" / "local_results.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "results" / "metrics.json"
RecordKey = tuple[str, str]


def _load_artifact(path: Path) -> JsonObject:
    try:
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load result artifact: {path}") from error
    if not isinstance(artifact, dict) or not isinstance(artifact.get("metadata"), dict) or not isinstance(artifact.get("records"), list):
        raise ValueError(f"result artifact has invalid shape: {path}")
    return artifact


def _expected_keys(dataset: list[JsonObject], tasks: Mapping[str, Any]) -> set[RecordKey]:
    return {(example["id"], name) for example in dataset for name in tasks}


def _expected_values(dataset: list[JsonObject]) -> dict[RecordKey, Any]:
    return {
        (example["id"], question_name): expected
        for example in dataset
        for question_name, expected in example["expected"].items()
    }


def _validate_hashes(
    artifact: Mapping[str, Any], *, dataset_hash: str, tasks_hash: str, service: str
) -> None:
    metadata = artifact["metadata"]
    for field, expected_hash in (("dataset_sha256", dataset_hash), ("tasks_sha256", tasks_hash)):
        value = metadata.get(field)
        if value is not None and value != expected_hash:
            raise ValueError(f"{service} artifact {field} does not match frozen input")


def _index_records(
    artifact: Mapping[str, Any],
    *,
    service: str,
    expected_keys: set[RecordKey],
    expected_values: Mapping[RecordKey, Any],
) -> dict[RecordKey, JsonObject]:
    index: dict[RecordKey, JsonObject] = {}
    for record in artifact["records"]:
        if not isinstance(record, dict):
            raise ValueError(f"{service} artifact contains a non-object record")
        dataset_id = record.get("dataset_id")
        question_name = record.get("question_name")
        if not isinstance(dataset_id, str) or not isinstance(question_name, str):
            raise ValueError(f"{service} artifact record has invalid alignment key")
        key = (dataset_id, question_name)
        if key in index:
            raise ValueError(f"{service} artifact has duplicate record for {key}")
        if key not in expected_keys:
            raise ValueError(f"{service} artifact has unknown or extra record for {key}")
        if record.get("expected") != expected_values[key]:
            raise ValueError(f"{service} artifact expected value disagrees with frozen dataset for {key}")
        index[key] = record
    return index


def _is_valid_prediction(record: Mapping[str, Any] | None, question_type: str) -> bool:
    if record is None or record.get("error") is not None or record.get("actual") is None:
        return False
    actual = record.get("actual")
    if question_type == "noul":
        return isinstance(actual, bool)
    return isinstance(actual, str)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _confidence_summary(records: list[Mapping[str, Any] | None], expected: list[Any], *, service: str, question_type: str) -> JsonObject:
    correct_values: list[float] = []
    incorrect_values: list[float] = []
    for record, target in zip(records, expected, strict=True):
        if not _is_valid_prediction(record, question_type):
            continue
        confidence = record.get("native_confidence") if record is not None else None
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            continue
        values = correct_values if record["actual"] == target else incorrect_values
        values.append(float(confidence))

    if service == "jev" and question_type == "choice":
        kind = "jev_native_choice_confidence"
    elif service == "jev":
        kind = "not_observed_for_jev_noul"
    elif question_type == "choice":
        kind = "local_heuristic_choice_confidence"
    else:
        kind = "local_derived_noul_confidence"
    return {
        "kind": kind,
        "average_confidence_correct": _mean(correct_values),
        "average_confidence_incorrect": _mean(incorrect_values),
        "correct_confidence_count": len(correct_values),
        "incorrect_confidence_count": len(incorrect_values),
    }


def _task_metrics(
    records: list[Mapping[str, Any] | None], expected: list[Any], *, service: str, question_type: str
) -> JsonObject:
    denominator = len(expected)
    valid_pairs = [
        (record, target)
        for record, target in zip(records, expected, strict=True)
        if _is_valid_prediction(record, question_type)
    ]
    correct = sum(record["actual"] == target for record, target in valid_pairs)
    valid_count = len(valid_pairs)
    failures = denominator - valid_count
    metrics: JsonObject = {
        "end_to_end_accuracy": correct / denominator if denominator else None,
        "valid_only_accuracy": correct / valid_count if valid_count else None,
        "coverage": valid_count / denominator if denominator else None,
        "correct": correct,
        "incorrect": denominator - correct,
        "failures": failures,
        "correct_count": correct,
        "incorrect_count": denominator - correct,
        "failure_count": failures,
        "valid_prediction_count": valid_count,
        "denominator": denominator,
        "confidence": _confidence_summary(records, expected, service=service, question_type=question_type),
    }
    if question_type == "noul":
        true_positive = sum(record["actual"] is True and target is True for record, target in valid_pairs)
        false_positive = sum(record["actual"] is True and target is False for record, target in valid_pairs)
        false_negative = sum(record["actual"] is False and target is True for record, target in valid_pairs)
        true_negative = sum(record["actual"] is False and target is False for record, target in valid_pairs)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "zero_division_policy": "null",
            }
        )
    return metrics


def _utterance(example: Mapping[str, Any]) -> str | None:
    state = example.get("state")
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) and isinstance(state.get("utterance"), str):
        return state["utterance"]
    return None


def _record_status(record: Mapping[str, Any] | None, expected: Any, question_type: str) -> str:
    if not _is_valid_prediction(record, question_type):
        return "failed"
    return "correct" if record["actual"] == expected else "wrong"


def compare_artifacts(
    dataset: list[JsonObject], tasks: Mapping[str, Any], jev_artifact: JsonObject, local_artifact: JsonObject, *, dataset_hash: str, tasks_hash: str
) -> JsonObject:
    """Validate two canonical artifacts and produce metrics without changing either."""

    expected_keys = _expected_keys(dataset, tasks)
    expected_values = _expected_values(dataset)
    _validate_hashes(jev_artifact, dataset_hash=dataset_hash, tasks_hash=tasks_hash, service="jev")
    _validate_hashes(local_artifact, dataset_hash=dataset_hash, tasks_hash=tasks_hash, service="local")
    jev_index = _index_records(jev_artifact, service="jev", expected_keys=expected_keys, expected_values=expected_values)
    local_index = _index_records(local_artifact, service="local", expected_keys=expected_keys, expected_values=expected_values)

    task_metrics: dict[str, Any] = {"jev": {}, "local": {}}
    errors: list[JsonObject] = []
    comparison_counts = {
        "both_correct": 0,
        "jev_correct_local_wrong": 0,
        "local_correct_jev_wrong": 0,
        "both_wrong": 0,
        "one_or_both_failed": 0,
    }
    examples_by_id = {example["id"]: example for example in dataset}

    for question_name, task in tasks.items():
        question_type = task["type"]
        keys = [(example["id"], question_name) for example in dataset]
        expected = [expected_values[key] for key in keys]
        jev_records = [jev_index.get(key) for key in keys]
        local_records = [local_index.get(key) for key in keys]
        task_metrics["jev"][question_name] = _task_metrics(
            jev_records, expected, service="jev", question_type=question_type
        )
        task_metrics["local"][question_name] = _task_metrics(
            local_records, expected, service="local", question_type=question_type
        )

        for key, target, jev_record, local_record in zip(keys, expected, jev_records, local_records, strict=True):
            jev_status = _record_status(jev_record, target, question_type)
            local_status = _record_status(local_record, target, question_type)
            if "failed" in {jev_status, local_status}:
                comparison_counts["one_or_both_failed"] += 1
            elif jev_status == local_status == "correct":
                comparison_counts["both_correct"] += 1
            elif jev_status == "correct":
                comparison_counts["jev_correct_local_wrong"] += 1
            elif local_status == "correct":
                comparison_counts["local_correct_jev_wrong"] += 1
            else:
                comparison_counts["both_wrong"] += 1

            if jev_status != "correct" or local_status != "correct":
                example = examples_by_id[key[0]]
                errors.append(
                    {
                        "dataset_id": key[0],
                        "utterance": _utterance(example),
                        "question_name": question_name,
                        "expected": target,
                        "jev_actual": jev_record.get("actual") if jev_record else None,
                        "local_actual": local_record.get("actual") if local_record else None,
                        "jev_confidence": jev_record.get("native_confidence") if jev_record else None,
                        "local_confidence": local_record.get("native_confidence") if local_record else None,
                        "jev_error": jev_record.get("error") if jev_record else {"code": "missing_record"},
                        "local_error": local_record.get("error") if local_record else {"code": "missing_record"},
                    }
                )

    total_correct = sum(task_metrics["jev"][name]["correct"] for name in tasks)
    local_total_correct = sum(task_metrics["local"][name]["correct"] for name in tasks)
    total_records = len(expected_keys)
    return {
        "metadata": {
            "dataset_sha256": dataset_hash,
            "tasks_sha256": tasks_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alignment_key": ["dataset_id", "question_name"],
        },
        "jev": {
            **task_metrics["jev"],
            "overall": {"correct": total_correct, "total": total_records, "accuracy": total_correct / total_records},
        },
        "local": {
            **task_metrics["local"],
            "overall": {"correct": local_total_correct, "total": total_records, "accuracy": local_total_correct / total_records},
        },
        "comparison": {
            **comparison_counts,
            "jev_missing_records": [list(key) for key in sorted(expected_keys - set(jev_index))],
            "local_missing_records": [list(key) for key in sorted(expected_keys - set(local_index))],
        },
        "errors": errors,
    }


def compare_files(
    dataset_path: Path,
    jev_path: Path,
    local_path: Path,
    output_path: Path,
    *,
    tasks_path: Path = DEFAULT_TASKS_PATH,
) -> JsonObject:
    """Read immutable inputs, calculate metrics, and atomically write only output."""

    dataset = load_evaluation_dataset(dataset_path)
    tasks = load_evaluation_tasks(tasks_path)
    artifact = compare_artifacts(
        dataset,
        tasks,
        _load_artifact(jev_path),
        _load_artifact(local_path),
        dataset_hash=sha256_file(dataset_path),
        tasks_hash=sha256_file(tasks_path),
    )
    atomic_write_json(output_path, artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--jev", type=Path, default=DEFAULT_JEV_PATH)
    parser.add_argument("--local", type=Path, default=DEFAULT_LOCAL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    metrics = compare_files(args.dataset, args.jev, args.local, args.output, tasks_path=args.tasks)
    print(f"Wrote metrics for {metrics['metadata']['dataset_sha256']} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
