"""Hand-calculated metrics tests for the offline comparison tool."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from script.compare_results import compare_artifacts, compare_files
from script.evaluation_common import atomic_write_json


DATASET_HASH = "dataset-hash"
TASKS_HASH = "tasks-hash"


def dataset(expected_values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"example-{index}",
            "state": {"utterance": f"synthetic utterance {index}"},
            "expected": expected,
        }
        for index, expected in enumerate(expected_values)
    ]


def tasks(*, include_boolean: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = {
        "intent": {
            "type": "choice",
            "instructions": "Choose a label.",
            "criteria": {"a": "Label A.", "b": "Label B."},
        }
    }
    if include_boolean:
        output["mentions_location"] = {
            "type": "noul",
            "instructions": "Is the condition true?",
            "criteria": {"true": "It is true.", "false": "It is false."},
        }
    return output


def record(example: dict[str, Any], question_name: str, question_type: str, actual: Any, *, confidence: float | None = None, error: Any = None) -> dict[str, Any]:
    return {
        "dataset_id": example["id"],
        "question_name": question_name,
        "question_type": question_type,
        "expected": example["expected"][question_name],
        "actual": actual,
        "native_confidence": confidence,
        "raw_answer": None,
        "latency_ms": 1.0,
        "error": error,
    }


def artifact(records: list[dict[str, Any]], *, dataset_hash: str = DATASET_HASH, tasks_hash: str = TASKS_HASH) -> dict[str, Any]:
    return {"metadata": {"dataset_sha256": dataset_hash, "tasks_sha256": tasks_hash}, "records": records}


def compare(data: list[dict[str, Any]], task_map: dict[str, Any], jev_records: list[dict[str, Any]], local_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return compare_artifacts(
        data,
        task_map,
        artifact(jev_records),
        artifact(local_records if local_records is not None else copy.deepcopy(jev_records)),
        dataset_hash=DATASET_HASH,
        tasks_hash=TASKS_HASH,
    )


def test_choice_metrics_hand_calculated_two_correct_one_wrong_one_failure() -> None:
    data = dataset([{"intent": "a"}, {"intent": "a"}, {"intent": "b"}, {"intent": "b"}])
    records = [
        record(data[0], "intent", "choice", "a", confidence=0.9),
        record(data[1], "intent", "choice", "a", confidence=0.8),
        record(data[2], "intent", "choice", "a", confidence=0.6),
        record(data[3], "intent", "choice", None, error={"code": "http_error"}),
    ]

    metrics = compare(data, tasks(), records)["jev"]["intent"]

    assert metrics["end_to_end_accuracy"] == 0.5
    assert metrics["valid_prediction_count"] == 3
    assert metrics["valid_only_accuracy"] == pytest.approx(2 / 3)
    assert metrics["coverage"] == 0.75
    assert metrics["correct"] == 2
    assert metrics["incorrect"] == 2
    assert metrics["failures"] == 1


def test_boolean_metrics_hand_calculated_tp_fp_fn_tn() -> None:
    data = dataset(
        [
            {"intent": "a", "mentions_location": True},
            {"intent": "a", "mentions_location": True},
            {"intent": "a", "mentions_location": False},
            {"intent": "a", "mentions_location": True},
            {"intent": "a", "mentions_location": False},
        ]
    )
    task_map = tasks(include_boolean=True)
    records = []
    for example, actual in zip(data, [True, True, True, False, False], strict=True):
        records.append(record(example, "intent", "choice", "a"))
        records.append(record(example, "mentions_location", "noul", actual, confidence=0.7))

    metrics = compare(data, task_map, records)["jev"]["mentions_location"]

    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "records, expected_precision, expected_recall, expected_f1",
    [
        ([False, False], None, 0.0, None),
        ([False, True], 0.0, None, None),
    ],
)
def test_boolean_zero_division_policy(records, expected_precision, expected_recall, expected_f1) -> None:
    expected = [True, False] if records == [False, False] else [False, False]
    data = dataset([{"intent": "a", "mentions_location": value} for value in expected])
    task_map = tasks(include_boolean=True)
    result_records = []
    for example, actual in zip(data, records, strict=True):
        result_records.extend(
            [
                record(example, "intent", "choice", "a"),
                record(example, "mentions_location", "noul", actual),
            ]
        )

    metrics = compare(data, task_map, result_records)["jev"]["mentions_location"]

    assert metrics["precision"] == expected_precision
    assert metrics["recall"] == expected_recall
    assert metrics["f1"] == expected_f1
    assert metrics["zero_division_policy"] == "null"


def test_duplicate_unknown_and_extra_records_fail_safely() -> None:
    data = dataset([{"intent": "a"}])
    valid = record(data[0], "intent", "choice", "a")
    with pytest.raises(ValueError, match="duplicate"):
        compare(data, tasks(), [valid, copy.deepcopy(valid)])

    unknown = copy.deepcopy(valid)
    unknown["dataset_id"] = "unknown"
    with pytest.raises(ValueError, match="unknown or extra"):
        compare(data, tasks(), [unknown])

    extra = copy.deepcopy(valid)
    extra["question_name"] = "other_question"
    with pytest.raises(ValueError, match="unknown or extra"):
        compare(data, tasks(), [extra])


def test_missing_record_is_explicit_failure_in_fixed_denominator() -> None:
    data = dataset([{"intent": "a"}, {"intent": "b"}])
    metrics = compare(data, tasks(), [record(data[0], "intent", "choice", "a")])

    intent = metrics["jev"]["intent"]
    assert intent["end_to_end_accuracy"] == 0.5
    assert intent["failures"] == 1
    assert metrics["comparison"]["jev_missing_records"] == [["example-1", "intent"]]
    assert metrics["errors"][0]["jev_error"] == {"code": "missing_record"}

    local_missing = compare(data, tasks(), [record(data[0], "intent", "choice", "a")], [])
    assert local_missing["comparison"]["local_missing_records"] == [
        ["example-0", "intent"],
        ["example-1", "intent"],
    ]


def test_hash_mismatch_and_expected_mismatch_fail_safely() -> None:
    data = dataset([{"intent": "a"}])
    valid = record(data[0], "intent", "choice", "a")
    with pytest.raises(ValueError, match="dataset_sha256"):
        compare_artifacts(data, tasks(), artifact([valid], dataset_hash="wrong"), artifact([valid]), dataset_hash=DATASET_HASH, tasks_hash=TASKS_HASH)

    bad_expected = copy.deepcopy(valid)
    bad_expected["expected"] = "b"
    with pytest.raises(ValueError, match="expected value"):
        compare(data, tasks(), [bad_expected])


def test_all_failures_all_correct_all_wrong_and_confidence_null_handling() -> None:
    data = dataset([{"intent": "a"}, {"intent": "b"}])
    failures = [
        record(data[0], "intent", "choice", None, error={"code": "failed"}),
        record(data[1], "intent", "choice", None, error={"code": "failed"}),
    ]
    failed_metrics = compare(data, tasks(), failures)["jev"]["intent"]
    assert failed_metrics["coverage"] == 0.0
    assert failed_metrics["valid_only_accuracy"] is None

    correct = [record(data[0], "intent", "choice", "a"), record(data[1], "intent", "choice", "b")]
    correct_metrics = compare(data, tasks(), correct)["jev"]["intent"]
    assert correct_metrics["end_to_end_accuracy"] == 1.0
    assert correct_metrics["confidence"]["average_confidence_correct"] is None

    wrong = [record(data[0], "intent", "choice", "b", confidence=0.2), record(data[1], "intent", "choice", "a", confidence=0.4)]
    wrong_metrics = compare(data, tasks(), wrong)["jev"]["intent"]
    assert wrong_metrics["end_to_end_accuracy"] == 0.0
    assert wrong_metrics["confidence"]["average_confidence_incorrect"] == pytest.approx(0.3)


def test_mixed_null_confidence_is_separated_by_service_and_question_type() -> None:
    data = dataset([{"intent": "a", "mentions_location": True}, {"intent": "b", "mentions_location": False}])
    task_map = tasks(include_boolean=True)
    jev_records = [
        record(data[0], "intent", "choice", "a", confidence=0.8),
        record(data[0], "mentions_location", "noul", True, confidence=None),
        record(data[1], "intent", "choice", "a", confidence=0.4),
        record(data[1], "mentions_location", "noul", False, confidence=None),
    ]
    local_records = copy.deepcopy(jev_records)
    local_records[1]["native_confidence"] = 0.9
    local_records[3]["native_confidence"] = 1.0

    metrics = compare(data, task_map, jev_records, local_records)

    assert metrics["jev"]["mentions_location"]["confidence"]["kind"] == "not_observed_for_jev_noul"
    assert metrics["jev"]["mentions_location"]["confidence"]["average_confidence_correct"] is None
    assert metrics["local"]["mentions_location"]["confidence"]["kind"] == "local_derived_noul_confidence"
    assert metrics["local"]["mentions_location"]["confidence"]["average_confidence_correct"] == pytest.approx(0.95)


def test_compare_files_writes_metrics_without_rewriting_input_artifacts(tmp_path) -> None:
    data = dataset([{"intent": "a"}])
    task_map = tasks()
    dataset_path = tmp_path / "dataset.json"
    tasks_path = tmp_path / "tasks.json"
    jev_path = tmp_path / "jev.json"
    local_path = tmp_path / "local.json"
    output_path = tmp_path / "metrics.json"
    atomic_write_json(dataset_path, data)
    atomic_write_json(tasks_path, task_map)
    before_jev = {"metadata": {}, "records": [record(data[0], "intent", "choice", "a")]}
    before_local = copy.deepcopy(before_jev)
    atomic_write_json(jev_path, before_jev)
    atomic_write_json(local_path, before_local)

    compare_files(dataset_path, jev_path, local_path, output_path, tasks_path=tasks_path)

    assert json.loads(jev_path.read_text(encoding="utf-8")) == before_jev
    assert json.loads(local_path.read_text(encoding="utf-8")) == before_local
    assert json.loads(output_path.read_text(encoding="utf-8"))["metadata"]["dataset_sha256"]
