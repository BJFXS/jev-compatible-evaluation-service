"""Tests for provider-neutral evaluation utilities."""

from __future__ import annotations

import json

from script.evaluation_common import (
    DEFAULT_DATASET_PATH,
    DEFAULT_TASKS_PATH,
    atomic_write_json,
    build_request_body,
    expected_for,
    load_evaluation_dataset,
    load_evaluation_tasks,
    sha256_file,
)


def test_loads_valid_project_dataset_and_tasks() -> None:
    dataset = load_evaluation_dataset()
    tasks = load_evaluation_tasks()

    assert len(dataset) == 28
    assert set(tasks) == {"intent", "mentions_location"}


def test_input_hashes_are_stable() -> None:
    assert sha256_file(DEFAULT_DATASET_PATH) == sha256_file(DEFAULT_DATASET_PATH)
    assert sha256_file(DEFAULT_TASKS_PATH) == sha256_file(DEFAULT_TASKS_PATH)


def test_request_body_contains_only_state_model_and_questions() -> None:
    example = load_evaluation_dataset()[0]
    tasks = load_evaluation_tasks()

    body = build_request_body(example, tasks, "jev-test-model")

    assert set(body) == {"state", "model", "questions"}
    assert body["state"] == example["state"]
    assert body["questions"] == tasks
    serialized = json.dumps(body)
    for forbidden in ("expected", "source_labels", "source", '"id"', "test_index"):
        assert forbidden not in serialized


def test_tasks_are_reused_but_not_shared_mutably_between_requests() -> None:
    dataset = load_evaluation_dataset()
    tasks = load_evaluation_tasks()

    first = build_request_body(dataset[0], tasks, "jev-test-model")
    second = build_request_body(dataset[1], tasks, "jev-test-model")

    assert first["questions"] == second["questions"] == tasks
    assert first["questions"] is not second["questions"]


def test_generic_non_snips_request_and_expected_extraction() -> None:
    example = {
        "id": "generic-1",
        "state": ["opaque", {"status": "pending"}],
        "expected": {"route": "defer", "is_pending": True},
    }
    tasks = {
        "route": {
            "type": "choice",
            "instructions": "Choose a workflow action.",
            "criteria": {"approve": "Proceed.", "defer": "Wait."},
        },
        "is_pending": {
            "type": "noul",
            "instructions": "Is the status pending?",
            "criteria": {"true": "It is pending.", "false": "It is not pending."},
        },
    }

    body = build_request_body(example, tasks, "generic-model")

    assert body["questions"]["route"]["criteria"] == {"approve": "Proceed.", "defer": "Wait."}
    assert expected_for(example, "route") == "defer"
    assert expected_for(example, "is_pending") is True


def test_atomic_write_produces_valid_json(tmp_path) -> None:
    output = tmp_path / "nested" / "result.json"
    artifact = {"metadata": {"run_status": "dry-run"}, "records": []}

    atomic_write_json(output, artifact)

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert not list(output.parent.glob("tmp*"))
