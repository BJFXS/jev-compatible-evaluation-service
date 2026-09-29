"""Offline Jev runner tests with an injected HTTP boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from script.evaluation_common import atomic_write_json
from script.run_jev import run_evaluation


def synthetic_dataset() -> list[dict[str, Any]]:
    return [
        {
            "id": "generic-1",
            "state": {"message": "The request is approved at the north office."},
            "source": {"dataset": "synthetic", "test_index": 7},
            "source_labels": {"route": "approve"},
            "expected": {"route": "approve", "has_place": True},
        }
    ]


def synthetic_tasks() -> dict[str, Any]:
    return {
        "route": {
            "type": "choice",
            "instructions": "Choose the approval state.",
            "criteria": {"approve": "It is approved.", "reject": "It is rejected."},
        },
        "has_place": {
            "type": "noul",
            "instructions": "Does the message mention a place?",
            "criteria": {"true": "It mentions a place.", "false": "It does not mention a place."},
        },
    }


def write_inputs(tmp_path: Path, dataset: list[dict[str, Any]] | None = None, tasks: dict[str, Any] | None = None) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.json"
    tasks_path = tmp_path / "tasks.json"
    atomic_write_json(dataset_path, dataset or synthetic_dataset())
    atomic_write_json(tasks_path, tasks or synthetic_tasks())
    return dataset_path, tasks_path


def settings(*, api_key: str | None = "test-key") -> Settings:
    return Settings(
        jev_api_key=api_key,
        jev_model="jev-test-model",
        jev_base_url="https://jev.example.invalid",
        model_timeout_seconds=1.0,
    )


def jev_success() -> dict[str, Any]:
    return {
        "model": "jev-actual-model",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "approve",
                "confidence": 0.8,
                "probabilities": {"approve": 0.8, "reject": 0.2},
            },
            "has_place": {"type": "noul", "noul": 0.9},
        },
        "usage": {"input_tokens": 1},
    }


def test_success_sends_one_request_and_creates_two_native_records(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    output = tmp_path / "result.json"
    calls: list[tuple[str, dict[str, Any], str, float]] = []

    def sender(url: str, body: dict[str, Any], api_key: str, timeout: float) -> tuple[int, Any]:
        calls.append((url, body, api_key, timeout))
        return 200, jev_success()

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=output, dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert len(calls) == 1
    assert set(calls[0][1]) == {"state", "model", "questions"}
    assert len(artifact["records"]) == 2
    choice, noul = artifact["records"]
    assert choice["actual"] == "approve"
    assert choice["native_confidence"] == 0.8
    assert noul["actual"] is True
    assert noul["native_confidence"] is None
    assert noul["raw_answer"] == {"type": "noul", "noul": 0.9}


def test_http_error_retains_two_failure_records(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(),
        live=True,
        output_path=tmp_path / "result.json",
        dataset_path=dataset_path,
        tasks_path=tasks_path,
        sender=lambda *_: (503, {"detail": "ignored"}),
    )

    assert [record["actual"] for record in artifact["records"]] == [None, None]
    assert {record["error"]["code"] for record in artifact["records"]} == {"http_error"}


def test_missing_named_answer_retains_relevant_failure(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    response = jev_success()
    del response["answers"]["has_place"]
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda *_: (200, response),
    )

    assert artifact["records"][0]["actual"] == "approve"
    assert artifact["records"][1]["actual"] is None
    assert artifact["records"][1]["error"]["code"] == "invalid_response"


def test_malformed_success_response_retains_failures(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda *_: (200, {"model": "jev", "answers": {}}),
    )

    assert all(record["actual"] is None for record in artifact["records"])
    assert all(record["error"]["code"] == "invalid_response" for record in artifact["records"])


def test_dry_run_makes_zero_network_calls(tmp_path) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    calls = 0

    def sender(*_: Any) -> tuple[int, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("dry-run must not send HTTP")

    artifact = run_evaluation(
        settings(), live=False, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert calls == 0
    assert artifact["metadata"]["run_status"] == "dry-run"
    assert artifact["records"] == []
    assert artifact["planned_requests"] == [{"dataset_id": "generic-1", "question_names": ["route", "has_place"]}]


def test_limit_processes_only_requested_examples(tmp_path, monkeypatch) -> None:
    dataset = synthetic_dataset() * 2
    dataset[1] = {**dataset[1], "id": "generic-2"}
    dataset_path, tasks_path = write_inputs(tmp_path, dataset=dataset)
    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, limit=1, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda _url, body, _key, _timeout: (calls.append(body) or (200, jev_success())),
    )

    assert len(calls) == 1
    assert artifact["metadata"]["requested_examples"] == 1
    assert len(artifact["records"]) == 2


def test_full_project_dataset_plans_28_requests_and_56_records(tmp_path, monkeypatch) -> None:
    from script.evaluation_common import DEFAULT_DATASET_PATH, DEFAULT_TASKS_PATH

    calls: list[dict[str, Any]] = []

    def sender(_url: str, body: dict[str, Any], _key: str, _timeout: float) -> tuple[int, Any]:
        calls.append(body)
        questions = body["questions"]
        return 200, {
            "model": "jev-actual-model",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": next(iter(questions["intent"]["criteria"])),
                    "confidence": 1.0,
                    "probabilities": {key: 0.0 for key in questions["intent"]["criteria"]},
                },
                "mentions_location": {"type": "noul", "noul": 0.0},
            },
            "usage": {},
        }

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=DEFAULT_DATASET_PATH,
        tasks_path=DEFAULT_TASKS_PATH, sender=sender,
    )

    assert len(calls) == 28
    assert len(artifact["records"]) == 56
    assert all(set(body) == {"state", "model", "questions"} for body in calls)


def test_live_guard_blocks_missing_flag_or_key(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    sender = lambda *_: (_ for _ in ()).throw(AssertionError("must not call sender"))
    monkeypatch.delenv("RUN_LIVE_TESTS", raising=False)

    with pytest.raises(RuntimeError, match="RUN_LIVE_TESTS"):
        run_evaluation(settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender)

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        run_evaluation(settings(api_key=None), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender)


def test_output_redacts_authorization_and_request_never_contains_ground_truth(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    sent: list[dict[str, Any]] = []
    response = jev_success()
    response["answers"]["route"]["authorization"] = "Bearer test-secret"
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    output = tmp_path / "result.json"

    artifact = run_evaluation(
        settings(api_key="test-secret"), live=True, output_path=output, dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda _url, body, _key, _timeout: (sent.append(body) or (200, response)),
    )

    assert artifact["records"][0]["raw_answer"]["authorization"] == "[REDACTED]"
    outbound = json.dumps(sent[0])
    for forbidden in ("expected", "source_labels", "source", '"id"', "test_index"):
        assert forbidden not in outbound
    assert "test-secret" not in output.read_text(encoding="utf-8")
    assert "Bearer" not in output.read_text(encoding="utf-8")
