"""Offline local-runner tests with only the HTTP sender faked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from script.evaluation_common import atomic_write_json
from script.run_local import LOCAL_MODEL_ALIAS, run_evaluation


def synthetic_dataset() -> list[dict[str, Any]]:
    return [
        {
            "id": "generic-local-1",
            "state": {"message": "The review is approved at the harbor office."},
            "source": {"dataset": "synthetic", "test_index": 9},
            "source_labels": {"route": "approve"},
            "expected": {"route": "approve", "has_place": True},
        }
    ]


def synthetic_tasks() -> dict[str, Any]:
    return {
        "route": {
            "type": "choice",
            "instructions": "Choose the review status.",
            "criteria": {"approve": "It is approved.", "reject": "It is rejected."},
        },
        "has_place": {
            "type": "noul",
            "instructions": "Does the message mention a place?",
            "criteria": {"true": "It mentions a place.", "false": "It does not mention a place."},
        },
    }


def write_inputs(tmp_path: Path, dataset: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    dataset_path = tmp_path / "dataset.json"
    tasks_path = tmp_path / "tasks.json"
    atomic_write_json(dataset_path, dataset or synthetic_dataset())
    atomic_write_json(tasks_path, synthetic_tasks())
    return dataset_path, tasks_path


def settings() -> Settings:
    return Settings(
        local_base_url="http://127.0.0.1:8765/",
        model_timeout_seconds=1.0,
        noul_boolean_threshold=0.5,
    )


def local_success() -> dict[str, Any]:
    return {
        "model": "actual-local-model",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "approve",
                "probabilities": {"approve": 0.8, "reject": 0.2},
                "confidence": 0.8,
            },
            "has_place": {"type": "noul", "noul": 0.9, "answer": True, "confidence": 0.9},
        },
    }


def test_success_uses_one_gpt_local_request_and_two_records(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    calls: list[tuple[str, dict[str, Any], float]] = []

    def sender(url: str, body: dict[str, Any], timeout: float) -> tuple[int, Any]:
        calls.append((url, body, timeout))
        return 200, local_success()

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert len(calls) == 1
    assert calls[0][0] == "http://127.0.0.1:8765/v1/systemone"
    assert calls[0][1]["model"] == LOCAL_MODEL_ALIAS
    assert set(calls[0][1]) == {"state", "model", "questions"}
    assert artifact["metadata"]["model"] == "actual-local-model"
    assert len(artifact["records"]) == 2
    choice, noul = artifact["records"]
    assert choice["actual"] == "approve"
    assert choice["native_confidence"] == 0.8
    assert noul["actual"] is True
    assert noul["native_confidence"] == 0.9


def test_noul_answer_must_match_probability_threshold(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    response = local_success()
    response["answers"]["has_place"] = {"type": "noul", "noul": 0.9, "answer": False, "confidence": 0.9}
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda *_: (200, response),
    )

    assert artifact["records"][0]["error"] is None
    assert artifact["records"][1]["actual"] is None
    assert artifact["records"][1]["error"]["code"] == "invalid_response"


@pytest.mark.parametrize(
    "response",
    [
        {"model": "actual-local-model", "answers": {"has_place": local_success()["answers"]["has_place"]}},
        {
            "model": "actual-local-model",
            "answers": {
                "route": {"type": "choice", "choice": "other", "probabilities": {}, "confidence": 0.5},
                "has_place": local_success()["answers"]["has_place"],
            },
        },
        {
            "model": "actual-local-model",
            "answers": {
                "route": local_success()["answers"]["route"],
                "has_place": {"type": "noul", "noul": 0.4, "answer": False},
            },
        },
    ],
)
def test_missing_or_malformed_named_answer_is_retained_as_failure(tmp_path, monkeypatch, response: dict[str, Any]) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda *_: (200, response),
    )

    assert any(record["error"] is not None for record in artifact["records"])
    assert len(artifact["records"]) == 2


@pytest.mark.parametrize("response", [None, "not-json"])
def test_invalid_json_body_retains_two_failure_records(tmp_path, monkeypatch, response: Any) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    monkeypatch.setenv("RUN_LIVE_TESTS", "1")

    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda *_: (200, response),
    )

    assert [record["actual"] for record in artifact["records"]] == [None, None]


def test_http_error_retains_two_failure_records_without_retry(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    calls = 0

    def sender(*_: Any) -> tuple[int, Any]:
        nonlocal calls
        calls += 1
        return 503, {"error": {"code": "model_unavailable"}}

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert calls == 1
    assert {record["error"]["code"] for record in artifact["records"]} == {"http_error"}


def test_transport_error_retains_two_failure_records_without_retry(tmp_path, monkeypatch) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    calls = 0

    def sender(*_: Any) -> tuple[int, Any]:
        nonlocal calls
        calls += 1
        raise TimeoutError("not exposed")

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert calls == 1
    assert {record["error"]["code"] for record in artifact["records"]} == {"transport_error"}


def test_dry_run_has_zero_network_calls(tmp_path) -> None:
    dataset_path, tasks_path = write_inputs(tmp_path)
    calls = 0

    def sender(*_: Any) -> tuple[int, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError("dry-run must not call sender")

    artifact = run_evaluation(
        settings(), live=False, output_path=tmp_path / "result.json", dataset_path=dataset_path, tasks_path=tasks_path, sender=sender
    )

    assert calls == 0
    assert artifact["metadata"]["run_status"] == "dry-run"
    assert artifact["records"] == []


def test_limit_live_guard_and_ground_truth_secret_safety(tmp_path, monkeypatch) -> None:
    dataset = synthetic_dataset() * 2
    dataset[1] = {**dataset[1], "id": "generic-local-2"}
    dataset_path, tasks_path = write_inputs(tmp_path, dataset=dataset)
    sent: list[dict[str, Any]] = []
    response = local_success()
    response["answers"]["route"]["authorization"] = "Bearer should-not-persist"
    output = tmp_path / "result.json"
    monkeypatch.delenv("RUN_LIVE_TESTS", raising=False)

    with pytest.raises(RuntimeError, match="RUN_LIVE_TESTS"):
        run_evaluation(
            settings(), live=True, output_path=output, dataset_path=dataset_path, tasks_path=tasks_path,
            sender=lambda *_: (_ for _ in ()).throw(AssertionError("must not send")),
        )

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, limit=1, output_path=output, dataset_path=dataset_path, tasks_path=tasks_path,
        sender=lambda _url, body, _timeout: (sent.append(body) or (200, response)),
    )

    assert artifact["metadata"]["requested_examples"] == 1
    assert len(sent) == 1
    serialized_request = json.dumps(sent[0])
    for forbidden in ("expected", "source_labels", "source", '"id"', "test_index"):
        assert forbidden not in serialized_request
    artifact_text = output.read_text(encoding="utf-8")
    assert "should-not-persist" not in artifact_text
    assert "Bearer" not in artifact_text


def test_full_project_dataset_uses_28_requests_and_56_records(tmp_path, monkeypatch) -> None:
    from script.evaluation_common import DEFAULT_DATASET_PATH, DEFAULT_TASKS_PATH

    calls: list[dict[str, Any]] = []

    def sender(_url: str, body: dict[str, Any], _timeout: float) -> tuple[int, Any]:
        calls.append(body)
        questions = body["questions"]
        return 200, {
            "model": "actual-local-model",
            "answers": {
                "intent": {
                    "type": "choice",
                    "choice": next(iter(questions["intent"]["criteria"])),
                    "probabilities": {key: 0.0 for key in questions["intent"]["criteria"]},
                    "confidence": 1.0,
                },
                "mentions_location": {"type": "noul", "noul": 0.0, "answer": False, "confidence": 1.0},
            },
        }

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        settings(), live=True, output_path=tmp_path / "result.json", dataset_path=DEFAULT_DATASET_PATH,
        tasks_path=DEFAULT_TASKS_PATH, sender=sender,
    )

    assert len(calls) == 28
    assert len(artifact["records"]) == 56
    assert all(set(body) == {"state", "model", "questions"} for body in calls)
