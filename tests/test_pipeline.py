"""End-to-end local runner pipeline with the external provider boundary faked."""

from __future__ import annotations

from copy import deepcopy

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas import ChoiceModelOutput, NoulModelOutput
from script.evaluation_common import atomic_write_json
from script.run_local import run_evaluation
from tests.fakes import FakeEvaluator


def test_local_runner_to_asgi_service_pipeline(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "dataset.json"
    tasks_path = tmp_path / "tasks.json"
    output_path = tmp_path / "result.json"
    example = {
        "id": "pipeline-1",
        "state": {"document": "The release has been approved at Cedar Point."},
        "source": {"dataset": "synthetic", "test_index": 1},
        "source_labels": {"route": "approve"},
        "expected": {"route": "approve", "has_place": True},
    }
    tasks = {
        "route": {
            "type": "choice",
            "instructions": "Choose the release disposition.",
            "criteria": {"approve": "The release is approved.", "reject": "The release is rejected."},
        },
        "has_place": {
            "type": "noul",
            "instructions": "Does the document name a place?",
            "criteria": {"true": "It names a place.", "false": "It does not name a place."},
        },
    }
    atomic_write_json(dataset_path, [example])
    atomic_write_json(tasks_path, tasks)

    evaluator = FakeEvaluator(
        model_name="pipeline-model",
        choice_outcomes=[ChoiceModelOutput(choice="approve", probabilities={"approve": 0.8, "reject": 0.2})],
        noul_outcomes=[NoulModelOutput(probability_yes=0.9)],
    )
    client = TestClient(create_app(evaluator=evaluator))
    outbound_bodies: list[dict[str, object]] = []

    def asgi_sender(url: str, body: dict[str, object], timeout: float):
        assert url == "http://local-pipeline/v1/systemone"
        outbound_bodies.append(deepcopy(body))
        response = client.post("/v1/systemone", json=body)
        return response.status_code, response.json()

    monkeypatch.setenv("RUN_LIVE_TESTS", "1")
    artifact = run_evaluation(
        Settings(local_base_url="http://local-pipeline", model_timeout_seconds=1.0),
        live=True,
        output_path=output_path,
        dataset_path=dataset_path,
        tasks_path=tasks_path,
        sender=asgi_sender,
    )

    assert len(outbound_bodies) == 1
    assert set(outbound_bodies[0]) == {"state", "model", "questions"}
    assert outbound_bodies[0]["model"] == "gpt-local"
    assert len(artifact["records"]) == 2
    assert artifact["metadata"]["model"] == "pipeline-model"
    assert artifact["records"][0]["expected"] == artifact["records"][0]["actual"] == "approve"
    assert artifact["records"][1]["expected"] is artifact["records"][1]["actual"] is True
    assert artifact["records"][1]["native_confidence"] == 0.9
    assert artifact["records"][1]["raw_answer"] == {
        "type": "noul", "noul": 0.9, "answer": True, "confidence": 0.9
    }
