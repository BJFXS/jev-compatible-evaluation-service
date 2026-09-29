"""HTTP entry-layer tests with the real service and normalization layers."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.errors import ModelProviderError
from app.main import create_app
from app.schemas import ChoiceModelOutput, NoulModelOutput
from app.service import SystemOneService
from tests.fakes import FakeEvaluator


def choice_question() -> dict[str, object]:
    return {
        "type": "choice",
        "instructions": "Choose the document disposition.",
        "criteria": {"accept": "The document meets policy.", "reject": "The document does not meet policy."},
    }


def noul_question() -> dict[str, object]:
    return {"type": "noul", "instructions": "Does the document include approval?"}


def choice_output() -> ChoiceModelOutput:
    return ChoiceModelOutput(choice="accept", probabilities={"accept": 0.8, "reject": 0.2})


def noul_output() -> NoulModelOutput:
    return NoulModelOutput(probability_yes=0.75)


def payload(questions: dict[str, object], state: object = "document text") -> dict[str, object]:
    return {"state": state, "model": "gpt-local", "questions": questions}


def client_for(evaluator: FakeEvaluator) -> TestClient:
    async def fake_sleep(_: float) -> None:
        return None

    return TestClient(create_app(service=SystemOneService(evaluator, sleep=fake_sleep)))


def test_valid_choice_returns_normalized_local_response() -> None:
    response = client_for(FakeEvaluator(choice_outcomes=[choice_output()])).post(
        "/v1/systemone", json=payload({"disposition": choice_question()})
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "fake-configured-model",
        "answers": {
            "disposition": {
                "type": "choice",
                "choice": "accept",
                "probabilities": {"accept": 0.8, "reject": 0.2},
                "confidence": 0.8,
            }
        },
    }


def test_injected_evaluator_does_not_require_provider_credentials() -> None:
    response = TestClient(create_app(evaluator=FakeEvaluator(choice_outcomes=[choice_output()]))).post(
        "/v1/systemone", json=payload({"disposition": choice_question()})
    )

    assert response.status_code == 200


def test_valid_noul_returns_local_extension_fields() -> None:
    response = client_for(FakeEvaluator(noul_outcomes=[noul_output()])).post(
        "/v1/systemone", json=payload({"approved": noul_question()})
    )

    assert response.status_code == 200
    assert response.json()["answers"]["approved"] == {
        "type": "noul",
        "noul": 0.75,
        "answer": True,
        "confidence": 0.75,
    }


def test_multiple_questions_return_named_answers() -> None:
    response = client_for(
        FakeEvaluator(choice_outcomes=[choice_output()], noul_outcomes=[noul_output()])
    ).post(
        "/v1/systemone", json=payload({"disposition": choice_question(), "approved": noul_question()})
    )

    assert response.status_code == 200
    assert list(response.json()["answers"]) == ["disposition", "approved"]


@pytest.mark.parametrize("state", ["plain text", {"text": "nested data"}, ["list", {"data": True}]])
def test_state_string_dict_and_list_are_accepted(state: object) -> None:
    response = client_for(FakeEvaluator(choice_outcomes=[choice_output()])).post(
        "/v1/systemone", json=payload({"disposition": choice_question()}, state)
    )

    assert response.status_code == 200


@pytest.mark.parametrize("field", ["state", "model", "questions"])
def test_missing_required_field_returns_safe_422(field: str) -> None:
    request_body = payload({"disposition": choice_question()})
    del request_body[field]

    response = client_for(FakeEvaluator()).post("/v1/systemone", json=request_body)

    assert response.status_code == 422
    assert response.json() == {
        "error": {"code": "invalid_request", "message": "The request does not match the local contract."}
    }


@pytest.mark.parametrize(
    "questions",
    [
        {},
        {"disposition": {"type": "unsupported", "instructions": "No."}},
        {"disposition": {"type": "choice", "instructions": "Choose."}},
    ],
)
def test_invalid_questions_return_safe_422(questions: dict[str, object]) -> None:
    response = client_for(FakeEvaluator()).post("/v1/systemone", json=payload(questions))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_invalid_model_output_after_retry_returns_502() -> None:
    invalid_output = ChoiceModelOutput.model_construct(
        choice="accept", probabilities={"accept": 0.1, "reject": 0.1}
    )
    response = client_for(FakeEvaluator(choice_outcomes=[invalid_output, invalid_output])).post(
        "/v1/systemone", json=payload({"disposition": choice_question()})
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


def test_temporary_provider_failure_after_retry_returns_503() -> None:
    error = ModelProviderError("OpenAI provider timed out")
    response = client_for(FakeEvaluator(choice_outcomes=[error, error])).post(
        "/v1/systemone", json=payload({"disposition": choice_question()})
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_refusal_returns_safe_502_without_provider_details() -> None:
    secret = "not-a-real-api-key"
    response = client_for(FakeEvaluator(noul_outcomes=[ModelProviderError(f"OpenAI model refused: {secret}")])).post(
        "/v1/systemone", json=payload({"approved": noul_question()}, {"private": secret})
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {"code": "model_refusal", "message": "The model refused the evaluation."}
    }
    assert secret not in response.text
    assert "private" not in response.text
    assert "traceback" not in response.text.lower()


def test_provider_configuration_error_is_safe() -> None:
    response = client_for(
        FakeEvaluator(noul_outcomes=[ModelProviderError("OpenAI API key configuration is missing")])
    ).post("/v1/systemone", json=payload({"approved": noul_question()}))

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "model_configuration_error"
    assert "api key" not in response.text.lower()


def test_unexpected_error_returns_safe_500() -> None:
    async def fake_sleep(_: float) -> None:
        return None

    client = TestClient(
        create_app(service=SystemOneService(FakeEvaluator(choice_outcomes=[RuntimeError("internal detail")]), sleep=fake_sleep)),
        raise_server_exceptions=False,
    )
    response = client.post(
        "/v1/systemone", json=payload({"disposition": choice_question()})
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "The service could not complete the request."}
    }
    assert "internal detail" not in response.text


def test_root_main_is_untouched_and_entrypoint_is_app_main() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert (repository_root / "main.py").read_bytes() == b""
    assert Path(create_app.__code__.co_filename).resolve() == repository_root / "app" / "main.py"
