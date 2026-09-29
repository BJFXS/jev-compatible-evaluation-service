"""Contract tests for Module 1 request and response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    ChoiceAnswer,
    ChoiceModelOutput,
    NoulAnswer,
    NoulModelOutput,
    SystemOneResponse,
    SystemOneRequest,
)


def choice_question(**overrides: object) -> dict[str, object]:
    question: dict[str, object] = {
        "type": "choice",
        "instructions": "Classify the message.",
        "criteria": {
            "approve": "The request should be approved.",
            "reject": "The request should be rejected.",
        },
    }
    question.update(overrides)
    return question


def noul_question(**overrides: object) -> dict[str, object]:
    question: dict[str, object] = {
        "type": "noul",
        "instructions": "Does the text satisfy the condition?",
    }
    question.update(overrides)
    return question


def request_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "state": "Please approve my expense report.",
        "model": "gpt-local",
        "questions": {"decision": choice_question()},
    }
    payload.update(overrides)
    return payload


def test_valid_choice() -> None:
    request = SystemOneRequest.model_validate(request_payload())

    assert request.questions["decision"].type == "choice"


@pytest.mark.parametrize(
    "question",
    [
        noul_question(),
        noul_question(criteria=None),
        noul_question(criteria={"true": "Matches.", "false": "Does not match."}),
    ],
)
def test_valid_noul(question: dict[str, object]) -> None:
    request = SystemOneRequest.model_validate(
        request_payload(questions={"condition": question})
    )

    assert request.questions["condition"].type == "noul"


def test_valid_typed_local_response() -> None:
    response = SystemOneResponse.model_validate(
        {
            "model": "configured-model-name",
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": "approve",
                    "probabilities": {"approve": 0.8, "reject": 0.2},
                    "confidence": 0.8,
                },
                "condition": {
                    "type": "noul",
                    "noul": 0.7,
                    "answer": True,
                    "confidence": 0.7,
                },
            },
        }
    )

    assert isinstance(response.answers["decision"], ChoiceAnswer)
    assert isinstance(response.answers["condition"], NoulAnswer)


def test_valid_multiple_named_choice_and_noul_questions() -> None:
    request = SystemOneRequest.model_validate(
        request_payload(
            questions={"decision": choice_question(), "condition": noul_question()}
        )
    )

    assert set(request.questions) == {"decision", "condition"}


@pytest.mark.parametrize("state", ["plain state", {"utterance": "hello"}, ["one", 2, None]])
def test_valid_state_outer_shapes(state: object) -> None:
    request = SystemOneRequest.model_validate(request_payload(state=state))

    assert request.state == state


def test_valid_generic_non_snips_choice() -> None:
    request = SystemOneRequest.model_validate(
        request_payload(
            questions={
                "review": choice_question(
                    criteria={
                        "publish": "Content meets editorial standards.",
                        "revise": "Content needs an editorial revision.",
                    }
                )
            }
        )
    )

    assert set(request.questions["review"].criteria) == {"publish", "revise"}


@pytest.mark.parametrize("field", ["state", "model", "questions"])
def test_missing_required_request_field_is_rejected(field: str) -> None:
    payload = request_payload()
    del payload[field]

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


def test_empty_questions_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request_payload(questions={}))


def test_too_many_questions_are_rejected() -> None:
    questions = {f"question_{index}": choice_question() for index in range(11)}

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request_payload(questions=questions))


def test_blank_question_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request_payload(questions={"  ": choice_question()}))


def test_unsupported_question_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            request_payload(questions={"unknown": {"type": "score", "instructions": "Rate it"}})
        )


def test_choice_missing_criteria_is_rejected() -> None:
    question = choice_question()
    del question["criteria"]

    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request_payload(questions={"decision": question}))


def test_choice_with_only_one_criterion_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            request_payload(
                questions={
                    "decision": choice_question(criteria={"approve": "Approve the request."})
                }
            )
        )


@pytest.mark.parametrize(
    "criteria",
    [
        {"": "Approve the request.", "reject": "Reject the request."},
        {"approve": "", "reject": "Reject the request."},
    ],
)
def test_choice_empty_key_or_description_is_rejected(criteria: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            request_payload(questions={"decision": choice_question(criteria=criteria)})
        )


@pytest.mark.parametrize(
    "criteria",
    [
        {"true": "Matches"},
        {"true": "Matches", "false": "Does not match", "maybe": "Uncertain"},
        {"true": "", "false": "Does not match"},
    ],
)
def test_noul_invalid_criteria_is_rejected(criteria: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(
            request_payload(questions={"condition": noul_question(criteria=criteria)})
        )


@pytest.mark.parametrize(
    "payload",
    [
        request_payload(extra="forbidden"),
        request_payload(questions={"decision": choice_question(extra="forbidden")}),
    ],
)
def test_unknown_forbidden_fields_are_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(payload)


@pytest.mark.parametrize("state", [None, True, False, 0, 2.5])
def test_invalid_state_outer_shapes_are_rejected(state: object) -> None:
    with pytest.raises(ValidationError):
        SystemOneRequest.model_validate(request_payload(state=state))


@pytest.mark.parametrize("probability", [True, "0.5", -0.01, 1.01, float("nan"), float("inf")])
def test_choice_model_output_rejects_invalid_probability_type_or_range(probability: object) -> None:
    with pytest.raises(ValidationError):
        ChoiceModelOutput.model_validate(
            {"choice": "approve", "probabilities": {"approve": probability}}
        )


@pytest.mark.parametrize("probability", [True, "0.5", -0.01, 1.01, float("nan"), float("inf")])
def test_noul_model_output_rejects_invalid_probability_type_or_range(probability: object) -> None:
    with pytest.raises(ValidationError):
        NoulModelOutput.model_validate({"probability_yes": probability})
