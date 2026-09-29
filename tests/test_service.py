"""Service orchestration tests using real schemas and normalization with a fake provider."""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from app.errors import InvalidModelOutputError, ModelProviderError
from app.schemas import ChoiceModelOutput, ChoiceQuestion, NoulModelOutput, State, SystemOneRequest
from app.service import SystemOneService
from tests.fakes import EvaluatorCall, FakeEvaluator


def choice_question() -> dict[str, object]:
    return {
        "type": "choice",
        "instructions": "Choose the review decision.",
        "criteria": {"approve": "Review passes.", "revise": "Review needs revision."},
    }


def noul_question() -> dict[str, object]:
    return {"type": "noul", "instructions": "Does the data contain confirmation?"}


def request(questions: dict[str, object], state: object = "shared state") -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {"state": state, "model": "gpt-local", "questions": questions}
    )


def run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def success_choice() -> ChoiceModelOutput:
    return ChoiceModelOutput(choice="approve", probabilities={"approve": 0.8, "revise": 0.2})


def success_noul(probability_yes: float = 0.75) -> NoulModelOutput:
    return NoulModelOutput(probability_yes=probability_yes)


def test_choice_dispatch_normalizes_and_uses_actual_model_name() -> None:
    evaluator = FakeEvaluator(choice_outcomes=[success_choice()])

    response = run(SystemOneService(evaluator).evaluate_request(request({"decision": choice_question()})))

    assert response.model == "fake-configured-model"
    assert response.answers["decision"].model_dump() == {
        "type": "choice",
        "choice": "approve",
        "probabilities": {"approve": 0.8, "revise": 0.2},
        "confidence": 0.8,
    }
    assert [call.kind for call in evaluator.calls] == ["choice"]


def test_noul_dispatch_normalizes() -> None:
    evaluator = FakeEvaluator(noul_outcomes=[success_noul(0.5)])

    response = run(SystemOneService(evaluator).evaluate_request(request({"condition": noul_question()})))

    assert response.answers["condition"].model_dump() == {
        "type": "noul", "noul": 0.5, "answer": True, "confidence": 0.5
    }
    assert [call.kind for call in evaluator.calls] == ["noul"]


def test_multiple_questions_preserve_names_order_and_shared_state() -> None:
    shared_state = {"text": "The review is confirmed."}
    evaluator = FakeEvaluator(choice_outcomes=[success_choice()], noul_outcomes=[success_noul()])

    response = run(
        SystemOneService(evaluator).evaluate_request(
            request({"first": choice_question(), "second": noul_question()}, shared_state)
        )
    )

    assert list(response.answers) == ["first", "second"]
    assert [call.state for call in evaluator.calls] == [shared_state, shared_state]
    assert evaluator.calls[0].state is not evaluator.calls[1].state


def test_service_does_not_mutate_request_state_or_leak_prior_answer() -> None:
    shared_state = {"items": ["original"]}
    request_value = request({"decision": choice_question(), "condition": noul_question()}, shared_state)
    state_before = deepcopy(request_value.state)
    evaluator = FakeEvaluator(choice_outcomes=[success_choice()], noul_outcomes=[success_noul()])

    response = run(SystemOneService(evaluator).evaluate_request(request_value))

    assert request_value.state == state_before
    assert evaluator.calls[0].state == state_before
    assert evaluator.calls[1].state == state_before
    assert "decision" not in evaluator.calls[1].state
    assert set(response.answers) == {"decision", "condition"}


def test_mutating_evaluator_cannot_change_request_or_later_question_state() -> None:
    class MutatingFakeEvaluator(FakeEvaluator):
        async def evaluate_choice(self, state: object, question: object, feedback: str | None = None) -> ChoiceModelOutput:
            state["items"].append("mutated")  # type: ignore[index]
            return await super().evaluate_choice(state, question, feedback)  # type: ignore[arg-type]

    shared_state = {"items": ["original"]}
    evaluator = MutatingFakeEvaluator(choice_outcomes=[success_choice()], noul_outcomes=[success_noul()])

    run(
        SystemOneService(evaluator).evaluate_request(
            request({"decision": choice_question(), "condition": noul_question()}, shared_state)
        )
    )

    assert shared_state == {"items": ["original"]}
    assert evaluator.calls[1].state == {"items": ["original"]}


def test_transient_first_failure_then_success_retries_once_with_fake_sleep() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    evaluator = FakeEvaluator(
        choice_outcomes=[ModelProviderError("OpenAI provider timed out"), success_choice()]
    )

    response = run(
        SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request({"decision": choice_question()}))
    )

    assert response.answers["decision"].choice == "approve"
    assert len(evaluator.calls) == 2
    assert sleeps == [1.0]
    assert [call.feedback for call in evaluator.calls] == [None, None]


def test_retry_attempts_receive_fresh_state_copies() -> None:
    class MutatingRetryEvaluator(FakeEvaluator):
        async def evaluate_choice(
            self, state: State, question: ChoiceQuestion, feedback: str | None = None
        ) -> ChoiceModelOutput:
            self.calls.append(EvaluatorCall("choice", deepcopy(state), question, feedback))
            state["items"].append("mutated")  # type: ignore[index,union-attr]
            if len(self.calls) == 1:
                raise ModelProviderError("OpenAI provider timed out")
            return success_choice()

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    shared_state = {"items": ["original"]}
    request_value = request({"decision": choice_question()}, shared_state)
    evaluator = MutatingRetryEvaluator()

    response = run(SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request_value))

    assert response.answers["decision"].choice == "approve"
    assert [call.state for call in evaluator.calls] == [
        {"items": ["original"]},
        {"items": ["original"]},
    ]
    assert request_value.state == {"items": ["original"]}
    assert shared_state == {"items": ["original"]}
    assert len(evaluator.calls) == 2
    assert sleeps == [1.0]


def test_invalid_output_then_corrected_output_retries_with_contract_feedback() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    evaluator = FakeEvaluator(
        choice_outcomes=[
            ChoiceModelOutput.model_construct(
                choice="approve", probabilities={"approve": 0.1, "revise": 0.1}
            ),
            success_choice(),
        ]
    )

    response = run(
        SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request({"decision": choice_question()}))
    )

    assert response.answers["decision"].choice == "approve"
    assert len(evaluator.calls) == 2
    assert sleeps == [1.0]
    assert evaluator.calls[1].feedback is not None
    assert "correct" not in evaluator.calls[1].feedback.lower()


def test_retry_exhaustion_uses_total_two_attempt_budget() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    evaluator = FakeEvaluator(
        choice_outcomes=[InvalidModelOutputError("invalid one"), InvalidModelOutputError("invalid two")]
    )

    with pytest.raises(InvalidModelOutputError):
        run(SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request({"decision": choice_question()})))

    assert len(evaluator.calls) == 2
    assert sleeps == [1.0]


def test_mixed_retryable_errors_still_have_only_two_attempts() -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    evaluator = FakeEvaluator(
        choice_outcomes=[
            ModelProviderError("OpenAI provider transport failed"),
            InvalidModelOutputError("invalid output"),
        ]
    )

    with pytest.raises(InvalidModelOutputError):
        run(SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request({"decision": choice_question()})))

    assert len(evaluator.calls) == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    "error",
    [
        ModelProviderError("OpenAI model refused evaluation"),
        ModelProviderError("OpenAI provider authentication failed"),
        ModelProviderError("OpenAI API key configuration is missing"),
    ],
)
def test_non_retryable_provider_errors_do_not_sleep_or_retry(error: ModelProviderError) -> None:
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    evaluator = FakeEvaluator(noul_outcomes=[error])

    with pytest.raises(ModelProviderError):
        run(SystemOneService(evaluator, sleep=fake_sleep).evaluate_request(request({"condition": noul_question()})))

    assert len(evaluator.calls) == 1
    assert sleeps == []


def test_second_question_failure_fails_request_without_partial_response() -> None:
    evaluator = FakeEvaluator(
        choice_outcomes=[success_choice()],
        noul_outcomes=[ModelProviderError("OpenAI model refused evaluation")],
    )

    with pytest.raises(ModelProviderError):
        run(
            SystemOneService(evaluator).evaluate_request(
                request({"decision": choice_question(), "condition": noul_question()})
            )
        )

    assert [call.kind for call in evaluator.calls] == ["choice", "noul"]
