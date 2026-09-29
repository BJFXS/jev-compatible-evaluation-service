"""Semantic validation and deterministic normalization tests for Module 2."""

from __future__ import annotations

from copy import deepcopy
from math import inf, nan

import pytest

from app.config import Settings
from app.errors import InvalidModelOutputError
from app.normalization import (
    NOUL_BOOLEAN_THRESHOLD,
    PROBABILITY_SUM_TOLERANCE,
    normalize_choice,
    normalize_noul,
)
from app.schemas import ChoiceModelOutput, ChoiceQuestion, NoulModelOutput


def choice_question() -> ChoiceQuestion:
    return ChoiceQuestion(
        type="choice",
        instructions="Choose the applicable review outcome.",
        criteria={"approve": "The review passes.", "reject": "The review fails."},
    )


def choice_output(choice: object, probabilities: object) -> ChoiceModelOutput:
    """Bypass structural parsing so semantic validation is tested directly."""

    return ChoiceModelOutput.model_construct(choice=choice, probabilities=probabilities)


def noul_output(probability_yes: object) -> NoulModelOutput:
    """Bypass structural parsing so semantic validation is tested directly."""

    return NoulModelOutput.model_construct(probability_yes=probability_yes)


def test_normalize_choice_exact_valid_distribution() -> None:
    answer = normalize_choice(
        choice_question(),
        ChoiceModelOutput(choice="approve", probabilities={"approve": 0.8, "reject": 0.2}),
    )

    assert answer.type == "choice"
    assert answer.choice == "approve"
    assert answer.probabilities == {"approve": 0.8, "reject": 0.2}
    assert answer.confidence == 0.8


@pytest.mark.parametrize(
    "probabilities",
    [
        {"approve": 0.4995, "reject": 0.4995},
        {"approve": 0.5005, "reject": 0.5005},
    ],
)
def test_normalize_choice_accepts_and_normalizes_sum_within_tolerance(
    probabilities: dict[str, float],
) -> None:
    answer = normalize_choice(choice_question(), choice_output("approve", probabilities))

    assert answer.probabilities["approve"] == pytest.approx(0.5)
    assert answer.probabilities["reject"] == pytest.approx(0.5)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.confidence == pytest.approx(0.5)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"approve": 0.75, "reject": 0.26},
        {"approve": 0.75, "reject": 0.24},
        {"approve": 0.75, "reject": 0.2599},
    ],
)
def test_normalize_choice_accepts_exact_and_inside_tolerance_boundaries(
    probabilities: dict[str, float],
) -> None:
    answer = normalize_choice(choice_question(), choice_output("approve", probabilities))

    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"approve": 0.75, "reject": 0.2601},
        {"approve": 0.75, "reject": 0.2399},
    ],
)
def test_normalize_choice_rejects_just_outside_tolerance_boundaries(
    probabilities: dict[str, float],
) -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("approve", probabilities))


def test_normalization_constants_use_frozen_settings_defaults() -> None:
    settings = Settings()

    assert PROBABILITY_SUM_TOLERANCE == settings.probability_sum_tolerance
    assert NOUL_BOOLEAN_THRESHOLD == settings.noul_boolean_threshold


def test_normalize_choice_rejects_zero_sum() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("approve", {"approve": 0.0, "reject": 0.0}))


def test_normalize_choice_rejects_sum_outside_tolerance() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("approve", {"approve": 0.6, "reject": 0.42}))


@pytest.mark.parametrize("value", [-0.1, 1.1, nan, inf, True, "0.5"])
def test_normalize_choice_rejects_invalid_probability(value: object) -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("approve", {"approve": value, "reject": 0.5}))


def test_normalize_choice_rejects_missing_probability_key() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("approve", {"approve": 1.0}))


def test_normalize_choice_rejects_extra_probability_key() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(
            choice_question(),
            choice_output("approve", {"approve": 0.5, "reject": 0.4, "defer": 0.1}),
        )


def test_normalize_choice_rejects_choice_outside_criteria() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("defer", {"approve": 0.5, "reject": 0.5}))


def test_normalize_choice_rejects_choice_that_is_not_argmax() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(choice_question(), choice_output("reject", {"approve": 0.7, "reject": 0.3}))


def test_normalize_choice_accepts_valid_tie_selection() -> None:
    answer = normalize_choice(
        choice_question(), choice_output("reject", {"approve": 0.5, "reject": 0.5})
    )

    assert answer.choice == "reject"
    assert answer.confidence == 0.5


def test_normalize_choice_rejects_invalid_tie_selection() -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_choice(
            choice_question(),
            choice_output("defer", {"approve": 0.5, "reject": 0.5}),
        )


def test_normalize_choice_does_not_mutate_inputs() -> None:
    question = choice_question()
    model_output = ChoiceModelOutput(
        choice="approve", probabilities={"approve": 0.5005, "reject": 0.5005}
    )
    question_before = question.model_dump()
    output_before = model_output.model_dump()
    probabilities_before = deepcopy(model_output.probabilities)

    answer = normalize_choice(question, model_output)

    assert question.model_dump() == question_before
    assert model_output.model_dump() == output_before
    assert model_output.probabilities == probabilities_before
    assert answer.probabilities is not model_output.probabilities


@pytest.mark.parametrize(
    ("probability_yes", "expected_answer", "expected_confidence"),
    [
        (0.0, False, 1.0),
        (0.49, False, 0.51),
        (0.5, True, 0.5),
        (1.0, True, 1.0),
    ],
)
def test_normalize_noul_derives_answer_and_confidence(
    probability_yes: float, expected_answer: bool, expected_confidence: float
) -> None:
    answer = normalize_noul(NoulModelOutput(probability_yes=probability_yes))

    assert answer.type == "noul"
    assert answer.noul == probability_yes
    assert answer.answer is expected_answer
    assert answer.confidence == expected_confidence


def test_normalize_noul_does_not_mutate_input() -> None:
    model_output = NoulModelOutput(probability_yes=0.5)
    output_before = model_output.model_dump()

    normalize_noul(model_output)

    assert model_output.model_dump() == output_before


@pytest.mark.parametrize("probability_yes", [-0.1, 1.1, nan, inf, True, "0.5"])
def test_normalize_noul_rejects_invalid_probability(probability_yes: object) -> None:
    with pytest.raises(InvalidModelOutputError):
        normalize_noul(noul_output(probability_yes))
