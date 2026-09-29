"""Pure semantic validation and deterministic answer normalization."""

from __future__ import annotations

from math import isclose, isfinite, ulp

from app.config import Settings
from app.errors import InvalidModelOutputError
from app.schemas import (
    ChoiceAnswer,
    ChoiceModelOutput,
    ChoiceQuestion,
    NoulAnswer,
    NoulModelOutput,
)


# Scalars are derived from the frozen Settings defaults without loading .env.
PROBABILITY_SUM_TOLERANCE = Settings().probability_sum_tolerance
NOUL_BOOLEAN_THRESHOLD = Settings().noul_boolean_threshold


def _validated_probability(value: object) -> float:
    """Return a finite JSON number in the closed probability interval."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidModelOutputError("probability must be a number")

    probability = float(value)
    if not isfinite(probability):
        raise InvalidModelOutputError("probability must be finite")
    if not 0 <= probability <= 1:
        raise InvalidModelOutputError("probability must be between 0 and 1")
    return probability


def _sum_is_within_tolerance(total: float) -> bool:
    """Apply the inclusive frozen tolerance despite float representation noise."""

    difference = abs(total - 1.0)
    if difference <= PROBABILITY_SUM_TOLERANCE:
        return True
    rounding_allowance = 4 * max(ulp(total), ulp(PROBABILITY_SUM_TOLERANCE))
    return isclose(
        difference,
        PROBABILITY_SUM_TOLERANCE,
        rel_tol=0.0,
        abs_tol=rounding_allowance,
    )


def normalize_choice(question: ChoiceQuestion, model_output: ChoiceModelOutput) -> ChoiceAnswer:
    """Validate and normalize a Choice model output into a local API answer."""

    criteria_keys = set(question.criteria)
    probabilities = model_output.probabilities
    if not isinstance(model_output.choice, str):
        raise InvalidModelOutputError("selected choice must be a string")
    if not isinstance(probabilities, dict):
        raise InvalidModelOutputError("probabilities must be an object")
    if set(probabilities) != criteria_keys:
        raise InvalidModelOutputError("probability keys must exactly match choice criteria")
    if model_output.choice not in criteria_keys:
        raise InvalidModelOutputError("selected choice must exist in choice criteria")

    validated_probabilities = {
        key: _validated_probability(value) for key, value in probabilities.items()
    }
    total = sum(validated_probabilities.values())
    if total <= 0:
        raise InvalidModelOutputError("probabilities must sum to more than zero")
    if not _sum_is_within_tolerance(total):
        raise InvalidModelOutputError("probabilities must sum to approximately one")

    normalized_probabilities = {
        key: probability / total for key, probability in validated_probabilities.items()
    }
    maximum = max(normalized_probabilities.values())
    if normalized_probabilities[model_output.choice] != maximum:
        raise InvalidModelOutputError("selected choice must have maximum probability")

    return ChoiceAnswer(
        type="choice",
        choice=model_output.choice,
        probabilities=normalized_probabilities,
        confidence=maximum,
    )


def normalize_noul(model_output: NoulModelOutput) -> NoulAnswer:
    """Validate Noul probability and derive the local boolean answer and confidence."""

    probability_yes = _validated_probability(model_output.probability_yes)
    return NoulAnswer(
        type="noul",
        noul=probability_yes,
        answer=probability_yes >= NOUL_BOOLEAN_THRESHOLD,
        confidence=max(probability_yes, 1 - probability_yes),
    )
