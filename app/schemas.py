"""Foundation-layer Pydantic schemas for local SystemOne data contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator, model_validator


MAX_QUESTIONS = 10

# State is intentionally opaque. Only its outer JSON shape is constrained here.
State: TypeAlias = StrictStr | dict[str, Any] | list[Any]
Probability = Annotated[
    float,
    Field(strict=True, ge=0, le=1, allow_inf_nan=False),
]


class _ContractModel(BaseModel):
    """Base model that rejects fields outside the frozen local contract."""

    model_config = ConfigDict(extra="forbid")


def _require_non_blank(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


class ChoiceQuestion(_ContractModel):
    """A named classification question with caller-supplied choices."""

    type: Literal["choice"]
    instructions: StrictStr
    criteria: dict[StrictStr, StrictStr] = Field(min_length=2, max_length=255)

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "instructions")

    @field_validator("criteria")
    @classmethod
    def criteria_values_must_not_be_blank(cls, value: dict[str, str]) -> dict[str, str]:
        for key, description in value.items():
            _require_non_blank(key, "criteria key")
            _require_non_blank(description, "criteria description")
        return value


class NoulQuestion(_ContractModel):
    """A binary true/false question with optional fixed criteria descriptions."""

    type: Literal["noul"]
    instructions: StrictStr
    criteria: dict[StrictStr, StrictStr] | None = None

    @field_validator("instructions")
    @classmethod
    def instructions_must_not_be_blank(cls, value: str) -> str:
        return _require_non_blank(value, "instructions")

    @model_validator(mode="after")
    def criteria_must_match_boolean_contract(self) -> NoulQuestion:
        if self.criteria is None:
            return self
        if set(self.criteria) != {"true", "false"}:
            raise ValueError('noul criteria must contain exactly "true" and "false"')
        for description in self.criteria.values():
            _require_non_blank(description, "noul criteria description")
        return self


Question = Annotated[ChoiceQuestion | NoulQuestion, Field(discriminator="type")]


### FastAPI 使用 SystemOneRequest 验证请求
class SystemOneRequest(_ContractModel):
    """Validated input to the local SystemOne evaluation service."""

    state: State
    model: Literal["gpt-local"]
    questions: dict[StrictStr, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)

    @field_validator("questions")
    @classmethod
    def question_names_must_not_be_blank(cls, value: dict[str, Question]) -> dict[str, Question]:
        for name in value:
            _require_non_blank(name, "question name")
        return value


class ChoiceModelOutput(_ContractModel):
    """Model-derived fields for a Choice question before normalization."""

    choice: StrictStr
    probabilities: dict[StrictStr, Probability]


class NoulModelOutput(_ContractModel):
    """Model-derived field for a Noul question before normalization."""

    probability_yes: Probability


class ChoiceAnswer(_ContractModel):
    """Final local API answer for a Choice question."""

    type: Literal["choice"]
    choice: StrictStr
    probabilities: dict[StrictStr, Probability]
    confidence: Probability


class NoulAnswer(_ContractModel):
    """Final local API answer for a Noul question."""

    type: Literal["noul"]
    noul: Probability
    answer: StrictBool
    confidence: Probability


Answer = Annotated[ChoiceAnswer | NoulAnswer, Field(discriminator="type")]


class SystemOneResponse(_ContractModel):
    """Successful local SystemOne response with typed named answers."""

    model: StrictStr
    answers: dict[StrictStr, Answer] = Field(min_length=1)
