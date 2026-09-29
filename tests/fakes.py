"""Deterministic provider fakes for service-layer tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas import ChoiceModelOutput, ChoiceQuestion, NoulModelOutput, NoulQuestion, State


@dataclass
class EvaluatorCall:
    kind: str
    state: State
    question: ChoiceQuestion | NoulQuestion
    feedback: str | None


@dataclass
class FakeEvaluator:
    model_name: str = "fake-configured-model"
    choice_outcomes: list[ChoiceModelOutput | Exception] = field(default_factory=list)
    noul_outcomes: list[NoulModelOutput | Exception] = field(default_factory=list)
    calls: list[EvaluatorCall] = field(default_factory=list)

    async def evaluate_choice(
        self, state: State, question: ChoiceQuestion, feedback: str | None = None
    ) -> ChoiceModelOutput:
        self.calls.append(EvaluatorCall("choice", state, question, feedback))
        return self._next(self.choice_outcomes)

    async def evaluate_noul(
        self, state: State, question: NoulQuestion, feedback: str | None = None
    ) -> NoulModelOutput:
        self.calls.append(EvaluatorCall("noul", state, question, feedback))
        return self._next(self.noul_outcomes)

    @staticmethod
    def _next(outcomes: list[Any]) -> Any:
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
