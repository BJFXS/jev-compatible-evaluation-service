"""Sequential request orchestration with a bounded, service-owned retry policy."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Awaitable, Callable

from app.config import Settings
from app.errors import InvalidModelOutputError, ModelProviderError, UnsupportedQuestionTypeError
from app.model import Evaluator
from app.normalization import normalize_choice, normalize_noul
from app.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    State,
    SystemOneRequest,
    SystemOneResponse,
)


MAX_MODEL_RETRIES = Settings().max_model_retries
RETRY_BACKOFF_SECONDS = 1.0
Sleep = Callable[[float], Awaitable[None]]

_RETRYABLE_PROVIDER_MESSAGES = (
    "timed out",
    "transport failed",
    "rate limited",
    "server error",
    "response was incomplete",
)


class SystemOneService:
    """Evaluate named questions sequentially with one total retry budget each."""

    def __init__(self, evaluator: Evaluator, *, sleep: Sleep = asyncio.sleep) -> None:
        self._evaluator = evaluator
        self._sleep = sleep

    async def evaluate_request(self, request: SystemOneRequest) -> SystemOneResponse:
        """Evaluate every question or fail the entire request without partial answers."""

        # This private snapshot protects the caller. Each provider attempt receives
        # its own copy below, so a provider cannot contaminate a retry or later question.
        state_snapshot = deepcopy(request.state)
        answers: dict[str, ChoiceAnswer | NoulAnswer] = {}
        for name, question in request.questions.items():
            answers[name] = await self._evaluate_question(state_snapshot, question)
        return SystemOneResponse(model=self._evaluator.model_name, answers=answers)

    async def _evaluate_question(
        self, state_snapshot: State, question: ChoiceQuestion | NoulQuestion
    ) -> ChoiceAnswer | NoulAnswer:
        feedback: str | None = None
        for attempt in range(MAX_MODEL_RETRIES + 1):
            try:
                if isinstance(question, ChoiceQuestion):
                    output = await self._evaluator.evaluate_choice(
                        deepcopy(state_snapshot), question, feedback
                    )
                    return normalize_choice(question, output)
                if isinstance(question, NoulQuestion):
                    output = await self._evaluator.evaluate_noul(
                        deepcopy(state_snapshot), question, feedback
                    )
                    return normalize_noul(output)
                raise UnsupportedQuestionTypeError("unsupported question type")
            except (InvalidModelOutputError, ModelProviderError) as error:
                if attempt >= MAX_MODEL_RETRIES or not self._is_retryable(error):
                    raise
                feedback = self._feedback_for(error)
                await self._sleep(RETRY_BACKOFF_SECONDS)
        raise AssertionError("retry loop must return or raise")

    @staticmethod
    def _is_retryable(error: InvalidModelOutputError | ModelProviderError) -> bool:
        if isinstance(error, InvalidModelOutputError):
            return True
        message = str(error).lower()
        return any(marker in message for marker in _RETRYABLE_PROVIDER_MESSAGES)

    @staticmethod
    def _feedback_for(error: InvalidModelOutputError | ModelProviderError) -> str | None:
        if isinstance(error, InvalidModelOutputError):
            return "The previous output violated the required output schema or probability rules. Return a valid output."
        return None


async def evaluate_request(
    request: SystemOneRequest, evaluator: Evaluator, *, sleep: Sleep = asyncio.sleep
) -> SystemOneResponse:
    """Convenience entry point for the future FastAPI layer."""

    return await SystemOneService(evaluator, sleep=sleep).evaluate_request(request)
