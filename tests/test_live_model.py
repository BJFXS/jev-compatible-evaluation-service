"""Opt-in live smoke test for the OpenAI adapter; skipped by default."""

from __future__ import annotations

import asyncio
import os

import pytest

from app.config import get_settings
from app.model import OpenAIEvaluator
from app.schemas import ChoiceQuestion, NoulQuestion


pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1"
    or not os.getenv("OPENAI_API_KEY")
    or not os.getenv("OPENAI_MODEL"),
    reason="requires RUN_LIVE_TESTS=1 plus OPENAI_API_KEY and OPENAI_MODEL",
)
def test_openai_noul_smoke() -> None:
    evaluator = OpenAIEvaluator(settings=get_settings())
    output = asyncio.run(
        evaluator.evaluate_noul(
            "The account owner confirmed the policy.",
            NoulQuestion(type="noul", instructions="Does the statement contain a confirmation?"),
        )
    )

    assert 0 <= output.probability_yes <= 1


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1"
    or not os.getenv("OPENAI_API_KEY")
    or not os.getenv("OPENAI_MODEL"),
    reason="requires RUN_LIVE_TESTS=1 plus OPENAI_API_KEY and OPENAI_MODEL",
)
def test_openai_choice_smoke() -> None:
    evaluator = OpenAIEvaluator(settings=get_settings())
    question = ChoiceQuestion(
        type="choice",
        instructions="Classify the review outcome.",
        criteria={
            "approve": "The review meets the stated requirements.",
            "revise": "The review needs changes before approval.",
        },
    )
    output = asyncio.run(
        evaluator.evaluate_choice("The review is complete and meets every requirement.", question)
    )

    assert output.choice in question.criteria
    assert set(output.probabilities) == set(question.criteria)
