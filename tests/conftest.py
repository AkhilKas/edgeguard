"""Shared pytest fixtures.

All tests use mock LLM and MCU objects so nothing needs
real hardware or a downloaded model to run in CI.
"""

import pytest

from edgeguard.guardrail.heuristic import HeuristicClassifier
from edgeguard.llm.base import BaseLLM, InferenceResult
from edgeguard.pipeline.pipeline import EdgeGuardPipeline


class MockLLM(BaseLLM):
    """Deterministic LLM that echoes the prompt. No model file needed."""

    def generate(self, prompt: str) -> InferenceResult:
        return InferenceResult(
            text=f"MOCK RESPONSE to: {prompt[:40]}",
            prompt_tokens=len(prompt.split()),
            completion_tokens=8,
            model_name="mock",
        )

    def is_loaded(self) -> bool:
        return True

    def unload(self) -> None:
        pass


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def heuristic_classifier() -> HeuristicClassifier:
    return HeuristicClassifier()


@pytest.fixture
def pipeline(
    mock_llm: MockLLM, heuristic_classifier: HeuristicClassifier
) -> EdgeGuardPipeline:
    return EdgeGuardPipeline(
        classifier=heuristic_classifier,
        llm=mock_llm,
        guardrail_enabled=True,
    )
