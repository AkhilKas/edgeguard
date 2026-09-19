"""Abstract interface for guardrail classifiers.

Allows the heuristic baseline and the Week 4 distilled ML model to be
swapped behind the same interface without touching the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    CLEAN = "clean"
    INJECTED = "injected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ClassifierResult:
    verdict: Verdict
    confidence: float  # 0.0–1.0 probability of injection
    matched_pattern: str | None = None  # for heuristic mode, the rule that fired


class BaseClassifier(ABC):
    @abstractmethod
    def classify(self, prompt: str) -> ClassifierResult:
        """Classify a prompt and return a structured result."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for logging."""
