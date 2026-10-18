"""Abstract interface for LLM backends.

Concrete implementations (LlamaCppEngine, MockEngine for tests) all satisfy
this contract. The pipeline never imports a concrete class directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_name: str


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> InferenceResult:
        """Run inference and return a structured result."""

    def wants_photo(self, prompt: str) -> bool:
        """Return True when the user is asking for a picture right now."""
        return False

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is loaded and ready."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
