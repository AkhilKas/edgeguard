"""Guardrail factory.

Reads settings and returns the correct BaseClassifier implementation.
The pipeline and API never instantiate classifiers directly.
"""

from edgeguard.config.settings import GuardrailMode, GuardrailSettings
from edgeguard.guardrail.base import BaseClassifier
from edgeguard.guardrail.heuristic import HeuristicClassifier
from edgeguard.guardrail.ml_classifier import MLClassifier


def build_classifier(settings: GuardrailSettings) -> BaseClassifier:
    if settings.mode == GuardrailMode.HEURISTIC:
        return HeuristicClassifier()

    if settings.mode == GuardrailMode.ML:
        if settings.model_path is None:
            raise ValueError("GUARDRAIL_MODEL_PATH must be set when GUARDRAIL_MODE=ml")
        if settings.tokenizer_path is None:
            raise ValueError(
                "GUARDRAIL_TOKENIZER_PATH must be set when GUARDRAIL_MODE=ml"
            )
        return MLClassifier(
            model_path=settings.model_path,
            tokenizer_path=settings.tokenizer_path,
            threshold=settings.confidence_threshold,
        )

    raise ValueError(f"Unknown guardrail mode: {settings.mode}")
