"""ML-based guardrail classifier (Week 4+).

Loads a distilled text classifier from disk.
Swap in by setting GUARDRAIL_MODE=ml and GUARDRAIL_MODEL_PATH=<path> in .env.
The interface is identical to HeuristicClassifier — the pipeline doesn't change.
"""

from pathlib import Path

import structlog

from edgeguard.guardrail.base import BaseClassifier, ClassifierResult, Verdict

log = structlog.get_logger(__name__)


class MLClassifier(BaseClassifier):
    """Distilled text classifier for prompt injection detection.

    Expected model format: a scikit-learn Pipeline or HuggingFace
    transformers model saved with joblib/pickle.
    Swap the loader below once the Week 4 training is done.
    """

    def __init__(self, model_path: Path, threshold: float = 0.75) -> None:
        self._threshold = threshold
        self._model = self._load(model_path)

    def _load(self, path: Path) -> object:
        try:
            import joblib

            model = joblib.load(path)
            log.info("guardrail.ml_loaded", path=str(path))
            return model
        except ImportError as e:
            raise RuntimeError("Install joblib: pip install joblib") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load ML classifier from {path}: {e}") from e

    def classify(self, prompt: str) -> ClassifierResult:
        # Adjust this call once the trained model interface is known
        proba = self._model.predict_proba([prompt])[0][1]  # type: ignore[attr-defined]
        if proba >= self._threshold:
            return ClassifierResult(
                verdict=Verdict.INJECTED,
                confidence=float(proba),
            )
        if proba >= self._threshold * 0.7:
            return ClassifierResult(
                verdict=Verdict.UNCERTAIN,
                confidence=float(proba),
            )
        return ClassifierResult(
            verdict=Verdict.CLEAN,
            confidence=float(1 - proba),
        )

    @property
    def name(self) -> str:
        return "ml_distilled"
