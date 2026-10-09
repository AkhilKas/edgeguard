"""ML-based guardrail classifier.

Runs a DistilBERT prompt-injection classifier, exported to ONNX, via
onnxruntime. Swap in by setting GUARDRAIL_MODE=ml, GUARDRAIL_MODEL_PATH=<onnx
file>, and GUARDRAIL_TOKENIZER_PATH=<tokenizer.json> in .env.

See src/edgeguard/training/edge_guard_training.ipynb for how these two
artifacts are produced: it fine-tunes distilbert-base-uncased on
deepset/prompt-injections (label 0 = safe, label 1 = injection) and exports
the model to ONNX (inputs "input_ids"/"attention_mask", output "logits",
shape [batch, 2]) alongside the matching tokenizer's tokenizer.json.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import structlog
from tokenizers import Tokenizer

from edgeguard.guardrail.base import BaseClassifier, ClassifierResult, Verdict

log = structlog.get_logger(__name__)

# Must match the padding/truncation length used during training.
MAX_LENGTH = 128


class MLClassifier(BaseClassifier):
    """Distilled DistilBERT classifier for prompt injection detection."""

    def __init__(
        self, model_path: Path, tokenizer_path: Path, threshold: float = 0.75
    ) -> None:
        self._threshold = threshold
        self._session = self._load_model(model_path)
        self._tokenizer = self._load_tokenizer(tokenizer_path)

    def _load_model(self, path: Path) -> ort.InferenceSession:
        try:
            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            log.info("guardrail.ml_model_loaded", path=str(path))
            return session
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {path}: {e}") from e

    def _load_tokenizer(self, path: Path) -> Tokenizer:
        try:
            tokenizer = Tokenizer.from_file(str(path))
            # The HF tokenizer wrapper used in training applies padding and
            # truncation at call time rather than baking it into the saved
            # tokenizer.json, so we have to re-enable both here ourselves.
            tokenizer.enable_padding(length=MAX_LENGTH)
            tokenizer.enable_truncation(max_length=MAX_LENGTH)
            log.info("guardrail.ml_tokenizer_loaded", path=str(path))
            return tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer from {path}: {e}") from e

    def classify(self, prompt: str) -> ClassifierResult:
        encoding = self._tokenizer.encode(prompt)
        input_ids: np.ndarray = np.asarray([encoding.ids], dtype=np.int64)
        attention_mask: np.ndarray = np.asarray(
            [encoding.attention_mask], dtype=np.int64
        )

        (logits,) = self._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        proba = float(_softmax(logits[0])[1])  # P(injection)

        if proba >= self._threshold:
            return ClassifierResult(verdict=Verdict.INJECTED, confidence=proba)
        if proba >= self._threshold * 0.7:
            return ClassifierResult(verdict=Verdict.UNCERTAIN, confidence=proba)
        return ClassifierResult(verdict=Verdict.CLEAN, confidence=1 - proba)

    @property
    def name(self) -> str:
        return "ml_distilbert"


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    result: np.ndarray = exp / exp.sum()
    return result
