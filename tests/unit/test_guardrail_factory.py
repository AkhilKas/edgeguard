"""Unit tests for the guardrail factory."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from edgeguard.config.settings import GuardrailMode, GuardrailSettings
from edgeguard.guardrail.factory import build_classifier
from edgeguard.guardrail.heuristic import HeuristicClassifier


def test_heuristic_mode_returns_heuristic_classifier() -> None:
    settings = GuardrailSettings(mode=GuardrailMode.HEURISTIC)

    classifier = build_classifier(settings)

    assert isinstance(classifier, HeuristicClassifier)


def test_ml_mode_without_model_path_raises() -> None:
    settings = GuardrailSettings(
        mode=GuardrailMode.ML, model_path=None, tokenizer_path=Path("tokenizer.json")
    )

    with pytest.raises(ValueError, match="GUARDRAIL_MODEL_PATH must be set"):
        build_classifier(settings)


def test_ml_mode_without_tokenizer_path_raises(tmp_path: Path) -> None:
    settings = GuardrailSettings(
        mode=GuardrailMode.ML,
        model_path=tmp_path / "classifier.onnx",
        tokenizer_path=None,
    )

    with pytest.raises(ValueError, match="GUARDRAIL_TOKENIZER_PATH must be set"):
        build_classifier(settings)


def test_ml_mode_constructs_ml_classifier(tmp_path: Path, mocker) -> None:
    model_path = tmp_path / "classifier.onnx"
    tokenizer_path = tmp_path / "tokenizer.json"
    mock_ml_classifier = mocker.patch("edgeguard.guardrail.factory.MLClassifier")
    settings = GuardrailSettings(
        mode=GuardrailMode.ML,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        confidence_threshold=0.6,
    )

    build_classifier(settings)

    mock_ml_classifier.assert_called_once_with(
        model_path=model_path, tokenizer_path=tokenizer_path, threshold=0.6
    )


def test_unknown_mode_raises() -> None:
    fake_settings = SimpleNamespace(mode="not-a-real-mode")

    with pytest.raises(ValueError, match="Unknown guardrail mode"):
        build_classifier(fake_settings)  # type: ignore[arg-type]
