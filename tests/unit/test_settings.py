"""Unit tests for configuration loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from edgeguard.config.settings import (
    APISettings,
    GuardrailMode,
    GuardrailSettings,
    LLMSettings,
    LogLevel,
    MCUSettings,
    Settings,
    get_settings,
)


def test_llm_settings_raises_when_model_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.gguf"
    with pytest.raises(ValidationError, match="Model file not found"):
        LLMSettings(model_path=missing_path)


def test_llm_settings_accepts_existing_model(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.touch()

    settings = LLMSettings(model_path=model_file, context_window=1024, n_threads=2)

    assert settings.model_path == model_file
    assert settings.context_window == 1024
    assert settings.n_threads == 2


def test_guardrail_settings_defaults() -> None:
    settings = GuardrailSettings()

    assert settings.mode == GuardrailMode.HEURISTIC
    assert settings.model_path is None
    assert settings.enabled is True
    assert 0.0 <= settings.confidence_threshold <= 1.0


def test_mcu_settings_can_be_disabled() -> None:
    settings = MCUSettings(enabled=False)

    assert settings.enabled is False
    assert settings.port == "/dev/ttyACM0"


def test_api_settings_rejects_privileged_port() -> None:
    with pytest.raises(ValidationError):
        APISettings(port=80)


def test_api_settings_accepts_valid_port() -> None:
    settings = APISettings(port=9000)

    assert settings.port == 9000


def test_settings_assembles_nested_sections(tmp_path: Path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.touch()

    settings = Settings(
        llm=LLMSettings(model_path=model_file),
        guardrail=GuardrailSettings(),
        mcu=MCUSettings(enabled=False),
        api=APISettings(),
    )

    assert settings.llm.model_path == model_file
    assert settings.mcu.enabled is False
    assert settings.log_level == LogLevel.INFO
    assert settings.deployment_context == "home_assistant"


def test_get_settings_returns_settings_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    monkeypatch.setenv("LLM_MODEL_PATH", str(model_file))
    monkeypatch.setenv("MCU_ENABLED", "false")

    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.mcu.enabled is False
