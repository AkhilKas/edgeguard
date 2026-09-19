"""Unit tests for the LlamaCpp-backed LLM engine.

The real llama_cpp.Llama class is mocked out so no model file needs to
be loaded into memory.
"""

from pathlib import Path

import pytest

from edgeguard.config.settings import LLMSettings
from edgeguard.llm.engine import LlamaCppEngine


@pytest.fixture
def llm_settings(tmp_path: Path) -> LLMSettings:
    model_file = tmp_path / "model.gguf"
    model_file.touch()
    return LLMSettings(model_path=model_file, context_window=1024, n_threads=2)


def test_not_loaded_before_load_called(llm_settings: LLMSettings) -> None:
    engine = LlamaCppEngine(llm_settings)

    assert engine.is_loaded() is False


def test_generate_before_load_raises(llm_settings: LLMSettings) -> None:
    engine = LlamaCppEngine(llm_settings)

    with pytest.raises(RuntimeError, match="Call load"):
        engine.generate("hello")


def test_load_constructs_llama_with_settings(llm_settings: LLMSettings, mocker) -> None:
    mock_llama_cls = mocker.patch("edgeguard.llm.engine.Llama")
    engine = LlamaCppEngine(llm_settings)

    engine.load()

    mock_llama_cls.assert_called_once_with(
        model_path=str(llm_settings.model_path),
        n_ctx=llm_settings.context_window,
        n_threads=llm_settings.n_threads,
        verbose=False,
    )
    assert engine.is_loaded() is True


def test_load_is_idempotent(llm_settings: LLMSettings, mocker) -> None:
    mock_llama_cls = mocker.patch("edgeguard.llm.engine.Llama")
    engine = LlamaCppEngine(llm_settings)

    engine.load()
    engine.load()

    mock_llama_cls.assert_called_once()


def test_generate_returns_inference_result(llm_settings: LLMSettings, mocker) -> None:
    mock_model = mocker.MagicMock()
    mock_model.return_value = {
        "choices": [{"text": "  a helpful reply  "}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }
    mocker.patch("edgeguard.llm.engine.Llama", return_value=mock_model)
    engine = LlamaCppEngine(llm_settings)
    engine.load()

    result = engine.generate("hello there")

    mock_model.assert_called_once_with(
        "hello there",
        max_tokens=llm_settings.max_tokens,
        temperature=llm_settings.temperature,
        stop=llm_settings.stop_tokens,
        echo=False,
    )
    assert result.text == "a helpful reply"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 3
    assert result.model_name == llm_settings.model_path.stem


def test_generate_defaults_usage_when_missing(
    llm_settings: LLMSettings, mocker
) -> None:
    mock_model = mocker.MagicMock()
    mock_model.return_value = {"choices": [{"text": "reply"}]}
    mocker.patch("edgeguard.llm.engine.Llama", return_value=mock_model)
    engine = LlamaCppEngine(llm_settings)
    engine.load()

    result = engine.generate("hi")

    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0


def test_unload_releases_model(llm_settings: LLMSettings, mocker) -> None:
    mocker.patch("edgeguard.llm.engine.Llama")
    engine = LlamaCppEngine(llm_settings)
    engine.load()

    engine.unload()

    assert engine.is_loaded() is False


def test_unload_when_not_loaded_is_a_noop(llm_settings: LLMSettings) -> None:
    engine = LlamaCppEngine(llm_settings)

    engine.unload()

    assert engine.is_loaded() is False
