"""LlamaCpp-backed LLM engine.

Wraps llama-cpp-python and exposes the BaseLLM interface.
Only this file knows about llama_cpp — the rest of the app is agnostic.
"""

import structlog
from llama_cpp import Llama

from edgeguard.config.settings import LLMSettings
from edgeguard.llm.base import BaseLLM, InferenceResult

log = structlog.get_logger(__name__)


class LlamaCppEngine(BaseLLM):
    """Runs a GGUF model locally via llama-cpp-python."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._model: Llama | None = None

    def load(self) -> None:
        """Load the model into memory. Call once at startup."""
        if self._model is not None:
            log.warning("llm.already_loaded")
            return

        log.info(
            "llm.loading",
            path=str(self._settings.model_path),
            n_ctx=self._settings.context_window,
            n_threads=self._settings.n_threads,
        )
        self._model = Llama(
            model_path=str(self._settings.model_path),
            n_ctx=self._settings.context_window,
            n_threads=self._settings.n_threads,
            verbose=False,
        )
        log.info("llm.loaded")

    def generate(self, prompt: str) -> InferenceResult:
        if self._model is None:
            raise RuntimeError("Call load() before generate()")

        log.debug("llm.inference_start", prompt_len=len(prompt))
        output = self._model(
            prompt,
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            stop=self._settings.stop_tokens,
            echo=False,
        )
        text: str = output["choices"][0]["text"].strip()
        usage = output.get("usage", {})

        log.debug("llm.inference_done", output_len=len(text))
        return InferenceResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model_name=self._settings.model_path.stem,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            log.info("llm.unloaded")
