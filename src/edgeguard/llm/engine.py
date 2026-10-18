"""LlamaCpp-backed LLM engine.

Wraps llama-cpp-python and exposes the BaseLLM interface.
Only this file knows about llama_cpp — the rest of the app is agnostic.
"""

import structlog
from llama_cpp import Llama

from edgeguard.config.settings import LLMSettings
from edgeguard.llm.base import BaseLLM, InferenceResult

log = structlog.get_logger(__name__)

# Fixed conversation turns prepended before every real prompt. Small models
# follow concrete examples far more reliably than prose rules alone — in
# testing, these examples were what actually produced consistent one-sentence
# confirmations and stopped the model from asking clarifying questions back,
# which the system prompt's wording alone did not reliably achieve.
FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {"role": "user", "content": "do not open a camera"},
    {"role": "assistant", "content": "Okay, I will not take a photo."},
    {"role": "user", "content": "Turn off the kitchen lights"},
    {"role": "assistant", "content": "Turning off the kitchen lights."},
    {"role": "user", "content": "What time is it"},
    {"role": "assistant", "content": "Sorry, I do not have access to the time."},
    {"role": "user", "content": "Is the garage door open"},
    {"role": "assistant", "content": "Sorry, I do not have a sensor reading for that."},
    {"role": "user", "content": "Lock the front door"},
    {"role": "assistant", "content": "Locking the front door."},
]


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
        output = self._model.create_chat_completion(
            messages=[
                {"role": "system", "content": self._settings.system_prompt},
                *FEW_SHOT_EXAMPLES,
                {"role": "user", "content": prompt},
            ],
            max_tokens=self._settings.max_tokens,
            temperature=self._settings.temperature,
            repeat_penalty=self._settings.repeat_penalty,
            stop=self._settings.stop_tokens,
        )
        text: str = (output["choices"][0]["message"]["content"] or "").strip()
        usage = output.get("usage", {})

        log.debug("llm.inference_done", output_len=len(text))
        return InferenceResult(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model_name=self._settings.model_path.stem,
        )

    def wants_photo(self, prompt: str) -> bool:
        """True only when the user is asking for a picture now."""
        if self._model is None:
            raise RuntimeError("Call load() before wants_photo()")

        output = self._model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Decide whether the user wants a picture taken right now. "
                        "A mention, a question, or a refusal is no. "
                        "Reply with one word: yes or no."
                    ),
                },
                {"role": "user", "content": "capture me"},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": "do not open a camera"},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": "what is a camera"},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": "take a photo of the room"},
                {"role": "assistant", "content": "yes"},
                {"role": "user", "content": "the camera is on the shelf"},
                {"role": "assistant", "content": "no"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4,
            temperature=0,
            stop=["\n"],
        )
        answer = (output["choices"][0]["message"]["content"] or "").strip().lower()
        return answer.startswith("yes")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            log.info("llm.unloaded")
