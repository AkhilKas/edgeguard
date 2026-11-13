"""Whisper-based speech-to-text via faster-whisper.

Optional at runtime: only loaded when STT_ENABLED=true and the
``faster-whisper`` package is installed (``pip install -e ".[stt]"``).
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import structlog

from edgeguard.config.settings import STTSettings

log = structlog.get_logger(__name__)

# Suffixes we accept from the browser MediaRecorder / uploads.
_ALLOWED_SUFFIXES = {".webm", ".wav", ".ogg", ".mp3", ".m4a", ".mpeg", ".mp4"}


class WhisperSTT:
    """Thin wrapper around faster-whisper for short voice commands."""

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._model: object | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "STT enabled but faster-whisper is not installed. "
                'Run: pip install -e ".[stt]"'
            ) from e

        log.info(
            "stt.loading",
            model_size=self._settings.model_size,
            device=self._settings.device,
            compute_type=self._settings.compute_type,
        )
        self._model = WhisperModel(
            self._settings.model_size,
            device=self._settings.device,
            compute_type=self._settings.compute_type,
        )
        log.info("stt.loaded")

    def unload(self) -> None:
        self._model = None
        log.info("stt.unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def transcribe(
        self, audio_bytes: bytes, filename: str = "audio.webm"
    ) -> tuple[str, float]:
        """Transcribe an audio blob. Returns (text, latency_ms)."""
        if self._model is None:
            raise RuntimeError("Call load() before transcribe()")
        if not audio_bytes:
            raise ValueError("Empty audio upload")

        suffix = Path(filename).suffix.lower() or ".webm"
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(
                f"Unsupported audio type '{suffix}'. "
                f"Use one of: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
            )

        t0 = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            # faster-whisper API: transcribe(path) -> (segments, info)
            segments, _info = self._model.transcribe(  # type: ignore[attr-defined]
                tmp.name,
                language=self._settings.language or None,
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "stt.transcribed",
            chars=len(text),
            latency_ms=round(latency_ms, 1),
        )
        return text, latency_ms
