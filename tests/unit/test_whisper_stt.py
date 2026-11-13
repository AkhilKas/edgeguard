"""Unit tests for WhisperSTT (mocked faster-whisper)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from edgeguard.config.settings import STTSettings
from edgeguard.stt.whisper_engine import WhisperSTT


def test_transcribe_requires_load() -> None:
    stt = WhisperSTT(STTSettings(enabled=True))
    with pytest.raises(RuntimeError, match="load"):
        stt.transcribe(b"abc", filename="a.webm")


def test_transcribe_rejects_empty_and_bad_suffix(mocker) -> None:
    stt = WhisperSTT(STTSettings(enabled=True))
    stt._model = mocker.MagicMock()

    with pytest.raises(ValueError, match="Empty"):
        stt.transcribe(b"", filename="a.webm")

    with pytest.raises(ValueError, match="Unsupported"):
        stt.transcribe(b"abc", filename="a.txt")


def test_transcribe_joins_segments(mocker) -> None:
    stt = WhisperSTT(STTSettings(enabled=True, language="en"))
    seg1 = SimpleNamespace(text="  Turn off ")
    seg2 = SimpleNamespace(text=" the lights ")
    model = mocker.MagicMock()
    model.transcribe.return_value = ([seg1, seg2], SimpleNamespace())
    stt._model = model

    text, latency = stt.transcribe(b"fake-audio-bytes", filename="cmd.webm")

    assert text == "Turn off the lights"
    assert latency >= 0
    model.transcribe.assert_called_once()
    path_arg = model.transcribe.call_args[0][0]
    assert Path(path_arg).suffix == ".webm"


def test_load_and_unload(mocker) -> None:
    fake_model = mocker.MagicMock()
    fake_cls = mocker.MagicMock(return_value=fake_model)
    mocker.patch.dict(
        "sys.modules",
        {"faster_whisper": SimpleNamespace(WhisperModel=fake_cls)},
    )

    stt = WhisperSTT(STTSettings(enabled=True, model_size="tiny"))
    stt.load()

    assert stt.is_loaded is True
    fake_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    stt.unload()
    assert stt.is_loaded is False


def test_load_raises_clear_error_when_import_fails(mocker) -> None:
    import builtins

    real_import = builtins.__import__

    def blocker(name: str, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            raise ImportError("not installed")
        return real_import(name, globals, locals, fromlist, level)

    mocker.patch.object(builtins, "__import__", side_effect=blocker)
    stt = WhisperSTT(STTSettings(enabled=True))

    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[stt\]\""):
        stt.load()
