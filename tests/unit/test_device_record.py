"""Unit tests for board-side ALSA recording."""

from pathlib import Path

import pytest

from edgeguard.stt.device_record import record_wav


def test_record_wav_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        record_wav(seconds=0)


def test_record_wav_requires_arecord(mocker) -> None:
    mocker.patch("edgeguard.stt.device_record.shutil.which", return_value=None)
    with pytest.raises(RuntimeError, match="arecord"):
        record_wav(seconds=2, alsa_device="default")


def test_record_wav_runs_arecord_and_returns_bytes(mocker) -> None:
    mocker.patch(
        "edgeguard.stt.device_record.shutil.which", return_value="/usr/bin/arecord"
    )

    def fake_run(cmd, check, capture_output, text, timeout):  # noqa: ARG001
        out = cmd[-1]
        Path(out).write_bytes(b"RIFF" + b"\x00" * 200)
        return mocker.MagicMock(returncode=0, stderr="", stdout="")

    mocker.patch("edgeguard.stt.device_record.subprocess.run", side_effect=fake_run)

    data = record_wav(seconds=2.0, alsa_device="plughw:1,0", sample_rate=16000)

    assert data.startswith(b"RIFF")
    assert len(data) > 100


def test_record_wav_surfaces_arecord_errors(mocker) -> None:
    mocker.patch(
        "edgeguard.stt.device_record.shutil.which", return_value="/usr/bin/arecord"
    )
    mocker.patch(
        "edgeguard.stt.device_record.subprocess.run",
        return_value=mocker.MagicMock(returncode=1, stderr="device busy", stdout=""),
    )

    with pytest.raises(RuntimeError, match="arecord failed"):
        record_wav(seconds=1, alsa_device="default")
