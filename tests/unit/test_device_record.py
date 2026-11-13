"""Unit tests for board-side ALSA recording."""

from pathlib import Path

import pytest

from edgeguard.stt import device_record
from edgeguard.stt.device_record import record_wav, resolve_alsa_device


def test_record_wav_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="positive"):
        record_wav(seconds=0)


def test_record_wav_requires_arecord(mocker) -> None:
    mocker.patch("edgeguard.stt.device_record.shutil.which", return_value=None)
    with pytest.raises(RuntimeError, match="arecord"):
        record_wav(seconds=2, alsa_device="plughw:1,0")


def test_resolve_passes_through_explicit_plughw() -> None:
    assert resolve_alsa_device("plughw:1,0") == "plughw:1,0"


def test_resolve_maps_hw_to_plughw() -> None:
    assert resolve_alsa_device("hw:1,0") == "plughw:1,0"


def test_resolve_auto_prefers_usb_brio(mocker) -> None:
    listing = """
**** List of CAPTURE Hardware Devices ****
card 0: PCH [HDA Intel], device 0: ALC269 Analog [ALC269 Analog]
card 1: Brio [Logitech BRIO], device 0: USB Audio [USB Audio]
"""
    mocker.patch(
        "edgeguard.stt.device_record.shutil.which", return_value="/usr/bin/arecord"
    )
    mocker.patch(
        "edgeguard.stt.device_record.subprocess.run",
        return_value=mocker.MagicMock(returncode=0, stdout=listing, stderr=""),
    )

    assert resolve_alsa_device("auto") == "plughw:1,0"
    assert resolve_alsa_device("default") == "plughw:1,0"


def test_record_wav_runs_arecord_and_returns_bytes(mocker) -> None:
    mocker.patch(
        "edgeguard.stt.device_record.shutil.which", return_value="/usr/bin/arecord"
    )
    mocker.patch(
        "edgeguard.stt.device_record.resolve_alsa_device", return_value="plughw:1,0"
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
        "edgeguard.stt.device_record.resolve_alsa_device", return_value="default"
    )
    mocker.patch.object(device_record, "list_capture_devices", return_value=[])
    mocker.patch(
        "edgeguard.stt.device_record.subprocess.run",
        return_value=mocker.MagicMock(
            returncode=1, stderr="audio open error: Invalid argument", stdout=""
        ),
    )

    with pytest.raises(RuntimeError, match="arecord failed"):
        record_wav(seconds=1, alsa_device="default")
