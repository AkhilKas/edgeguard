"""Record a short clip from a local ALSA mic (e.g. Logitech Brio on the UNO Q).

Uses the system ``arecord`` binary so we don't need an extra Python audio stack.
Install with: ``sudo apt install alsa-utils``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def record_wav(
    *,
    seconds: float,
    alsa_device: str = "default",
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """Capture mono PCM WAV from the board mic. Returns raw .wav bytes."""
    if seconds <= 0:
        raise ValueError("Record duration must be positive")

    arecord = shutil.which("arecord")
    if arecord is None:
        raise RuntimeError(
            "arecord not found. Install ALSA tools: sudo apt install alsa-utils"
        )

    duration = max(1, int(round(seconds)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        arecord,
        "-D",
        alsa_device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        str(channels),
        "-d",
        str(duration),
        "-t",
        "wav",
        "-q",
        str(out_path),
    ]
    log.info(
        "stt.device_record.start",
        device=alsa_device,
        seconds=duration,
        sample_rate=sample_rate,
    )
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=duration + 10,
        )
        if completed.returncode != 0:
            err = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"arecord failed (device={alsa_device!r}): {err or completed.returncode}"
            )
        data = out_path.read_bytes()
        if len(data) < 100:
            raise RuntimeError("arecord produced an empty clip — check the mic device")
        log.info("stt.device_record.done", bytes=len(data))
        return data
    finally:
        out_path.unlink(missing_ok=True)
