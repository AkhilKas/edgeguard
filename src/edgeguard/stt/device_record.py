"""Record a short clip from a local ALSA mic (e.g. Logitech Brio on the UNO Q).

Uses the system ``arecord`` binary so we don't need an extra Python audio stack.
Install with: ``sudo apt install alsa-utils``.

``default`` often fails on USB webcams with ``Invalid argument`` when forcing
16 kHz mono — prefer ``plughw:CARD,DEV`` (rate conversion) or ``auto``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

_USB_HINTS = ("usb", "brio", "logitech", "c920", "c922", "webcam", "microphone")


def list_capture_devices() -> list[tuple[str, str]]:
    """Return [(plughw:N,M, human_name), ...] from ``arecord -l``."""
    arecord = shutil.which("arecord")
    if arecord is None:
        return []
    completed = subprocess.run(
        [arecord, "-l"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    text = completed.stdout or ""
    found: list[tuple[str, str]] = []
    # arecord -l prints one card header then device lines; match card+device pairs
    for match in re.finditer(
        r"card\s+(\d+):\s+(\S+)\s+\[([^\]]+)\],\s*device\s+(\d+):\s+([^\n]+)",
        text,
        re.IGNORECASE,
    ):
        card, card_id, card_name, dev, dev_name = match.groups()
        plughw = f"plughw:{card},{dev}"
        label = f"{card_name.strip()} / {dev_name.strip()} ({card_id})"
        found.append((plughw, label))
    return found


def resolve_alsa_device(requested: str) -> str:
    """Map settings value to an arecord -D string.

    ``auto`` / empty / ``default`` → first USB-like capture device, else first
    listed plughw, else ``plug:default``.
    """
    req = (requested or "").strip()
    if req and req.lower() not in {"auto", "default"}:
        # Prefer plughw over hw so rate/channel conversion is allowed.
        if req.startswith("hw:"):
            return "plug" + req  # hw:1,0 -> plughw:1,0
        return req

    devices = list_capture_devices()
    if not devices:
        log.warning("stt.alsa_no_devices", fallback="plug:default")
        return "plug:default"

    for plughw, label in devices:
        lowered = label.lower()
        if any(hint in lowered for hint in _USB_HINTS):
            log.info("stt.alsa_autodetect", device=plughw, label=label)
            return plughw

    plughw, label = devices[0]
    log.info("stt.alsa_autodetect_first", device=plughw, label=label)
    return plughw


def record_wav(
    *,
    seconds: float,
    alsa_device: str = "auto",
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

    device = resolve_alsa_device(alsa_device)
    duration = max(1, int(round(seconds)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = Path(tmp.name)

    cmd = [
        arecord,
        "-D",
        device,
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
        requested=alsa_device,
        device=device,
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
            listed = list_capture_devices()
            hint = (
                " Run `arecord -l` on the board and set STT_ALSA_DEVICE=plughw:CARD,DEV "
                "in .env (USB Brio is usually not 'default')."
            )
            if listed:
                options = ", ".join(f"{d} ({name})" for d, name in listed)
                hint += f" Seen: {options}"
            raise RuntimeError(
                f"arecord failed (device={device!r}): {err or completed.returncode}.{hint}"
            )
        data = out_path.read_bytes()
        if len(data) < 100:
            raise RuntimeError("arecord produced an empty clip — check the mic device")
        log.info("stt.device_record.done", bytes=len(data), device=device)
        return data
    finally:
        out_path.unlink(missing_ok=True)
