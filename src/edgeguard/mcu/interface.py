"""UART interface between the Linux side and the STM32U585 MCU.

Responsibilities (single):
  Send structured status commands to the MCU; receive prompt strings from it.

Protocol (newline-delimited JSON over UART):
  Linux → MCU:  {"cmd": "SET_STATUS", "status": "clean|blocked|uncertain"}
  MCU → Linux:  {"event": "PROMPT", "text": "turn off the lights"}
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import serial
import structlog

from edgeguard.config.settings import MCUSettings

log = structlog.get_logger(__name__)

PromptCallback = Callable[[str], None]


class MCUInterface:
    """Manages serial communication with the STM32 MCU.

    Thread-safe: a background thread reads incoming messages and dispatches
    them via the registered prompt callback.
    """

    def __init__(self, settings: MCUSettings) -> None:
        self._settings = settings
        self._serial: serial.Serial | None = None
        self._reader_thread: threading.Thread | None = None
        self._running = False
        self._prompt_callback: PromptCallback | None = None

    def connect(self) -> None:
        if not self._settings.enabled:
            log.info("mcu.disabled", reason="MCU_ENABLED=false")
            return
        self._serial = serial.Serial(
            port=self._settings.port,
            baudrate=self._settings.baud_rate,
            timeout=self._settings.timeout,
        )
        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True, name="mcu-reader"
        )
        self._reader_thread.start()
        log.info("mcu.connected", port=self._settings.port)

    def disconnect(self) -> None:
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("mcu.disconnected")

    def on_prompt(self, callback: PromptCallback) -> None:
        """Register a callback that fires when the MCU sends a prompt."""
        self._prompt_callback = callback

    def send_status(self, status: str) -> None:
        """Send a status update to the MCU (clean | blocked | uncertain)."""
        if not self._serial or not self._serial.is_open:
            log.debug("mcu.send_skipped", reason="not_connected")
            return
        payload = json.dumps({"cmd": "SET_STATUS", "status": status}) + "\n"
        self._serial.write(payload.encode())
        log.debug("mcu.sent", payload=payload.strip())

    def _read_loop(self) -> None:
        while self._running and self._serial and self._serial.is_open:
            try:
                line = self._serial.readline().decode().strip()
                if not line:
                    continue
                self._dispatch(line)
            except serial.SerialException as e:
                log.error("mcu.read_error", error=str(e))
                break

    def _dispatch(self, raw: str) -> None:
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("mcu.malformed_message", raw=raw)
            return

        if msg.get("event") == "PROMPT" and self._prompt_callback:
            self._prompt_callback(msg["text"])
        else:
            log.debug("mcu.unknown_event", msg=msg)
