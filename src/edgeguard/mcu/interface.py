"""RPC bridge between the Linux side and the STM32U585 MCU.

Responsibilities (single):
  Send structured status updates to the MCU; receive prompt strings from it.

The UNO Q's Linux <-> MCU UART link is owned by the Arduino Router and is
not exposed to application code as a raw serial port. All communication
goes through the Router Bridge (MessagePack-RPC over a local socket).

Contract with the MCU sketch:
  Linux -> MCU:  bridge.notify("set_status", status)
                 status: "clean" | "blocked" | "uncertain"
  MCU -> Linux:  Bridge.call("on_prompt", text) or Bridge.notify("on_prompt", text)
                 (the MCU sketch must `Bridge.provide()` nothing for this —
                 it calls into the name this class provides on the Linux side)
"""

from __future__ import annotations

from collections.abc import Callable

import structlog
from arduino.router_bridge import Bridge

from edgeguard.config.settings import MCUSettings

log = structlog.get_logger(__name__)

PromptCallback = Callable[[str], None]


class MCUInterface:
    """Manages the Router Bridge connection to the STM32 MCU.

    The bridge library owns its own connection and dispatcher threads, so
    this class is a thin wrapper mapping EdgeGuard's method names onto it.
    """

    def __init__(self, settings: MCUSettings) -> None:
        self._settings = settings
        self._bridge: Bridge | None = None
        self._prompt_callback: PromptCallback | None = None

    def connect(self) -> None:
        if not self._settings.enabled:
            log.info("mcu.disabled", reason="MCU_ENABLED=false")
            return

        self._bridge = Bridge(address=self._settings.address)
        self._bridge.provide("on_prompt", self._handle_prompt)

        if self._bridge.connect(timeout=self._settings.timeout):
            log.info("mcu.connected", address=self._settings.address)
        else:
            log.warning("mcu.connect_timeout", address=self._settings.address)

    def disconnect(self) -> None:
        if self._bridge is not None:
            self._bridge.disconnect()
            log.info("mcu.disconnected")

    def on_prompt(self, callback: PromptCallback) -> None:
        """Register a callback that fires when the MCU sends a prompt."""
        self._prompt_callback = callback

    def send_status(self, status: str) -> None:
        """Notify the MCU of a verdict (clean | blocked | uncertain).

        Fire-and-forget: best-effort, dropped silently if not connected.
        """
        if self._bridge is None:
            log.debug("mcu.send_skipped", reason="not_connected")
            return
        self._bridge.notify("set_status", status)
        log.debug("mcu.sent", status=status)

    def _handle_prompt(self, text: str) -> None:
        """Invoked by the bridge's dispatcher thread when the MCU calls "on_prompt"."""
        if self._prompt_callback is None:
            log.debug("mcu.prompt_dropped", reason="no_callback")
            return
        self._prompt_callback(text)
