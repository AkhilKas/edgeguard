"""Unit tests for the MCU serial interface.

pyserial and threading are mocked out so tests run without real hardware
and without spinning up actual background threads.
"""

import json

import pytest

from edgeguard.config.settings import MCUSettings
from edgeguard.mcu.interface import MCUInterface


@pytest.fixture
def enabled_settings() -> MCUSettings:
    return MCUSettings(enabled=True, port="/dev/ttyFAKE", baud_rate=9600, timeout=1.0)


@pytest.fixture
def disabled_settings() -> MCUSettings:
    return MCUSettings(enabled=False)


def test_connect_skips_when_disabled(disabled_settings: MCUSettings, mocker) -> None:
    mock_serial_cls = mocker.patch("edgeguard.mcu.interface.serial.Serial")
    mcu = MCUInterface(disabled_settings)

    mcu.connect()

    mock_serial_cls.assert_not_called()
    assert mcu._serial is None


def test_connect_opens_serial_and_starts_reader(
    enabled_settings: MCUSettings, mocker
) -> None:
    mock_serial_instance = mocker.MagicMock()
    mock_serial_cls = mocker.patch(
        "edgeguard.mcu.interface.serial.Serial", return_value=mock_serial_instance
    )
    mock_thread_cls = mocker.patch("edgeguard.mcu.interface.threading.Thread")
    mcu = MCUInterface(enabled_settings)

    mcu.connect()

    mock_serial_cls.assert_called_once_with(
        port=enabled_settings.port,
        baudrate=enabled_settings.baud_rate,
        timeout=enabled_settings.timeout,
    )
    mock_thread_cls.assert_called_once_with(
        target=mcu._read_loop, daemon=True, name="mcu-reader"
    )
    mock_thread_cls.return_value.start.assert_called_once()


def test_disconnect_closes_open_serial(enabled_settings: MCUSettings, mocker) -> None:
    mcu = MCUInterface(enabled_settings)
    mock_serial = mocker.MagicMock()
    mock_serial.is_open = True
    mcu._serial = mock_serial
    mcu._running = True

    mcu.disconnect()

    mock_serial.close.assert_called_once()
    assert mcu._running is False


def test_on_prompt_registers_callback(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)
    received: list[str] = []

    mcu.on_prompt(received.append)
    mcu._dispatch(json.dumps({"event": "PROMPT", "text": "turn off the lights"}))

    assert received == ["turn off the lights"]


def test_send_status_noop_when_not_connected(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)

    mcu.send_status("clean")  # should not raise


def test_send_status_writes_json_payload(enabled_settings: MCUSettings, mocker) -> None:
    mcu = MCUInterface(enabled_settings)
    mock_serial = mocker.MagicMock()
    mock_serial.is_open = True
    mcu._serial = mock_serial

    mcu.send_status("blocked")

    sent_bytes = mock_serial.write.call_args[0][0]
    payload = json.loads(sent_bytes.decode())
    assert payload == {"cmd": "SET_STATUS", "status": "blocked"}


def test_send_status_noop_when_serial_closed(
    enabled_settings: MCUSettings, mocker
) -> None:
    mcu = MCUInterface(enabled_settings)
    mock_serial = mocker.MagicMock()
    mock_serial.is_open = False
    mcu._serial = mock_serial

    mcu.send_status("clean")

    mock_serial.write.assert_not_called()


def test_dispatch_ignores_malformed_json(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)
    received: list[str] = []
    mcu.on_prompt(received.append)

    mcu._dispatch("not valid json {{{")

    assert received == []


def test_dispatch_ignores_unknown_event(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)
    received: list[str] = []
    mcu.on_prompt(received.append)

    mcu._dispatch(json.dumps({"event": "SOMETHING_ELSE"}))

    assert received == []


def test_dispatch_noop_without_registered_callback(
    enabled_settings: MCUSettings,
) -> None:
    mcu = MCUInterface(enabled_settings)

    # Should not raise even though no callback has been registered.
    mcu._dispatch(json.dumps({"event": "PROMPT", "text": "hello"}))


def test_read_loop_dispatches_lines_until_stopped(
    enabled_settings: MCUSettings, mocker
) -> None:
    mcu = MCUInterface(enabled_settings)
    received: list[str] = []
    mcu.on_prompt(received.append)

    mock_serial = mocker.MagicMock()
    mock_serial.is_open = True
    line = json.dumps({"event": "PROMPT", "text": "lock the door"}).encode() + b"\n"

    def fake_readline() -> bytes:
        mcu._running = False
        return line

    mock_serial.readline.side_effect = fake_readline
    mcu._serial = mock_serial
    mcu._running = True

    mcu._read_loop()

    assert received == ["lock the door"]


def test_read_loop_stops_on_serial_exception(
    enabled_settings: MCUSettings, mocker
) -> None:
    import serial

    mcu = MCUInterface(enabled_settings)
    mock_serial = mocker.MagicMock()
    mock_serial.is_open = True
    mock_serial.readline.side_effect = serial.SerialException("device disconnected")
    mcu._serial = mock_serial
    mcu._running = True

    mcu._read_loop()  # should exit the loop instead of raising
