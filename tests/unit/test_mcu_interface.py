"""Unit tests for the MCU Router Bridge interface.

The arduino.router_bridge.Bridge class is mocked out so tests run without
a real Arduino Router socket or MCU hardware.
"""

import pytest

from edgeguard.config.settings import MCUSettings
from edgeguard.mcu.interface import MCUInterface


@pytest.fixture
def enabled_settings() -> MCUSettings:
    return MCUSettings(enabled=True, address="tcp://localhost:9999", timeout=3.0)


@pytest.fixture
def disabled_settings() -> MCUSettings:
    return MCUSettings(enabled=False)


def test_connect_skips_when_disabled(disabled_settings: MCUSettings, mocker) -> None:
    mock_bridge_cls = mocker.patch("edgeguard.mcu.interface.Bridge")
    mcu = MCUInterface(disabled_settings)

    mcu.connect()

    mock_bridge_cls.assert_not_called()
    assert mcu._bridge is None


def test_connect_creates_bridge_provides_handler_and_connects(
    enabled_settings: MCUSettings, mocker
) -> None:
    mock_bridge_instance = mocker.MagicMock()
    mock_bridge_instance.connect.return_value = True
    mock_bridge_cls = mocker.patch(
        "edgeguard.mcu.interface.Bridge", return_value=mock_bridge_instance
    )
    mcu = MCUInterface(enabled_settings)

    mcu.connect()

    mock_bridge_cls.assert_called_once_with(address=enabled_settings.address)
    mock_bridge_instance.provide.assert_called_once_with(
        "on_prompt", mcu._handle_prompt
    )
    mock_bridge_instance.connect.assert_called_once_with(
        timeout=enabled_settings.timeout
    )


def test_connect_logs_warning_on_timeout(enabled_settings: MCUSettings, mocker) -> None:
    mock_bridge_instance = mocker.MagicMock()
    mock_bridge_instance.connect.return_value = False
    mocker.patch("edgeguard.mcu.interface.Bridge", return_value=mock_bridge_instance)
    mcu = MCUInterface(enabled_settings)

    mcu.connect()  # should not raise even though the bridge never connected

    assert mcu._bridge is mock_bridge_instance


def test_disconnect_closes_bridge(enabled_settings: MCUSettings, mocker) -> None:
    mcu = MCUInterface(enabled_settings)
    mock_bridge = mocker.MagicMock()
    mcu._bridge = mock_bridge

    mcu.disconnect()

    mock_bridge.disconnect.assert_called_once()


def test_disconnect_when_never_connected_is_a_noop(
    enabled_settings: MCUSettings,
) -> None:
    mcu = MCUInterface(enabled_settings)

    mcu.disconnect()  # should not raise; _bridge is still None


def test_on_prompt_registers_callback(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)
    received: list[str] = []

    mcu.on_prompt(received.append)
    mcu._handle_prompt("turn off the lights")

    assert received == ["turn off the lights"]


def test_handle_prompt_noop_without_registered_callback(
    enabled_settings: MCUSettings,
) -> None:
    mcu = MCUInterface(enabled_settings)

    mcu._handle_prompt("hello")  # should not raise


def test_send_status_noop_when_not_connected(enabled_settings: MCUSettings) -> None:
    mcu = MCUInterface(enabled_settings)

    mcu.send_status("clean")  # should not raise; _bridge is None


def test_send_status_notifies_bridge(enabled_settings: MCUSettings, mocker) -> None:
    mcu = MCUInterface(enabled_settings)
    mock_bridge = mocker.MagicMock()
    mcu._bridge = mock_bridge

    mcu.send_status("blocked")

    mock_bridge.notify.assert_called_once_with("set_status", "blocked")
