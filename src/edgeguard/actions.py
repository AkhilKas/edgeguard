"""Take a picture after the model decides the user asked for one."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

PHOTO_DIR = Path("/home/arduino/photos")


def take_photo() -> str:
    try:
        import cv2
    except ImportError:
        return "Camera failed: opencv is not installed."

    index = _camera_index()
    if index is None:
        return "Camera failed: no camera device found."

    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return f"Camera failed: could not read /dev/video{index}."

    PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = PHOTO_DIR / f"shot-{datetime.now():%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(path), frame)
    log.info("action.photo", path=str(path), device=index)
    return f"Photo saved to {path}."


def _camera_index() -> int | None:
    root = Path("/sys/class/video4linux")
    if not root.exists():
        return None
    for node in sorted(root.glob("video*")):
        name = (node / "name").read_text(encoding="utf-8").lower()
        if "venus" in name or "qcom" in name:
            continue
        return int(node.name.removeprefix("video"))
    return None
