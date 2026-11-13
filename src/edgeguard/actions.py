"""Take a picture after the model decides the user asked for one."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def take_photo(
    photo_dir: Path,
    *,
    warmup_seconds: float = 1.5,
    discard_frames: int = 8,
) -> str:
    try:
        import cv2
    except ImportError:
        return "Camera failed: opencv is not installed."

    index = _camera_index()
    if index is None:
        return "Camera failed: no camera device found."

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return f"Camera failed: could not open /dev/video{index}."

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # type: ignore[attr-defined]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Keep the device streaming so USB cams (e.g. Brio) turn their activity
    # LED on and auto-exposure can settle before we save a frame.
    deadline = time.monotonic() + max(0.5, warmup_seconds)
    last_ok = False
    last_frame = None
    dropped = 0
    while time.monotonic() < deadline or dropped < discard_frames:
        ok, frame = cap.read()
        if ok:
            last_ok, last_frame = True, frame
            dropped += 1
        else:
            time.sleep(0.05)
        if time.monotonic() >= deadline and dropped >= discard_frames:
            break

    ok, frame = cap.read()
    cap.release()
    if ok:
        last_ok, last_frame = True, frame

    if not last_ok or last_frame is None:
        return f"Camera failed: could not read /dev/video{index}."

    photo_dir.mkdir(parents=True, exist_ok=True)
    path = photo_dir / f"shot-{datetime.now():%Y%m%d-%H%M%S}.jpg"
    cv2.imwrite(str(path), last_frame)
    log.info(
        "action.photo",
        path=str(path),
        device=index,
        warmup_seconds=warmup_seconds,
        discarded_frames=dropped,
    )
    return f"Photo saved to {path}."


def _camera_index(root: Path = Path("/sys/class/video4linux")) -> int | None:
    if not root.exists():
        return None
    for node in sorted(root.glob("video*")):
        name = (node / "name").read_text(encoding="utf-8").lower()
        if "venus" in name or "qcom" in name:
            continue
        return int(node.name.removeprefix("video"))
    return None
