"""Unit tests for actions.take_photo() and camera device selection."""

from pathlib import Path

from edgeguard.actions import _camera_index, take_photo


def _make_video_node(root: Path, index: int, name: str) -> None:
    node = root / f"video{index}"
    node.mkdir(parents=True)
    (node / "name").write_text(name, encoding="utf-8")


def test_camera_index_returns_none_when_root_missing(tmp_path: Path) -> None:
    assert _camera_index(root=tmp_path / "does-not-exist") is None


def test_camera_index_skips_venus_and_qcom_nodes(tmp_path: Path) -> None:
    _make_video_node(tmp_path, 0, "qcom-venus-encoder")
    _make_video_node(tmp_path, 1, "qcom-camera-isp")
    _make_video_node(tmp_path, 2, "USB Camera: USB Camera")

    assert _camera_index(root=tmp_path) == 2


def test_camera_index_returns_none_when_only_soc_nodes_exist(tmp_path: Path) -> None:
    _make_video_node(tmp_path, 0, "qcom-venus-encoder")

    assert _camera_index(root=tmp_path) is None


def test_take_photo_reports_missing_opencv(mocker, tmp_path: Path) -> None:
    mocker.patch.dict("sys.modules", {"cv2": None})

    result = take_photo(photo_dir=tmp_path)

    assert result == "Camera failed: opencv is not installed."


def test_take_photo_reports_no_camera_found(mocker, tmp_path: Path) -> None:
    mocker.patch("edgeguard.actions._camera_index", return_value=None)

    result = take_photo(photo_dir=tmp_path)

    assert result == "Camera failed: no camera device found."


def test_take_photo_reports_failed_read(mocker, tmp_path: Path) -> None:
    mocker.patch("edgeguard.actions._camera_index", return_value=3)
    mock_cap = mocker.MagicMock()
    mock_cap.read.return_value = (False, None)
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("edgeguard.actions.time.sleep")

    result = take_photo(photo_dir=tmp_path)

    assert result == "Camera failed: could not read /dev/video3."
    mock_cap.release.assert_called_once()


def test_take_photo_saves_frame_to_photo_dir(mocker, tmp_path: Path) -> None:
    photo_dir = tmp_path / "photos"
    mocker.patch("edgeguard.actions._camera_index", return_value=0)
    mock_cap = mocker.MagicMock()
    mock_cap.read.return_value = (True, "fake-frame")
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    mocker.patch("edgeguard.actions.time.sleep")
    mock_imwrite = mocker.patch("cv2.imwrite")

    result = take_photo(photo_dir=photo_dir)

    assert result.startswith("Photo saved to ")
    assert photo_dir.exists()
    saved_path = Path(result.removeprefix("Photo saved to ").removesuffix("."))
    assert saved_path.parent == photo_dir
    mock_imwrite.assert_called_once_with(str(saved_path), "fake-frame")
    mock_cap.release.assert_called_once()
