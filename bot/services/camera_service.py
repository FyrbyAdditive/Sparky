"""Single owner of the robot camera.

Two consumers need the one physical camera: the vision path (one-shot RGB
grab per image request) and the panel's live MJPEG stream (continuous ~8fps
JPEG). Two cv2.VideoCapture handles cannot share a device, so this module
owns the capture behind one lock and both consumers call in.

Capture resolution is a runtime setting (panel Camera section -> POST
/camera). Drivers (AVFoundation especially) often ignore the requested
mode and deliver native frames, so frames are downscaled after read —
that also keeps the convert/encode cost at the requested size.
"""

import os
import threading

from loguru import logger

CAMERA_RESOLUTIONS = ["320x240", "640x480", "1280x720", "1920x1080"]

# Shared with robot_api (same cross-thread pattern as the speaker registry):
# the panel writes, the capture path reads and reopens on change.
CAMERA = {"resolution": os.getenv("CAMERA_RESOLUTION", "640x480")}

_lock = threading.Lock()
_capture = None
_applied_resolution: str | None = None
_device_index = int(os.getenv("ROBOT_CAMERA_INDEX", "0"))


def _parse_resolution(value: str) -> tuple[int, int] | None:
    try:
        w, h = value.lower().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return None


def _open_locked() -> bool:
    """Open (or reopen after a resolution change) the capture. Lock held."""
    import cv2

    global _capture, _applied_resolution
    wanted = CAMERA["resolution"]
    if _capture is not None and _capture.isOpened():
        if _applied_resolution == wanted:
            return True
        _capture.release()
        _capture = None
    _capture = cv2.VideoCapture(_device_index)
    if not _capture.isOpened():
        logger.warning(f"camera_service: cannot open camera {_device_index}")
        _capture = None
        return False
    parsed = _parse_resolution(wanted)
    if parsed:
        _capture.set(cv2.CAP_PROP_FRAME_WIDTH, parsed[0])
        _capture.set(cv2.CAP_PROP_FRAME_HEIGHT, parsed[1])
    _applied_resolution = wanted
    return True


def _read_frame_locked():
    """Read one raw BGR frame, or None. Lock held — keep this minimal:
    only the VideoCapture access itself; downscale happens outside."""
    global _capture
    if not _open_locked():
        return None
    ok, frame_bgr = _capture.read()
    if not ok or frame_bgr is None:
        logger.warning("camera_service: capture failed, reopening next time")
        _capture.release()
        _capture = None
        return None
    return frame_bgr


def _downscale(frame_bgr):
    """Downscale to the panel-selected resolution (AVFoundation ignores
    cv2 capture-size requests, so this happens after read). Lock-free."""
    import cv2

    wanted = _parse_resolution(CAMERA["resolution"])
    h, w = frame_bgr.shape[:2]
    if wanted and (w, h) != wanted and w > wanted[0]:
        frame_bgr = cv2.resize(frame_bgr, wanted, interpolation=cv2.INTER_AREA)
    return frame_bgr


def grab_bgr():
    """One downscaled BGR numpy frame (fresh — stale frames drained), or None.

    Only the VideoCapture handle needs mutual exclusion — pixel work
    (convert/copy) happens outside the lock so the panel's MJPEG stream
    isn't stalled behind a vision or tracker grab (and vice versa).
    """
    with _lock:
        # Drain a couple of stale frames so the answer reflects "now"
        if _open_locked():
            for _ in range(2):
                _capture.grab()
        frame_bgr = _read_frame_locked()
    if frame_bgr is None:
        return None
    return _downscale(frame_bgr)


def grab_rgb():
    """One-shot RGB grab for the vision path: (bytes, (w, h)) or None."""
    import cv2

    frame_bgr = grab_bgr()
    if frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    return frame_rgb.tobytes(), (w, h)


def grab_jpeg(quality: int = 70):
    """One JPEG frame for the MJPEG stream, or None. Encodes straight from
    BGR (no RGB round-trip); JPEG encode runs outside the capture lock."""
    import cv2

    with _lock:
        frame_bgr = _read_frame_locked()
    if frame_bgr is None:
        return None
    frame_bgr = _downscale(frame_bgr)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
