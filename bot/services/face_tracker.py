"""Continuous face/speaker tracking: the camera_worker for MovementManager.

MovementManager has always polled an optional camera_worker for
face-tracking offsets at 100Hz (moves.py step 3) — this module is the
first real implementation of that contract. A worker thread grabs frames
through camera_service (single shared VideoCapture), detects faces with
YuNet (cv2.FaceDetectorYN; Haar cascade fallback when the small ONNX
model is unavailable), picks a target, and steers yaw/pitch offsets so
the head follows the person. The camera rides in the head, so this is a
visual servo: each detection nudges the offset by a fraction of the
remaining angular error rather than commanding an absolute pose.

Speaker association: while the user is speaking (GATE["user_speaking"],
VAD-driven), faces whose mouth region shows frame-to-frame motion score
far higher in target selection, so the head locks onto the talker rather
than the largest face. A sound-direction hook (set_doa) lets the mic
array bias selection the same way and pull gaze toward an off-camera
voice.

Safety/arbitration: offsets ramp to zero (never snap) whenever the
AnimationDirector owns the stage, the e-stop latch is set, the robot is
disconnected, or the panel toggle is off. The tracker's own clamps stay
inside the SECONDARY_MAX_* envelope in moves.py, which saturates the
summed secondary offsets because the daemon rejects — not clamps —
unreachable poses.

Threading: the detector thread writes a (target_offsets, rate) snapshot;
get_face_tracking_offsets() runs on the move worker (its only caller)
and slew-limits toward the target per poll, so motion stays smooth at
100Hz regardless of detection fps. Tuple swaps are atomic under the GIL;
no locks on the hot path.
"""

import logging
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import camera_service

logger = logging.getLogger(__name__)

_MODEL_DIR = Path.home() / ".sparky" / "models"
_YUNET_FILE = _MODEL_DIR / "face_detection_yunet_2023mar.onnx"
_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")

# Detection runs on a fixed-width copy regardless of panel camera
# resolution: YuNet is fast and accurate at this scale and the mouth ROI
# diff only needs coarse pixels.
_DETECT_WIDTH = 320
_SCORE_THRESHOLD = 0.6

# Camera horizontal field of view (degrees). The Reachy Mini head camera
# is a wide-angle module; this only scales the servo gain, so a rough
# value is fine and env-tunable.
_H_FOV_DEG = float(os.getenv("FACE_TRACK_HFOV_DEG", "68"))

# Visual-servo tuning
_GAIN = 0.35            # fraction of remaining angular error applied per detection
_DEADBAND = 0.05        # normalized image error below which we hold still
_MAX_PITCH_RAD = 0.26   # ~15°; yaw limit is the max_yaw_deg panel param
_PURSUE_RATE = 0.9      # rad/s slew while following a face
_RECENTER_RATE = 0.35   # rad/s slew back to neutral (face lost / suppressed)
_HOLD_SECS = 1.5        # keep last offsets this long after losing all faces
_TRACK_TTL = 1.0        # seconds before an unmatched track is dropped

# Target-selection weights (fusion lives here; DOA joins in via set_doa)
_W_SIZE = 1.0
_W_CENTER = 0.4
_W_STICKY = 0.5         # hysteresis bonus for the current target
_W_MOUTH = 2.0          # mouth activity while the user is speaking
_W_DOA = 1.5            # agreement with mic-array direction while speaking
_MOUTH_NORM = 12.0      # mean-abs-diff (0-255 gray) treated as "clearly talking"
_DOA_FRESH_SECS = 1.0


class FaceTracker:
    """Camera worker: continuous detection thread + 100Hz offset poll."""

    def __init__(self, service):
        self.service = service
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # written by detector thread, read by move worker (atomic swaps)
        self._target: tuple[tuple[float, ...], float] = ((0.0,) * 6, _RECENTER_RATE)
        # owned by the move worker (single caller of the poll)
        self._current = [0.0] * 6
        self._last_poll = 0.0

        self._detector = None
        self._detector_kind = "none"
        self._detector_size: tuple[int, int] | None = None

        # track id -> {center, size, mouth_prev, mouth_ema, last_seen, landmarks}
        self._tracks: dict[int, dict] = {}
        self._next_track_id = 1
        self._target_track: int | None = None
        self._last_face_seen = 0.0

        # mic-array direction of arrival: set_doa() from the audio side
        self._doa = {"az_deg": 0.0, "conf": 0.0, "ts": 0.0}

        # telemetry for /status.tracking
        self._telemetry: dict = {"faces": 0, "target": None, "suppressed": "off"}

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="face-tracker")
        self._thread.start()
        logger.info("Face tracker started")

    def stop(self) -> None:
        self._stop_event.set()

    # --------------------------------------------------- move-worker contract

    def get_face_tracking_offsets(self) -> tuple[float, float, float, float, float, float]:
        """Poll contract from MovementManager (100Hz, move worker thread).

        Pure arithmetic — advances the applied offsets toward the detector
        thread's latest target under a slew-rate limit so 8Hz detections
        never turn into visible steps.
        """
        target, rate = self._target
        now = time.monotonic()
        dt = min(0.05, now - self._last_poll) if self._last_poll else 0.01
        self._last_poll = now
        step = rate * dt
        cur = self._current
        for i in range(6):
            delta = target[i] - cur[i]
            if delta > step:
                cur[i] += step
            elif delta < -step:
                cur[i] -= step
            else:
                cur[i] = target[i]
        return (cur[0], cur[1], cur[2], cur[3], cur[4], cur[5])

    # ------------------------------------------------------------- doa hook

    def set_doa(self, az_deg: float, conf: float) -> None:
        """Mic-array direction of arrival, degrees, head frame (left > 0)."""
        self._doa = {"az_deg": float(az_deg), "conf": float(conf),
                     "ts": time.monotonic()}

    # ------------------------------------------------------------- detector

    def _ensure_detector(self, frame_shape) -> None:
        import cv2

        h, w = frame_shape[:2]
        if self._detector is not None:
            if self._detector_kind == "yunet" and self._detector_size != (w, h):
                self._detector.setInputSize((w, h))
                self._detector_size = (w, h)
            return

        if not _YUNET_FILE.exists():
            self._download_model()
        if _YUNET_FILE.exists():
            try:
                self._detector = cv2.FaceDetectorYN.create(
                    str(_YUNET_FILE), "", (w, h), _SCORE_THRESHOLD)
                self._detector_kind = "yunet"
                self._detector_size = (w, h)
                logger.info("Face tracker using YuNet detector")
                return
            except Exception as e:
                logger.warning(f"YuNet init failed ({e}); falling back to Haar")
        cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(str(cascade))
        self._detector_kind = "haar"
        logger.info("Face tracker using Haar cascade detector (no landmarks)")

    def _download_model(self) -> None:
        """One-time YuNet model fetch (~230kB, MIT-licensed, opencv_zoo).
        Deploys can pre-place the file; on failure Haar carries tracking."""
        try:
            import urllib.request

            _MODEL_DIR.mkdir(parents=True, exist_ok=True)
            tmp = _YUNET_FILE.with_suffix(".onnx.tmp")
            with urllib.request.urlopen(_YUNET_URL, timeout=15) as r:
                tmp.write_bytes(r.read())
            os.replace(tmp, _YUNET_FILE)
            logger.info(f"Face tracker downloaded YuNet model to {_YUNET_FILE}")
        except Exception as e:
            logger.warning(f"YuNet model download failed ({e}); "
                           f"pre-place it at {_YUNET_FILE} or Haar will be used")

    def _detect(self, gray_or_bgr) -> list[dict]:
        """Run the detector; normalize to [{box, center, landmarks|None}]."""
        import cv2

        faces = []
        if self._detector_kind == "yunet":
            _, dets = self._detector.detect(gray_or_bgr)
            if dets is not None:
                for row in dets:
                    x, y, w, h = row[:4]
                    faces.append({
                        "box": (float(x), float(y), float(w), float(h)),
                        "center": (float(x + w / 2), float(y + h / 2)),
                        # right mouth corner, left mouth corner
                        "mouth": ((float(row[10]), float(row[11])),
                                  (float(row[12]), float(row[13]))),
                    })
        else:
            gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in self._detector.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=4,
                    minSize=(24, 24)):
                faces.append({
                    "box": (float(x), float(y), float(w), float(h)),
                    "center": (float(x + w / 2), float(y + h / 2)),
                    "mouth": None,  # approximated from the box in _mouth_roi
                })
        return faces

    # ---------------------------------------------------- association cues

    @staticmethod
    def _mouth_roi(gray, face) -> np.ndarray | None:
        """Small fixed-size grayscale crop around the mouth for diff energy."""
        import cv2

        h_img, w_img = gray.shape[:2]
        if face["mouth"] is not None:
            (rx, ry), (lx, ly) = face["mouth"]
            cx, cy = (rx + lx) / 2.0, (ry + ly) / 2.0
            half_w = max(6.0, abs(lx - rx) * 0.8)
            half_h = max(4.0, half_w * 0.55)
        else:
            x, y, w, h = face["box"]
            cx, cy = x + w / 2.0, y + h * 0.78
            half_w, half_h = w * 0.28, h * 0.14
        x0, x1 = int(cx - half_w), int(cx + half_w)
        y0, y1 = int(cy - half_h), int(cy + half_h)
        if x0 < 0 or y0 < 0 or x1 > w_img or y1 > h_img or x1 - x0 < 4 or y1 - y0 < 3:
            return None
        roi = gray[y0:y1, x0:x1]
        return cv2.resize(roi, (24, 14), interpolation=cv2.INTER_AREA).astype(np.int16)

    def _update_tracks(self, faces: list[dict], gray, now: float) -> None:
        """Greedy nearest-center matching; per-track mouth-motion EMA."""
        img_w = gray.shape[1]
        unmatched = dict(self._tracks)
        for face in faces:
            cx, cy = face["center"]
            best_id, best_d = None, 0.25 * img_w
            for tid, tr in unmatched.items():
                d = math.hypot(cx - tr["center"][0], cy - tr["center"][1])
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_id] = {"mouth_prev": None, "mouth_ema": 0.0}
            else:
                unmatched.pop(best_id)
            tr = self._tracks[best_id]
            tr["center"] = face["center"]
            tr["box"] = face["box"]
            tr["last_seen"] = now
            face["track_id"] = best_id

            roi = self._mouth_roi(gray, face)
            if roi is not None and tr["mouth_prev"] is not None \
                    and roi.shape == tr["mouth_prev"].shape:
                diff = float(np.mean(np.abs(roi - tr["mouth_prev"])))
                tr["mouth_ema"] = 0.6 * tr["mouth_ema"] + 0.4 * diff
            tr["mouth_prev"] = roi

        for tid, tr in list(self._tracks.items()):
            if now - tr.get("last_seen", 0.0) > _TRACK_TTL:
                del self._tracks[tid]

    def _face_azimuth_deg(self, face, img_w: int) -> float:
        """Face bearing in the head frame (left positive), current offsets in."""
        ex = (face["center"][0] - img_w / 2.0) / (img_w / 2.0)
        # image-right is robot-right, i.e. negative yaw
        cam_az = -ex * (_H_FOV_DEG / 2.0)
        return math.degrees(self._current[5]) + cam_az

    def _select_target(self, faces: list[dict], img_w: int, now: float,
                       user_speaking: bool) -> dict | None:
        doa = self._doa
        doa_live = (user_speaking and doa["conf"] > 0.2
                    and now - doa["ts"] < _DOA_FRESH_SECS)
        best, best_score = None, -1.0
        for face in faces:
            x, y, w, h = face["box"]
            ex = (face["center"][0] - img_w / 2.0) / (img_w / 2.0)
            score = _W_SIZE * (w / img_w) + _W_CENTER * (1.0 - abs(ex))
            if face["track_id"] == self._target_track:
                score += _W_STICKY
            if user_speaking:
                ema = self._tracks[face["track_id"]]["mouth_ema"]
                score += _W_MOUTH * min(1.0, ema / _MOUTH_NORM)
            if doa_live:
                err = abs(self._face_azimuth_deg(face, img_w) - doa["az_deg"])
                score += _W_DOA * doa["conf"] * max(0.0, 1.0 - err / 30.0)
            if score > best_score:
                best, best_score = face, score
        return best

    # ------------------------------------------------------------ main loop

    def _suppression_reason(self) -> str | None:
        from .robot_api import ESTOP, OPTIONAL_TOOLS

        if ESTOP["active"]:
            return "estop"
        if not OPTIONAL_TOOLS["face_tracking"]["enabled"]:
            return "off"
        if not self.service.connected:
            return "disconnected"
        from .animation_director import get_director

        if get_director().is_busy():
            return "animation"
        return None

    def _worker(self) -> None:
        import cv2

        from .local_audio import GATE
        from .robot_api import tool_param

        fps = 8.0
        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            try:
                fps = max(1.0, tool_param("face_tracking", "fps"))
                suppressed = self._suppression_reason()
                if suppressed in ("off", "disconnected", "estop"):
                    # fully dormant: no grabs, offsets ramp home
                    self._target = ((0.0,) * 6, _RECENTER_RATE)
                    self._telemetry = {"faces": 0, "target": None,
                                       "suppressed": suppressed,
                                       "detector": self._detector_kind}
                    if self._stop_event.wait(1.0):
                        break
                    continue

                frame = camera_service.grab_bgr()
                if frame is None:
                    self._target = ((0.0,) * 6, _RECENTER_RATE)
                    self._telemetry = {"faces": 0, "target": None,
                                       "suppressed": "no_camera",
                                       "detector": self._detector_kind}
                    if self._stop_event.wait(2.0):
                        break
                    continue

                h, w = frame.shape[:2]
                if w > _DETECT_WIDTH:
                    scale = _DETECT_WIDTH / w
                    frame = cv2.resize(frame, (_DETECT_WIDTH, int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
                self._ensure_detector(frame.shape)
                faces = self._detect(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                now = time.monotonic()
                self._update_tracks(faces, gray, now)

                user_speaking = bool(GATE["user_speaking"])
                img_w = frame.shape[1]
                img_h = frame.shape[0]
                target_face = self._select_target(faces, img_w, now, user_speaking) \
                    if faces else None

                max_yaw = math.radians(tool_param("face_tracking", "max_yaw_deg"))
                if suppressed == "animation":
                    # yield the stage but keep tracks warm for the resume
                    self._target = ((0.0,) * 6, _RECENTER_RATE)
                elif target_face is not None:
                    self._last_face_seen = now
                    self._target_track = target_face["track_id"]
                    ex = (target_face["center"][0] - img_w / 2.0) / (img_w / 2.0)
                    ey = (target_face["center"][1] - img_h / 2.0) / (img_h / 2.0)
                    half_fov = math.radians(_H_FOV_DEG / 2.0)
                    yaw = self._current[5]
                    pitch = self._current[4]
                    if abs(ex) > _DEADBAND:
                        yaw += _GAIN * (-ex * half_fov)   # image-right = -yaw
                    if abs(ey) > _DEADBAND:
                        pitch += _GAIN * (ey * half_fov * (img_h / img_w))
                    yaw = max(-max_yaw, min(max_yaw, yaw))
                    pitch = max(-_MAX_PITCH_RAD, min(_MAX_PITCH_RAD, pitch))
                    self._target = ((0.0, 0.0, 0.0, 0.0, pitch, yaw), _PURSUE_RATE)
                else:
                    doa = self._doa
                    if (user_speaking and doa["conf"] > 0.2
                            and now - doa["ts"] < _DOA_FRESH_SECS):
                        # no face visible: turn toward the voice
                        yaw = max(-max_yaw, min(max_yaw, math.radians(doa["az_deg"])))
                        self._target = ((0.0, 0.0, 0.0, 0.0,
                                         self._current[4], yaw), _PURSUE_RATE)
                    elif now - self._last_face_seen > _HOLD_SECS:
                        self._target_track = None
                        self._target = ((0.0,) * 6, _RECENTER_RATE)
                    # else: hold the current target through a brief dropout

                self._telemetry = {
                    "faces": len(faces),
                    "detector": self._detector_kind,
                    "suppressed": suppressed,
                    "user_speaking": user_speaking,
                    "target": None if target_face is None else {
                        "track": target_face["track_id"],
                        "mouth_ema": round(
                            self._tracks[target_face["track_id"]]["mouth_ema"], 1),
                    },
                    "offsets_deg": {"yaw": round(math.degrees(self._current[5]), 1),
                                    "pitch": round(math.degrees(self._current[4]), 1)},
                }
            except Exception as e:
                logger.warning(f"face tracker cycle failed: {e}")
                self._target = ((0.0,) * 6, _RECENTER_RATE)

            elapsed = time.monotonic() - cycle_start
            if self._stop_event.wait(max(0.0, 1.0 / fps - elapsed)):
                break
        # leave nothing behind on shutdown
        self._target = ((0.0,) * 6, _RECENTER_RATE)
        logger.info("Face tracker stopped")

    # ------------------------------------------------------------ telemetry

    def status(self) -> dict:
        doa = self._doa
        out = dict(self._telemetry)
        out["doa"] = ({"az_deg": round(doa["az_deg"], 1), "conf": round(doa["conf"], 2)}
                      if time.monotonic() - doa["ts"] < _DOA_FRESH_SECS else None)
        return out
