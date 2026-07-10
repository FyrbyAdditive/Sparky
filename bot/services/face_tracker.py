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
VAD-driven, or the mic array's own speech flag), faces whose mouth
region shows frame-to-frame motion score far higher in target selection,
so the head locks onto the talker rather than the largest face. The
head's ReSpeaker XVF3800 also computes 4-mic direction-of-arrival
on-chip; it is read out-of-band over USB vendor control (the audio
stream itself is downmixed mono, verified experimentally), and while
speech is detected it biases selection toward the face nearest that
bearing — or pulls gaze toward an off-camera voice when no face is
visible.

The camera and mic array ride in the moving head, so every measurement
is anchored to its own instant: tracking offsets are snapshotted at
frame-grab / DOA-read time and targets are formed as increments on that
snapshot. Head motion between measurement and correction (sway, texture,
the tracker's own slew) therefore never reads as target motion — the
sway keeps oscillating, but centered on the face.

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

# Visual-servo tuning. The correction is an increment on the offsets
# snapshotted at frame-grab time, so near-unity gain is stable; the
# remaining margin damps detection noise.
_GAIN = 0.8
_DEADBAND = 0.05        # normalized image error below which we hold still
_MAX_PITCH_RAD = 0.26   # ~15°; yaw limit is the max_yaw_deg panel param
_PURSUE_RATE = 0.9      # rad/s slew while following a face
_RECENTER_RATE = 0.25   # rad/s slew back to neutral (gentle across wide yaw)
_HOLD_SECS = 1.5        # keep last offsets this long after losing all faces
_TRACK_TTL = 1.0        # seconds before an unmatched track is dropped

# Body comfort seed: the daemon keeps the camera on the commanded world
# pose regardless of body_yaw (automatic body yaw), so this seed only
# re-postures the base under the gaze — it physically cannot disturb
# tracking, it just unwinds the neck by bringing the body around.
_BODY_SEED_DEADBAND = 0.26   # rad (~15°) of gaze yaw before the base follows
_BODY_SEED_RATE = 0.25       # rad/s chase toward the gaze yaw
_BODY_SEED_HOME_RATE = 0.15  # rad/s ease back when the gaze recenters
_BODY_SEED_MAX = 2.4         # rad; the solver itself caps body at 160°

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
        # body comfort seed (poll-thread owned) + mode set by the worker:
        # "track" chases the gaze yaw, "hold" freezes during animations,
        # "home" eases back to zero when tracking is off
        self._body_seed = 0.0
        self._seed_mode = "home"

        self._detector = None
        self._detector_kind = "none"
        self._detector_size: tuple[int, int] | None = None

        # track id -> {center, size, mouth_prev, mouth_ema, last_seen, landmarks}
        self._tracks: dict[int, dict] = {}
        self._next_track_id = 1
        self._target_track: int | None = None
        self._last_face_seen = 0.0

        # mic-array direction of arrival: read from the ReSpeaker over USB
        # vendor control each cycle (or fed externally via set_doa)
        self._doa = {"az_deg": 0.0, "conf": 0.0, "ts": 0.0}
        self._respeaker = None
        self._doa_ok = True
        self._doa_failures = 0
        self._chip_speech_ts = 0.0

        # telemetry for /status.tracking
        self._telemetry: dict = {"faces": 0, "target": None, "suppressed": "off"}
        # per-cycle overlay snapshot for the panel stream (atomic swaps)
        self._detections: dict | None = None

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

        # body comfort seed integrator (same thread, same dt): chase the
        # gaze yaw so the base comes around and the neck unwinds
        mode = self._seed_mode
        seed = self._body_seed
        if mode == "track" and abs(cur[5]) > _BODY_SEED_DEADBAND:
            goal, rate_s = cur[5], _BODY_SEED_RATE
        elif mode == "hold":
            goal, rate_s = seed, 0.0
        else:  # "home", or gaze back near center
            goal, rate_s = 0.0, _BODY_SEED_HOME_RATE
        step_s = rate_s * dt
        delta_s = goal - seed
        if delta_s > step_s:
            seed += step_s
        elif delta_s < -step_s:
            seed -= step_s
        else:
            seed = goal
        self._body_seed = max(-_BODY_SEED_MAX, min(_BODY_SEED_MAX, seed))

        return (cur[0], cur[1], cur[2], cur[3], cur[4], cur[5])

    def get_body_yaw_seed(self) -> float:
        """Body-yaw solver seed (radians), polled by MovementManager right
        after get_face_tracking_offsets on the same thread."""
        return self._body_seed

    # ------------------------------------------------------------- doa hook

    def set_doa(self, az_deg: float, conf: float) -> None:
        """Mic-array direction of arrival, degrees, head frame (left > 0)."""
        self._doa = {"az_deg": float(az_deg), "conf": float(conf),
                     "ts": time.monotonic()}

    def _init_doa(self) -> None:
        """Open the ReSpeaker XVF3800 control channel (once, best-effort)."""
        if self._respeaker is not None or not self._doa_ok:
            return
        try:
            from reachy_mini.media.audio_control_utils import init_respeaker_usb

            self._respeaker = init_respeaker_usb()
            if self._respeaker is None:
                raise RuntimeError("device not found")
            fw = self._respeaker.read("VERSION")
            logger.info(f"Face tracker: ReSpeaker DOA available (fw {fw})")
        except Exception as e:
            self._doa_ok = False
            logger.info(f"Face tracker: mic-array DOA unavailable ({e}); "
                        "vision-only speaker association")

    def _read_doa(self, now: float) -> None:
        """Poll on-chip DOA. SDK convention: 0 rad = left, π/2 = front,
        π = right — mapped to head-frame azimuth with left positive.
        Skipped while the robot itself is talking (its speaker sits under
        the mics; AEC should cancel it, but don't steer on the residue)."""
        from .local_audio import GATE

        if self._respeaker is None or GATE["bot_speaking"]:
            return
        try:
            result = self._respeaker.read("DOA_VALUE_RADIANS")
        except Exception:
            self._doa_failures += 1
            if self._doa_failures >= 5:
                logger.warning("Face tracker: DOA reads failing; disabling")
                self._respeaker = None
                self._doa_ok = False
            return
        self._doa_failures = 0
        if result is None:
            return
        az_deg = 90.0 - math.degrees(float(result[0]))
        if bool(result[1]):
            self._chip_speech_ts = now
            self.set_doa(az_deg, 1.0)

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

    @staticmethod
    def _face_bearing_deg(face, img_w: int) -> float:
        """Face bearing relative to the camera axis (left positive).

        The mic array and camera share the head, so this compares
        directly against the DOA azimuth — both are head-relative at
        (nearly) the same instant; no pose math needed.
        """
        ex = (face["center"][0] - img_w / 2.0) / (img_w / 2.0)
        # image-right is robot-right, i.e. negative yaw
        return -ex * (_H_FOV_DEG / 2.0)

    def _select_target(self, faces: list[dict], img_w: int, now: float,
                       user_speaking: bool) -> dict | None:
        doa = self._doa
        doa_live = doa["conf"] > 0.2 and now - doa["ts"] < _DOA_FRESH_SECS
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
                err = abs(self._face_bearing_deg(face, img_w) - doa["az_deg"])
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
                    # fully dormant: no grabs, offsets and body seed ramp home
                    self._seed_mode = "home"
                    self._target = ((0.0,) * 6, _RECENTER_RATE)
                    self._telemetry = {"faces": 0, "target": None,
                                       "suppressed": suppressed,
                                       "detector": self._detector_kind}
                    self._detections = {"ts": time.monotonic(), "frame_w": 0,
                                        "frame_h": 0, "faces": [], "doa": None,
                                        "suppressed": suppressed,
                                        "detector": self._detector_kind}
                    if self._stop_event.wait(1.0):
                        break
                    continue

                self._init_doa()

                # Snapshot the tracking offsets NOW: the head keeps moving
                # (sway, texture, our own slew) between this grab and the
                # correction below, and the image reflects this instant.
                # Targets are increments on this snapshot, so self-motion
                # during detection latency never reads as face motion.
                yaw_at_grab = self._current[5]
                pitch_at_grab = self._current[4]

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
                self._read_doa(now)
                self._update_tracks(faces, gray, now)

                user_speaking = bool(GATE["user_speaking"]) \
                    or now - self._chip_speech_ts < 0.5
                img_w = frame.shape[1]
                img_h = frame.shape[0]
                target_face = self._select_target(faces, img_w, now, user_speaking) \
                    if faces else None

                max_yaw = math.radians(tool_param("face_tracking", "max_yaw_deg"))
                if suppressed == "animation":
                    # HOLD the current gaze while a clip plays (a nod should
                    # happen facing the person, not swing home and back);
                    # tracks stay warm for the resume. Stale read of the
                    # poll-owned list is fine — it converges instantly.
                    self._seed_mode = "hold"
                    self._target = (tuple(self._current), _PURSUE_RATE)
                elif target_face is not None:
                    self._seed_mode = "track"
                    self._last_face_seen = now
                    self._target_track = target_face["track_id"]
                    ex = (target_face["center"][0] - img_w / 2.0) / (img_w / 2.0)
                    ey = (target_face["center"][1] - img_h / 2.0) / (img_h / 2.0)
                    half_fov = math.radians(_H_FOV_DEG / 2.0)
                    yaw = yaw_at_grab
                    pitch = pitch_at_grab
                    if abs(ex) > _DEADBAND:
                        yaw += _GAIN * (-ex * half_fov)   # image-right = -yaw
                    if abs(ey) > _DEADBAND:
                        pitch += _GAIN * (ey * half_fov * (img_h / img_w))
                    yaw = max(-max_yaw, min(max_yaw, yaw))
                    pitch = max(-_MAX_PITCH_RAD, min(_MAX_PITCH_RAD, pitch))
                    self._target = ((0.0, 0.0, 0.0, 0.0, pitch, yaw), _PURSUE_RATE)
                else:
                    self._seed_mode = "track"
                    doa = self._doa
                    if doa["conf"] > 0.2 and now - doa["ts"] < _DOA_FRESH_SECS:
                        # no face visible: turn toward the voice. DOA is
                        # head-relative at read time, so it too is an
                        # increment on the grab-time snapshot.
                        yaw = yaw_at_grab + math.radians(doa["az_deg"])
                        yaw = max(-max_yaw, min(max_yaw, yaw))
                        self._target = ((0.0, 0.0, 0.0, 0.0,
                                         pitch_at_grab, yaw), _PURSUE_RATE)
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
                        "bearing_deg": round(
                            self._face_bearing_deg(target_face, img_w), 1),
                        "mouth_ema": round(
                            self._tracks[target_face["track_id"]]["mouth_ema"], 1),
                    },
                    "offsets_deg": {"yaw": round(math.degrees(self._current[5]), 1),
                                    "pitch": round(math.degrees(self._current[4]), 1)},
                    "body_seed_deg": round(math.degrees(self._body_seed), 1),
                }

                # overlay snapshot for the panel stream: fresh structures
                # only (never expose _tracks — it mutates in place), swapped
                # atomically like _telemetry
                doa = self._doa
                doa_fresh = (doa["conf"] > 0.2
                             and now - doa["ts"] < _DOA_FRESH_SECS)
                self._detections = {
                    "ts": now,
                    "frame_w": img_w,
                    "frame_h": img_h,
                    "faces": [{
                        "box": face["box"],
                        "mouth": face["mouth"],
                        "track_id": face["track_id"],
                        "is_target": target_face is not None
                                     and face["track_id"] == target_face["track_id"],
                        "mouth_ema": self._tracks[face["track_id"]]["mouth_ema"],
                    } for face in faces],
                    "doa": ({"az_deg": doa["az_deg"], "conf": doa["conf"]}
                            if doa_fresh else None),
                    "suppressed": suppressed,
                    "detector": self._detector_kind,
                    "body_seed_deg": round(math.degrees(self._body_seed), 1),
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

    def detections(self) -> dict | None:
        """Latest per-cycle detection snapshot (panel overlay). May be None
        or stale — consumers check the ts field."""
        return self._detections

    def status(self) -> dict:
        doa = self._doa
        out = dict(self._telemetry)
        out["doa"] = ({"az_deg": round(doa["az_deg"], 1), "conf": round(doa["conf"], 2)}
                      if time.monotonic() - doa["ts"] < _DOA_FRESH_SECS else None)
        out["doa_available"] = self._respeaker is not None
        try:
            mm = self.service.motion_manager
            if mm is not None:
                roll, pitch, yaw = mm.get_commanded_head_ypr()
                out["head_deg"] = {"yaw": round(math.degrees(yaw), 1),
                                   "pitch": round(math.degrees(pitch), 1)}
        except Exception:
            pass
        try:
            # actual base rotation: joint[0] of the head chain (radians)
            robot = self.service.robot
            if robot is not None:
                joints, _ = robot.get_current_joint_positions()
                out["body_deg"] = round(math.degrees(float(joints[0])), 1)
        except Exception:
            pass
        return out
