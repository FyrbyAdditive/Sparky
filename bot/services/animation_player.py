"""Playback of authored animation clips through the movement queue.

Clip assets live in bot/animations/<name>/<name>.json — Blender-authored
keyframes from NVIDIA spark-reachy-photo-booth (Apache-2.0). Format:

    {"frame_rate": 24, "data": {
        "head_rotation":   {"joints": ["neck_roll","neck_pitch","neck_yaw"], "frames": [[r,p,y], ...]},
        "head_position":   {"joints": ["head_x","head_y","head_z"],          "frames": [[x,y,z], ...]},
        "r_antenna_angle": {"joints": "r_antenna_angle",                     "frames": [deg, ...]},
        "l_antenna_angle": {"joints": "l_antenna_angle",                     "frames": [deg, ...]},
        "body_angle":      {"joints": "body_angle",                          "frames": [deg, ...]}}}

Unit conventions match the photo-booth robot controller: rotations and
antenna/body angles in degrees, head positions in file-units * 10 = mm.
The Move interface expects antennas and body_yaw in radians and a 4x4 head
pose, so conversion happens in evaluate().
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reachy_mini.motion.move import Move
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation

logger = logging.getLogger(__name__)

# file-units -> mm, matching photo-booth's remap_head_translation scale=(10,10,10)
POSITION_SCALE_MM = 10.0
BLEND_IN_SECS = 0.25

# Clips whose direction is semantically meaningful and must never be
# auto-mirrored (names containing left/right are excluded automatically).
MIRROR_EXCLUDE: set[str] = set()

# The dance-library imports (plus the emotions-library's dance1-3) — used
# to categorize the panel's animation browser. Everything else with a
# description came from the Pollen emotions library; the remainder are the
# original photo-booth gesture/state clips.
DANCE_NAMES = {
    "dance1", "dance2", "dance3",
    "chicken_peck", "chin_lead", "dizzy_spin", "grid_snap",
    "groovy_sway_and_roll", "head_tilt_roll", "interwoven_spirals",
    "jackson_square", "neck_recoil", "pendulum_swing", "polyrhythm_combo",
    "sharp_side_tilt", "side_glance_flick", "side_peekaboo",
    "side_to_side_sway", "simple_nod", "stumble_and_recover",
    "uh_huh_tilt", "yeah_nod",
}

DEFAULT_ANIMATIONS_DIR = Path(__file__).resolve().parent.parent / "animations"


class AnimationClip:
    """A parsed animation clip with per-frame channel access."""

    def __init__(self, name: str, frame_rate: float, data: dict,
                 description: str = ""):
        self.name = name
        self.frame_rate = float(frame_rate)
        self.description = description
        # panel browser grouping: dances / emotions (Pollen, carry a
        # description) / classics (the original photo-booth set)
        self.category = ("dances" if name in DANCE_NAMES
                         else "emotions" if description else "classics")

        def scalar_channel(key) -> np.ndarray | None:
            entry = data.get(key)
            if not entry:
                return None
            return np.asarray(entry["frames"], dtype=np.float64)

        def vector_channel(key, joint_order) -> np.ndarray | None:
            """Map a channel's named joints onto fixed (N, 3) columns.

            Clips may animate a subset of axes (e.g. head_position with only
            ["head_y", "head_z"]); missing axes stay 0.
            """
            entry = data.get(key)
            if not entry:
                return None
            frames = np.atleast_2d(np.asarray(entry["frames"], dtype=np.float64))
            joints = entry.get("joints", joint_order)
            if isinstance(joints, str):
                joints = [joints]
            out = np.zeros((frames.shape[0], len(joint_order)))
            for col, joint in enumerate(joints):
                if joint in joint_order and col < frames.shape[1]:
                    out[:, joint_order.index(joint)] = frames[:, col]
            return out

        self.head_rotation = vector_channel("head_rotation", ["neck_roll", "neck_pitch", "neck_yaw"])  # (N, 3) deg
        self.head_position = vector_channel("head_position", ["head_x", "head_y", "head_z"])           # (N, 3) file units
        self.r_antenna = scalar_channel("r_antenna_angle")  # (N,) deg
        self.l_antenna = scalar_channel("l_antenna_angle")  # (N,) deg
        self.body_angle = scalar_channel("body_angle")      # (N,) deg

        lengths = [len(c) for c in (self.head_rotation, self.head_position,
                                    self.r_antenna, self.l_antenna, self.body_angle)
                   if c is not None]
        if not lengths:
            raise ValueError(f"Animation '{name}' has no supported channels")
        self.num_frames = min(lengths)

    @property
    def duration(self) -> float:
        return self.num_frames / self.frame_rate

    @property
    def mirrorable(self) -> bool:
        """Safe to horizontally mirror: direction isn't part of the clip's
        meaning (talkingLeftShoulder/talkingRightShoulder are the named
        exceptions; MIRROR_EXCLUDE catches future semantic cases)."""
        return (re.search(r"left|right", self.name, re.I) is None
                and self.name not in MIRROR_EXCLUDE)

    def mirrored(self) -> "AnimationClip":
        """Horizontally mirrored view of this clip (cached).

        Mirror across the sagittal plane: negate neck_roll, neck_yaw,
        lateral head_y and body_angle; swap the antenna tracks (the sides
        use opposite sign conventions — nod holds r=-20/l=+20 — so a plain
        swap is the correct mirror); neck_pitch, head_x, head_z unchanged.
        """
        cached = getattr(self, "_mirrored", None)
        if cached is not None:
            return cached
        m = object.__new__(AnimationClip)
        m.name = self.name
        m.frame_rate = self.frame_rate
        m.num_frames = self.num_frames
        m.description = self.description
        m.category = self.category
        if self.head_rotation is not None:
            m.head_rotation = self.head_rotation * np.array([-1.0, 1.0, -1.0])
        else:
            m.head_rotation = None
        if self.head_position is not None:
            m.head_position = self.head_position * np.array([1.0, -1.0, 1.0])
        else:
            m.head_position = None
        m.r_antenna = None if self.l_antenna is None else self.l_antenna.copy()
        m.l_antenna = None if self.r_antenna is None else self.r_antenna.copy()
        m.body_angle = None if self.body_angle is None else -self.body_angle
        m._mirrored = self  # mirroring twice returns the original
        self._mirrored = m
        return m

    def _sample(self, channel: np.ndarray | None, frame_t: float, default):
        if channel is None:
            return default
        i = int(frame_t)
        if i >= len(channel) - 1:
            return channel[-1]
        alpha = frame_t - i
        return channel[i] * (1.0 - alpha) + channel[i + 1] * alpha

    def pose_at(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        """Sample the clip at time t (seconds) -> (head_pose_4x4, antennas_rad, body_yaw_rad)."""
        frame_t = max(0.0, t) * self.frame_rate
        rot = self._sample(self.head_rotation, frame_t, np.zeros(3))
        pos = self._sample(self.head_position, frame_t, np.zeros(3)) * POSITION_SCALE_MM
        r_ant = float(self._sample(self.r_antenna, frame_t, 0.0))
        l_ant = float(self._sample(self.l_antenna, frame_t, 0.0))
        body = float(self._sample(self.body_angle, frame_t, 0.0))

        head_pose = create_head_pose(
            x=pos[0], y=pos[1], z=pos[2],
            roll=rot[0], pitch=rot[1], yaw=rot[2],
            degrees=True, mm=True,
        )
        antennas = np.deg2rad([r_ant, l_ant])
        return head_pose, antennas, float(np.deg2rad(body))


class AnimationQueueMove(Move):  # type: ignore
    """Plays an AnimationClip through the MovementManager queue.

    Blends from the robot's current pose into the clip over the first
    BLEND_IN_SECS so queued animations don't snap.
    """

    def __init__(
        self,
        clip: AnimationClip,
        start_head_pose: NDArray[np.float64] | None = None,
        start_antennas: tuple[float, float] | None = None,
    ):
        self.clip = clip
        self.start_head_pose = start_head_pose
        self.start_antennas = np.array(start_antennas if start_antennas is not None else (0.0, 0.0))

    @property
    def duration(self) -> float:
        return self.clip.duration

    def evaluate(self, t: float) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        try:
            head_pose, antennas, body_yaw = self.clip.pose_at(t)

            if self.start_head_pose is not None and t < BLEND_IN_SECS:
                alpha = max(0.0, min(1.0, t / BLEND_IN_SECS))
                head_pose = linear_pose_interpolation(self.start_head_pose, head_pose, alpha)
                antennas = self.start_antennas * (1.0 - alpha) + antennas * alpha

            return (head_pose, np.asarray(antennas, dtype=np.float64), body_yaw)
        except Exception as e:
            logger.error(f"Error evaluating animation '{self.clip.name}' at t={t}: {e}")
            neutral = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return (neutral, np.array([0.0, 0.0], dtype=np.float64), 0.0)


class AnimationLibrary:
    """Loads and caches all clips found in the animations directory."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory or os.getenv("ANIMATIONS_DIR", DEFAULT_ANIMATIONS_DIR))
        self._clips: dict[str, AnimationClip] = {}
        self._load()

    def _load(self):
        if not self.directory.is_dir():
            logger.warning(f"Animations directory not found: {self.directory}")
            return
        for clip_dir in sorted(self.directory.iterdir()):
            json_path = clip_dir / f"{clip_dir.name}.json"
            if not json_path.is_file():
                continue  # e.g. sound-effect-only folders like beep/
            try:
                with open(json_path) as f:
                    raw = json.load(f)
                clip = AnimationClip(clip_dir.name, raw.get("frame_rate", 24), raw.get("data", {}),
                                     description=str(raw.get("description", "")))
                self._clips[clip.name] = clip
            except Exception as e:
                logger.error(f"Failed to load animation '{clip_dir.name}': {e}")
        logger.info(f"Loaded {len(self._clips)} animations: {', '.join(sorted(self._clips))}")

    def names(self) -> list[str]:
        return sorted(self._clips)

    def catalog(self) -> list[dict]:
        """Panel browser metadata: name, category, description, duration."""
        return [{"name": c.name, "category": c.category,
                 "description": c.description,
                 "duration": round(c.duration, 1)}
                for _, c in sorted(self._clips.items())]

    def get(self, name: str) -> AnimationClip | None:
        return self._clips.get(name)
