"""Procedural motion texture: continuous, state-driven ambient head motion.

Replaces the canned speaking/idle clip loops as the robot's base layer of
liveliness (their panel toggles remain for re-enabling; deliberate clips
still play for gestures, emotions and requests). Instead of replaying
authored keyframes, a 50Hz thread synthesizes offsets from per-DOF
Ornstein-Uhlenbeck noise — mean-reverting random drift, so the motion is
smooth, bounded, and never repeats — plus occasional saccade-like gaze
shifts, with the noise character (amplitude, bandwidth, postural bias)
chosen by behavioral state:

  IDLE       slow organic wander + micro-saccades every few seconds
  ATTENTIVE  near-stillness with a slight upward, head-tilted "listening"
             posture while the user speaks or shortly after interaction
  SPEAKING   moderate wander + occasional gaze shifts layered under the
             loudness-driven speech sway (SwayRollRT), so long replies
             drift around the room instead of oscillating about one point

Output goes through MovementManager.set_texture_offsets — the third
secondary source, summed with speech sway and face tracking and saturated
by the SECONDARY_MAX_* clamp in moves.py. It deliberately does not count
as activity, so the idle BreathingMove keeps re-centering underneath.

Safety: per-DOF amplitude clamps stay well inside the global envelope;
every output is slew-rate limited (fast only during a saccade window);
state changes cross-fade generator parameters over ~1.5s so posture never
snaps; when the AnimationDirector owns the stage, on e-stop, disconnect,
or toggle-off, parameters fade to zero and the output ramps home.
"""

import logging
import math
import os
import random
import threading
import time

logger = logging.getLogger(__name__)

_TICK_HZ = 50.0
_DT = 1.0 / _TICK_HZ

# DOF order matches the offsets tuple: x, y, z (m), roll, pitch, yaw (rad)
_DOFS = 6

# Hard per-DOF output clamp (ambient budget, inside SECONDARY_MAX_*)
_CLAMP = (0.004, 0.004, 0.006, math.radians(5), math.radians(6), math.radians(8))

# Slew limits: translation m/s, rotation rad/s; saccades briefly go faster
_SLEW_XYZ = 0.02
_SLEW_ROT = 0.5
_SACCADE_SLEW_ROT = 2.0
_SACCADE_WINDOW_SECS = 0.4

# How long after user speech / interaction the ATTENTIVE posture holds
_ATTENTIVE_HOLD_SECS = 20.0

# Parameter cross-fade time constants
_BLEND_SECS = 1.5
_SUPPRESS_BLEND_SECS = 0.5

# Per-state generator tables.
#   std:   OU steady-state standard deviation per DOF (m / rad)
#   theta: OU mean-reversion rate per DOF (1/s) — higher = quicker jitter
#   saccade: (min_gap_s, max_gap_s, max_yaw_jump_rad) or None
#   pitch_bias / roll_bias_mag: postural offsets (rad)
_ZERO6 = (0.0,) * 6


def _rad(deg: float) -> float:
    return math.radians(deg)


_STATES = {
    "idle": {
        "std": (0.0, 0.0, 0.002, _rad(1.2), _rad(2.5), _rad(4.0)),
        "theta": (0.5, 0.5, 0.3, 0.6, 0.5, 0.4),
        "saccade": (4.0, 12.0, _rad(6.0)),
        "pitch_bias": 0.0,
        "roll_bias_mag": 0.0,
    },
    "attentive": {
        "std": (0.0, 0.0, 0.0008, _rad(0.5), _rad(0.6), _rad(0.8)),
        "theta": (1.0, 1.0, 0.8, 1.2, 1.2, 1.2),
        "saccade": None,
        # the robot is small and looks up at people: a touch of up-pitch
        # (negative = up) plus a per-episode head-tilt reads as attention
        "pitch_bias": -_rad(3.0),
        "roll_bias_mag": _rad(3.5),
    },
    "speaking": {
        "std": (0.0, 0.0, 0.0015, _rad(0.8), _rad(1.5), _rad(2.2)),
        "theta": (0.6, 0.6, 0.4, 0.7, 0.6, 0.5),
        "saccade": (6.0, 15.0, _rad(4.0)),
        "pitch_bias": 0.0,
        "roll_bias_mag": 0.0,
    },
    "off": {
        "std": _ZERO6,
        "theta": (5.0,) * 6,  # fast mean-reversion so suppression settles quickly
        "saccade": None,
        "pitch_bias": 0.0,
        "roll_bias_mag": 0.0,
    },
}


class MotionTexture:
    def __init__(self, service, emit):
        """service: ReachyService (connection checks); emit: set_texture_offsets."""
        self.service = service
        self.emit = emit
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # generator state
        self._ou = [0.0] * _DOFS
        self._out = [0.0] * _DOFS
        # blended parameters (start at "off" so startup ramps in)
        self._std = list(_STATES["off"]["std"])
        self._theta = list(_STATES["off"]["theta"])
        self._pitch_bias = 0.0
        self._roll_bias = 0.0
        self._roll_bias_target = 0.0

        # saccade state
        self._gaze_bias_yaw = 0.0
        self._gaze_bias_pitch = 0.0
        self._next_saccade = 0.0
        self._saccade_until = 0.0

        self._state = "off"
        self._telemetry: dict = {"state": "off"}

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True,
                                        name="motion-texture")
        self._thread.start()
        logger.info("Motion texture started")

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------- states

    def _desired_state(self, now: float) -> str:
        from .animation_director import get_director
        from .local_audio import GATE
        from .robot_api import ESTOP, LAST_INTERACTION, OPTIONAL_TOOLS

        if (ESTOP["active"] or not self.service.connected
                or not OPTIONAL_TOOLS["motion_texture"]["enabled"]
                or get_director().is_busy()):
            return "off"
        if GATE["bot_speaking"] or now < GATE["tail_until"]:
            return "speaking"
        if GATE["user_speaking"]:
            return "attentive"
        recent = max(GATE["user_stopped_ts"], LAST_INTERACTION["ts"])
        if now - recent < _ATTENTIVE_HOLD_SECS:
            return "attentive"
        return "idle"

    def _enter_state(self, state: str, now: float) -> None:
        prev, self._state = self._state, state
        params = _STATES[state]
        if params["saccade"]:
            lo, hi, _ = params["saccade"]
            self._next_saccade = now + random.uniform(lo, hi)
        if state == "attentive" and prev != "attentive":
            self._roll_bias_target = (random.choice((-1.0, 1.0))
                                      * random.uniform(0.5, 1.0)
                                      * params["roll_bias_mag"])
        elif state != "attentive":
            self._roll_bias_target = 0.0
        if params["saccade"] is None:
            # states without saccades shouldn't keep holding an old gaze
            # shift (the 10s relax is far too slow for suppression)
            self._gaze_bias_yaw = 0.0
            self._gaze_bias_pitch = 0.0

    # ------------------------------------------------------------- worker

    def _worker(self) -> None:
        from .robot_api import tool_param

        was_zero = False
        while not self._stop_event.wait(_DT):
            try:
                now = time.monotonic()
                desired = self._desired_state(now)
                if desired != self._state:
                    self._enter_state(desired, now)
                params = _STATES[self._state]

                try:
                    intensity = tool_param("motion_texture", "intensity") / 100.0
                except Exception:
                    intensity = 1.0

                # cross-fade generator parameters toward the state's table
                blend_secs = (_SUPPRESS_BLEND_SECS if self._state == "off"
                              else _BLEND_SECS)
                alpha = min(1.0, _DT / blend_secs)
                for i in range(_DOFS):
                    self._std[i] += (params["std"][i] * intensity - self._std[i]) * alpha
                    self._theta[i] += (params["theta"][i] - self._theta[i]) * alpha
                    # the exponential blend never truly reaches zero; snap
                    # negligible noise off so suppression actually settles
                    if self._std[i] < 1e-4 and params["std"][i] == 0.0:
                        self._std[i] = 0.0
                self._pitch_bias += (params["pitch_bias"] * intensity
                                     - self._pitch_bias) * alpha
                self._roll_bias += (self._roll_bias_target * intensity
                                    - self._roll_bias) * alpha

                # Ornstein-Uhlenbeck step per DOF
                for i in range(_DOFS):
                    theta = self._theta[i]
                    sigma = self._std[i] * math.sqrt(2.0 * theta)
                    self._ou[i] += (-theta * self._ou[i] * _DT
                                    + sigma * math.sqrt(_DT) * random.gauss(0.0, 1.0))

                # saccade-like gaze shifts (idle/speaking): jump a held bias
                if params["saccade"] and now >= self._next_saccade:
                    lo, hi, jump = params["saccade"]
                    self._gaze_bias_yaw = random.uniform(-jump, jump) * intensity
                    self._gaze_bias_pitch = random.uniform(-jump, jump) * 0.5 * intensity
                    self._next_saccade = now + random.uniform(lo, hi)
                    self._saccade_until = now + _SACCADE_WINDOW_SECS
                # held gaze bias relaxes home slowly between saccades
                relax = math.exp(-_DT / 10.0)
                self._gaze_bias_yaw *= relax
                self._gaze_bias_pitch *= relax

                target = [
                    self._ou[0],
                    self._ou[1],
                    self._ou[2],
                    self._ou[3] + self._roll_bias,
                    self._ou[4] + self._pitch_bias + self._gaze_bias_pitch,
                    self._ou[5] + self._gaze_bias_yaw,
                ]

                # slew-limit and clamp the emitted offsets
                rot_rate = (_SACCADE_SLEW_ROT if now < self._saccade_until
                            else _SLEW_ROT)
                for i in range(_DOFS):
                    rate = _SLEW_XYZ if i < 3 else rot_rate
                    lim = _CLAMP[i]
                    goal = max(-lim, min(lim, target[i]))
                    step = rate * _DT
                    delta = goal - self._out[i]
                    if delta > step:
                        self._out[i] += step
                    elif delta < -step:
                        self._out[i] -= step
                    else:
                        self._out[i] = goal

                # snap tiny residue to true zero so the movement loop can
                # take its identity fast path when everything is quiet
                if self._state == "off" and all(abs(v) < 5e-4 for v in self._out):
                    if not was_zero:
                        self._out = [0.0] * _DOFS
                        self._ou = [0.0] * _DOFS
                        self.emit(_ZERO6)
                        was_zero = True
                else:
                    was_zero = False
                    self.emit(tuple(self._out))

                self._telemetry = {
                    "state": self._state,
                    "yaw_deg": round(math.degrees(self._out[5]), 2),
                    "pitch_deg": round(math.degrees(self._out[4]), 2),
                }
            except Exception as e:
                logger.warning(f"motion texture tick failed: {e}")

        try:
            self.emit(_ZERO6)
        except Exception:
            pass
        logger.info("Motion texture stopped")

    # ------------------------------------------------------------ telemetry

    def status(self) -> dict:
        return dict(self._telemetry)
