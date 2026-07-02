import os
import threading
import time
import logging
from reachy_mini import ReachyMini
from .moves import MovementManager
from .wobbler import HeadWobbler
from .dance_emotion_moves import GotoQueueMove
from .animation_player import AnimationLibrary, AnimationQueueMove
from reachy_mini.utils import create_head_pose

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# Daemon process helpers adapted from NVIDIA-AI-IOT/reachy-mini-jetson-assistant
# (Apache-2.0), app/reachy.py.

def is_daemon_running() -> bool:
    """Check if a reachy-mini-daemon process exists on this machine."""
    if not psutil:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("reachy-mini-daemon" in part or "reachy_mini.daemon" in part for part in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


def kill_daemon() -> bool:
    """Kill a stale reachy-mini-daemon process. Returns True if one was found."""
    if not psutil:
        return False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("reachy-mini-daemon" in part or "reachy_mini.daemon" in part for part in cmdline):
                logger.warning(f"Killing stale Reachy daemon (PID {proc.pid})")
                proc.kill()
                time.sleep(2)
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError):
            continue
    return False


class ReachyService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.connected = False
        self.animations = AnimationLibrary()

        # All connection behavior is env-driven so the same code runs against
        # the MuJoCo sim (default), a USB-attached robot, or a daemon reachable
        # over the network (REACHY_LOCALHOST_ONLY=false).
        self.use_sim = _env_bool("REACHY_USE_SIM", True)
        self.localhost_only = _env_bool("REACHY_LOCALHOST_ONLY", False)
        self.spawn_daemon = _env_bool("REACHY_SPAWN_DAEMON", False)
        self.wake_on_start = _env_bool("REACHY_WAKE_ON_START", True)
        self.timeout = float(os.getenv("REACHY_TIMEOUT", "15.0"))
        self.retry_attempts = int(os.getenv("REACHY_RETRY_ATTEMPTS", "3"))
        self.startup_wait = float(os.getenv("REACHY_STARTUP_WAIT", "5.0"))
        # The bot only drives motors through the daemon; camera/mic/speaker
        # come from the WebRTC transport, so don't grab the robot's devices.
        self.media_backend = os.getenv("REACHY_MEDIA_BACKEND", "no_media")

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = ReachyService()
        return cls._instance

    def _connect_with_retries(self) -> ReachyMini | None:
        """Connect to the daemon, retrying with escalating recovery.

        Attempt 0 connects directly; attempt 1 waits for a possibly
        still-starting daemon; later attempts kill a stale local daemon first.
        """
        for attempt in range(max(1, self.retry_attempts)):
            try:
                if attempt == 1:
                    logger.info(f"Daemon may still be starting, waiting {self.startup_wait:.0f}s...")
                    time.sleep(self.startup_wait)
                elif attempt > 1 and not self.use_sim:
                    kill_daemon()

                return ReachyMini(
                    use_sim=self.use_sim,
                    spawn_daemon=self.spawn_daemon,
                    localhost_only=self.localhost_only,
                    timeout=self.timeout,
                    log_level=os.getenv("REACHY_LOG_LEVEL", "INFO"),
                    media_backend=self.media_backend,
                )
            except Exception as e:
                err_msg = str(e).lower()
                retryable = attempt < self.retry_attempts - 1
                if retryable and ("localhost and network" in err_msg or "both localhost" in err_msg or "timeout" in err_msg):
                    logger.warning(f"Reachy connection attempt {attempt + 1} failed ({e}), retrying...")
                    continue
                if retryable:
                    logger.warning(f"Reachy connection attempt {attempt + 1} failed ({e}), retrying...")
                    continue
                raise
        return None

    def connect(self):
        # If already connected, return
        if self.connected:
            logger.debug("Reachy already connected")
            return

        # If previously disconnected, clean up any leftover state
        if self.robot or self.motion_manager or self.wobbler:
            logger.info("Cleaning up previous Reachy connection...")
            self.disconnect()

        try:
            mode = "simulation" if self.use_sim else "hardware"
            logger.info(f"Connecting to Reachy Mini daemon ({mode} mode)...")

            self.robot = self._connect_with_retries()
            logger.info("Successfully connected to Reachy Mini daemon")

            if not self.use_sim and self.wake_on_start:
                try:
                    self.robot.enable_motors()
                    self.robot.wake_up()
                    time.sleep(0.5)
                    logger.info("Reachy Mini awake (motors enabled)")
                except Exception as e:
                    logger.warning(f"Wake-up sequence failed (continuing): {e}")

            # 1. Initialize Motor Cortex (Background Thread)
            self.motion_manager = MovementManager(self.robot)
            self.motion_manager.start()

            # 2. Initialize Auditory Cortex (Links Audio -> Motion)
            self.wobbler = HeadWobbler(self.motion_manager.set_speech_offsets)
            self.wobbler.start()

            self.connected = True
            logger.info("Reachy Service Started: Breathing & Sway active.")
        except Exception as e:
            import traceback
            logger.warning(f"Reachy Mini daemon not available: {e}")
            logger.warning(f"Full traceback: {traceback.format_exc()}")
            logger.warning("Pipeline will continue without Reachy robot control.")
            logger.warning("To enable Reachy: start the daemon, e.g. "
                           "'uv run -m reachy_mini.daemon.app.main --no-localhost-only' "
                           "(add --sim for simulation; use mjpython on macOS for sim)")

            # Clean up partial robot object to avoid destructor errors
            self.robot = None
            # Don't raise - allow pipeline to run without Reachy

    def feed_audio(self, audio_chunk_base64):
        """Feeds audio from TTS to the wobble engine."""
        if self.wobbler:
            logger.info("Feeding audio to Reachy")
            self.wobbler.feed(audio_chunk_base64)

    def set_listening_pose(self):
        """Sets robot back to listening/idle pose."""
        if self.motion_manager:
            self.motion_manager.set_listening(True)
            logger.info("Reachy set to listening pose")

    def look_at(self, direction: str):
        """Maps semantic direction to robot pose."""
        if not self.connected or not self.motion_manager or not self.robot:
            logger.debug(f"Reachy not connected - ignoring look_at({direction})")
            return

        # Mapping adapted from Reachy tools
        DELTAS = {
            "left": (0, 0, 0, 0, 0, 40),
            "right": (0, 0, 0, 0, 0, -40),
            "up": (0, 0, 0, 0, -30, 0),
            "down": (0, 0, 0, 0, 30, 0),
            "front": (0, 0, 0, 0, 0, 0),
        }
        deltas = DELTAS.get(direction, DELTAS["front"])

        try:
            target_pose = create_head_pose(*deltas, degrees=True)
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()

            goto_move = GotoQueueMove(
                target_head_pose=target_pose,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),
                start_antennas=(current_antennas[0], current_antennas[1]),
                target_body_yaw=0,
                start_body_yaw=0,
                duration=1.0
            )
            self.motion_manager.queue_move(goto_move)
            self.motion_manager.set_moving_state(1.0)
            logger.info(f"Reachy looking {direction}")
        except Exception as e:
            logger.error(f"Look at failed: {e}")

    def list_animations(self) -> list:
        """Names of the available expressive animation clips."""
        return self.animations.names()

    def play_animation(self, name: str) -> bool:
        """Queue an expressive animation clip (e.g. nod, attentive, intrigued5).

        Returns True if the clip was queued.
        """
        clip = self.animations.get(name)
        if clip is None:
            logger.warning(f"Unknown animation '{name}' (available: {', '.join(self.list_animations())})")
            return False

        if not self.connected or not self.motion_manager or not self.robot:
            logger.debug(f"Reachy not connected - ignoring play_animation({name})")
            return False

        try:
            current_head_pose = self.robot.get_current_head_pose()
            _, current_antennas = self.robot.get_current_joint_positions()
            move = AnimationQueueMove(
                clip,
                start_head_pose=current_head_pose,
                start_antennas=(current_antennas[0], current_antennas[1]),
            )
            self.motion_manager.queue_move(move)
            self.motion_manager.set_moving_state(1.0)
            logger.info(f"Reachy playing animation '{name}' ({clip.duration:.1f}s)")
            return True
        except Exception as e:
            logger.error(f"play_animation('{name}') failed: {e}")
            return False

    def disconnect(self):
        """Disconnect and cleanup Reachy resources."""
        if not self.connected:
            return

        logger.info("Disconnecting Reachy service...")

        # Stop background threads
        if self.motion_manager:
            self.motion_manager.stop()
        if self.wobbler:
            self.wobbler.stop()

        # Disconnect robot
        if self.robot:
            try:
                # The robot client should disconnect gracefully
                if hasattr(self.robot, 'client') and self.robot.client:
                    self.robot.client.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting robot: {e}")

        # Reset state
        self.robot = None
        self.motion_manager = None
        self.wobbler = None
        self.connected = False

        logger.info("Reachy service disconnected")

    def stop(self):
        """Alias for disconnect for backwards compatibility."""
        self.disconnect()
