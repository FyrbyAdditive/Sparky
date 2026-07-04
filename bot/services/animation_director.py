"""Animation Director: single owner of expressive-animation decisions.

Every animation on the robot flows through here as a *request*. Sources
carry priorities; the director arbitrates against what is currently
playing, keeps at most one clip pending behind the current one, applies
per-source selection policies (emotion pools with cooldowns, the
speaking-reply repeat, idle stirring) and the random-mirror decision, and
exposes what is playing for /status.

It is deliberately an in-process module, not a separate service: motion
is latency-sensitive and the robot handle lives in this process. The
"service" semantics come from the request API and single ownership.

Priorities: user (panel / agent tool) > emotion > speaking > idle.
A request beats strictly lower-priority work: it replaces whatever is
pending (MovementManager.clear_pending) and queues behind the currently
playing move — hard mid-move preemption is intentionally out of scope.
Equal-priority user requests replace the pending slot so panel clicks
feel responsive without stacking a backlog.

The e-stop path bypasses the director entirely (safety goes straight to
the motion manager), and animation sound effects remain a concern of the
HTTP endpoint layer.
"""

import asyncio
import logging
import os
import random
import time

logger = logging.getLogger(__name__)

PRIORITIES = {"user": 3, "emotion": 2, "speaking": 1, "idle": 0}

# Emotion -> candidate clips (moved from emotion.py). One is chosen at
# random per reaction; names missing from the library drop out on first
# use. Keyed by Emotion.value strings.
EMOTION_POOLS = {
    "happy": ["cheerful1", "laughing1", "success1", "antennaSmallWiggle"],
    "excited": ["enthusiastic1", "enthusiastic2", "amazed1", "antennaLargeWiggle"],
    "sad": ["sad1", "sad2", "attentive"],
    "curious": ["curious1", "inquiring1", "inquiring2", "intrigued5"],
    "greeting": ["welcoming1", "welcoming2", "antennaLargeWiggle"],
    "farewell": ["nod", "yes1"],
    "grateful": ["grateful1", "proud1", "nod"],
}
EMOTION_COOLDOWN_SECS = 8.0

# Clips authored for talking; one is chosen per reply and repeated until
# the speech ends.
SPEAKING_CLIPS = ["talking", "talkingLeftShoulder", "talkingRightShoulder"]

# Gentle clips for idle stirring.
IDLE_CLIPS = [
    "attentive", "lookAroundShort", "antennaSmallWiggle", "idle3old",
    "listen1", "thoughtful1", "thoughtful2", "curious1", "boredom1",
    "boredom2", "serenity1", "calming1",
]

# small settle margin added after each clip's duration when estimating
# how long the motion queue stays busy
_SETTLE_SECS = 0.3


class AnimationDirector:
    def __init__(self, service):
        self.service = service
        # unexpired timeline entries: the playing clip plus at most one
        # pending one, each {"clip","source","priority","until"}
        self._timeline: list[dict] = []
        self.plays = {s: 0 for s in PRIORITIES}
        self.rejected = 0
        # speaking-reply state
        self._speaking_grace = float(os.getenv("SPEAKING_ANIM_GRACE_SECS", "3.0"))
        self._speaking_since: float | None = None
        self._reply_clip: str | None = None
        self._speaking_pause_until = 0.0
        # idle state
        self._idle_last_play = time.monotonic()
        self._idle_next_gap = 0.0
        # emotion cooldowns per intent
        self._emotion_last: dict[str, float] = {}
        self._pool_warned = False

    # ------------------------------------------------------------- helpers

    def _prune(self, now: float):
        self._timeline = [e for e in self._timeline if e["until"] > now]

    def _resolve_clip(self, source: str, clip: str | None, intent: str | None):
        """Explicit clip, or a pool pick for intent-based requests.
        Returns (clip_name, reason_if_failed)."""
        if clip is not None:
            if self.service.animations.get(clip) is None:
                return None, "unknown_animation"
            return clip, None
        if source == "emotion" and intent:
            pool = [c for c in EMOTION_POOLS.get(intent, [])
                    if self.service.animations.get(c)]
            if not pool:
                return None, "no_clips_for_intent"
            return random.choice(pool), None
        return None, "no_clip_or_intent"

    # ------------------------------------------------------------- requests

    def request(self, source: str, clip: str | None = None,
                intent: str | None = None, mirror: bool | None = None) -> dict:
        """Ask to play an animation. Returns {"accepted", "reason", "clip"}."""
        now = time.monotonic()
        self._prune(now)

        if source not in PRIORITIES:
            return {"accepted": False, "reason": "unknown_source", "clip": None}

        name, fail = self._resolve_clip(source, clip, intent)
        if name is None:
            return {"accepted": False, "reason": fail, "clip": None}

        if source == "emotion" and intent:
            if now - self._emotion_last.get(intent, 0.0) < EMOTION_COOLDOWN_SECS:
                return {"accepted": False, "reason": "cooldown", "clip": name}

        prio = PRIORITIES[source]
        if self._timeline:
            top = max(e["priority"] for e in self._timeline)
            # strictly higher wins; equal user requests replace the pending
            # slot so rapid panel clicks stay responsive without stacking
            if prio < top or (prio == top and source != "user"):
                self.rejected += 1
                return {"accepted": False, "reason": "busy", "clip": name}
            if len(self._timeline) > 1:
                self._timeline = self._timeline[:1]
            if self.service.motion_manager:
                self.service.motion_manager.clear_pending()

        if not self.service.play_animation(name, mirror=mirror):
            return {"accepted": False, "reason": "robot_unavailable", "clip": name}

        start = self._timeline[-1]["until"] if self._timeline else now
        duration = self.service.animations.get(name).duration
        self._timeline.append({"clip": name, "source": source, "priority": prio,
                               "until": start + duration + _SETTLE_SECS})
        self.plays[source] += 1
        if source == "emotion" and intent:
            self._emotion_last[intent] = now
        # ANY play restarts the idle gap: without this, an idle clip could
        # trail a user/emotion/speaking play the moment it finished (the
        # uninvited-encore bug) because the gap only measured idle-to-idle
        self._idle_last_play = now
        return {"accepted": True, "reason": None, "clip": name}

    # ------------------------------------------------------------- behaviors

    def _tick_speaking(self, now: float, gate: dict, enabled: bool):
        if not enabled or not gate["bot_speaking"]:
            self._speaking_since = None
            self._reply_clip = None
            return
        if self._speaking_since is None:
            self._speaking_since = now
        if now - self._speaking_since < self._speaking_grace:
            return
        if now < self._speaking_pause_until or self._timeline:
            return
        if self._reply_clip is None:
            avail = [c for c in SPEAKING_CLIPS if self.service.animations.get(c)]
            if not avail:
                return
            self._reply_clip = random.choice(avail)
        res = self.request("speaking", clip=self._reply_clip)
        if res["accepted"]:
            # brief intentional stillness between repeats
            self._speaking_pause_until = (self._timeline[-1]["until"]
                                          + random.uniform(0.4, 1.2))

    def _tick_idle(self, now: float, gate: dict, enabled: bool,
                   still_secs: float, min_gap: float, max_gap: float,
                   last_interaction: float):
        if not enabled or gate["bot_speaking"] or self._timeline:
            return
        if now - last_interaction < still_secs:
            return
        if self._idle_next_gap == 0.0:
            self._idle_next_gap = random.uniform(min_gap, max_gap)
        if now - self._idle_last_play < self._idle_next_gap:
            return
        avail = [c for c in IDLE_CLIPS if self.service.animations.get(c)]
        if not avail:
            if not self._pool_warned:
                self._pool_warned = True
                logger.warning("animation director: idle pool empty")
            return
        if self.request("idle", clip=random.choice(avail))["accepted"]:
            self._idle_next_gap = random.uniform(min_gap, max_gap)

    async def tick_task(self):
        """1s behavior tick: speaking-reply repeats and idle stirring."""
        # imported lazily: robot_api imports this module's singleton
        from .local_audio import GATE
        from .robot_api import LAST_INTERACTION, OPTIONAL_TOOLS, tool_param

        while True:
            await asyncio.sleep(1.0)
            try:
                if not self.service.connected:
                    continue
                now = time.monotonic()
                self._prune(now)
                self._tick_speaking(
                    now, GATE, OPTIONAL_TOOLS["speaking_animations"]["enabled"])
                self._tick_idle(
                    now, GATE, OPTIONAL_TOOLS["idle_animations"]["enabled"],
                    tool_param("idle_animations", "still_secs"),
                    tool_param("idle_animations", "min_gap_secs"),
                    tool_param("idle_animations", "max_gap_secs"),
                    LAST_INTERACTION["ts"])
            except Exception as e:
                logger.warning(f"animation director tick: {e}")

    # ------------------------------------------------------------- telemetry

    def status(self) -> dict:
        now = time.monotonic()
        self._prune(now)
        current = self._timeline[0] if self._timeline else None
        pending = self._timeline[1] if len(self._timeline) > 1 else None
        return {
            "clip": current and current["clip"],
            "source": current and current["source"],
            "remaining_secs": current and round(current["until"] - now, 1),
            "pending": pending and pending["clip"],
            "plays": dict(self.plays),
            "rejected": self.rejected,
        }


_director: AnimationDirector | None = None


def get_director() -> AnimationDirector:
    global _director
    if _director is None:
        from .reachy_service import ReachyService
        _director = AnimationDirector(ReachyService.get_instance())
    return _director
