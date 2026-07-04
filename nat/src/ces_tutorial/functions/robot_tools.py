"""Agent tools that let the ReAct agent deliberately move the robot.

They call the bot process's robot control API (bot/services/robot_api.py).
Pattern follows NVIDIA spark-reachy-photo-booth's agent tools -> robot
actions design (Apache-2.0).
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class PlayAnimationConfig(FunctionBaseConfig, name="robot_play_animation"):
    """Play an expressive animation on the Reachy Mini robot."""
    base_url: str = Field(
        default="http://localhost:7861",
        description="Base URL of the bot's robot control API",
    )


@register_function(config_type=PlayAnimationConfig)
async def robot_play_animation_fn(config: PlayAnimationConfig, builder: Builder):
    import httpx

    base_url = config.base_url.rstrip("/")
    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=10.0)

    async def _play(animation_name: str) -> str:
        name = animation_name.strip().strip("'\"")
        try:
            response = await client.post(
                f"{base_url}/robot/play_animation", json={"name": name}
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"play_animation tool failed: {e}")
            return f"Could not reach the robot ({e})."

        if data.get("ok"):
            return f"Playing animation '{name}' on the robot."
        if data.get("error") == "unknown_animation":
            available = ", ".join(data.get("animations") or [])
            return f"Unknown animation '{name}'. Available animations: {available}"
        return "The robot is not connected right now, so the animation could not be played."

    try:
        yield FunctionInfo.from_fn(
            _play,
            description=(
                "Play an expressive animation on the robot body. Input is one animation "
                "name. The library has ~117 clips across categories: gestures (nod, "
                "yes1, no1, attentive, intrigued5, antennaSmallWiggle, antennaLargeWiggle, "
                "lookAroundShort, scan), emotions (cheerful1, laughing1, amazed1, "
                "surprised1, proud1, sad1, scared1, shy1, loving1, grateful1, curious1, "
                "welcoming1, thoughtful1), dances (dance1, dance2, dance3, "
                "side_to_side_sway, groovy_sway_and_roll, jackson_square, dizzy_spin, "
                "yeah_nod), and states (wakeUp1, sleep3, focus, listen1, takePicture). "
                "Many emotions have numbered variants (e.g. proud2, proud3). If a name "
                "is unknown the tool replies with the full available list — pick from "
                "it and retry. Use when the user asks the robot to move, dance, nod, "
                "wiggle its antennas, look around, wake up, or go to sleep, or to add "
                "physical expression to a response."
            ),
        )
    finally:
        await client.aclose()


class RememberSpeakerNameConfig(FunctionBaseConfig, name="robot_remember_speaker_name"):
    """Associate a name with a diarized speaker number."""
    base_url: str = Field(
        default="http://localhost:7861",
        description="Base URL of the bot's robot control API",
    )


@register_function(config_type=RememberSpeakerNameConfig)
async def robot_remember_speaker_name_fn(config: RememberSpeakerNameConfig, builder: Builder):
    import re

    import httpx

    base_url = config.base_url.rstrip("/")
    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=10.0)

    async def _remember(speaker_and_name: str) -> str:
        text = speaker_and_name.strip().strip("'\"")
        m = re.search(r"(?:speaker\s*)?(\d+)\s*(?:is|:|=|,)?\s*(.*)", text, re.IGNORECASE)
        if not m:
            return ("Could not parse that. Input must be the speaker number then the "
                    "name, for example: 2 Tim")
        speaker, name = m.group(1), m.group(2).strip().strip("'\".")
        try:
            response = await client.post(
                f"{base_url}/speakers", json={"speaker": speaker, "name": name}
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"remember_speaker_name tool failed: {e}")
            return f"Could not reach the speaker registry ({e})."

        if not data.get("ok"):
            return f"Could not store that: {data.get('error', 'unknown error')}."
        if name and name.lower() not in ("forget", "none", "clear", "unknown"):
            return (f"Stored: speaker {speaker} is named {name}. Their future lines "
                    f"will be labeled with this name.")
        return f"Forgot the name for speaker {speaker}."

    try:
        yield FunctionInfo.from_fn(
            _remember,
            description=(
                "Remember the name of a person you are hearing. User lines are "
                "prefixed with the speaker, like 'Speaker 2:'. When a speaker tells "
                "you their name (or someone names another speaker), call this with "
                "the speaker number followed by the name, e.g. '2 Tim'. To forget a "
                "name use '2 forget'. Future lines from that speaker will then show "
                "their name."
            ),
        )
    finally:
        await client.aclose()


class LookAtConfig(FunctionBaseConfig, name="robot_look_at"):
    """Turn the robot's head to look in a direction."""
    base_url: str = Field(
        default="http://localhost:7861",
        description="Base URL of the bot's robot control API",
    )


@register_function(config_type=LookAtConfig)
async def robot_look_at_fn(config: LookAtConfig, builder: Builder):
    import httpx

    base_url = config.base_url.rstrip("/")
    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=10.0)

    async def _look(direction: str) -> str:
        direction = direction.strip().strip("'\"").lower()
        try:
            response = await client.post(
                f"{base_url}/robot/look_at", json={"direction": direction}
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"look_at tool failed: {e}")
            return f"Could not reach the robot ({e})."

        if data.get("ok"):
            return f"Robot is now looking {direction}."
        return f"Invalid direction '{direction}'. Valid: left, right, up, down, front."

    try:
        yield FunctionInfo.from_fn(
            _look,
            description=(
                "Turn the robot's head to look in a direction. Input is exactly one of: "
                "left, right, up, down, front. Use when the user asks the robot to look "
                "somewhere or turn its head."
            ),
        )
    finally:
        await client.aclose()


class TakePhotoConfig(FunctionBaseConfig, name="robot_take_photo"):
    """Take and save a photo with the robot's camera."""
    base_url: str = Field(default="http://localhost:7861",
                          description="Robot API base URL")


@register_function(config_type=TakePhotoConfig)
async def robot_take_photo_fn(config: TakePhotoConfig, builder: Builder):
    import httpx

    base = config.base_url.rstrip("/")
    client = httpx.AsyncClient(timeout=10.0)

    async def _photo(query: str = "") -> str:
        try:
            r = await client.post(f"{base}/photo")
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"I couldn't take a photo ({e})."
        if not data.get("ok"):
            return "The camera didn't cooperate — no photo this time."
        return ("Taking the picture now — say cheese! It's saved as "
                f"{data['file']} and viewable from the control panel.")

    try:
        yield FunctionInfo.from_fn(
            _photo,
            description=("Take a photo with the robot's camera (plays the "
                         "photo animation and saves the picture). Use when "
                         "someone asks to take a picture or photo. Input is "
                         "ignored."),
        )
    finally:
        await client.aclose()


class RobotAdjustConfig(FunctionBaseConfig, name="robot_settings"):
    """Adjust the robot's speaker volume or TTS voice by voice command."""
    base_url: str = Field(default="http://localhost:7861",
                          description="Robot API base URL")


@register_function(config_type=RobotAdjustConfig)
async def robot_adjust_fn(config: RobotAdjustConfig, builder: Builder):
    import random
    import re as _re

    import httpx

    base = config.base_url.rstrip("/")
    client = httpx.AsyncClient(timeout=10.0)

    _ACCENT = {"american": "a", "british": "b", "spanish": "e", "french": "f",
               "hindi": "h", "indian": "h", "italian": "i", "japanese": "j",
               "portuguese": "p", "chinese": "z"}

    async def _adjust(command: str) -> str:
        cmd = command.strip().lower()
        try:
            # ---- volume ----
            if any(w in cmd for w in ("volume", "quieter", "louder", "quiet", "loud")):
                r = await client.get(f"{base}/volume")
                current = int(r.json().get("percent", 100))
                m = _re.search(r"(\d{1,3})", cmd)
                if m:
                    target = int(m.group(1))
                elif any(w in cmd for w in ("down", "quieter", "quiet", "lower", "softer")):
                    target = current - 15
                elif any(w in cmd for w in ("up", "louder", "loud", "higher")):
                    target = current + 15
                else:
                    return f"The volume is at {current} percent."
                target = max(0, min(120, target))
                await client.post(f"{base}/volume", json={"percent": target})
                return f"Volume is now {target} percent."

            # ---- voice ----
            if "voice" in cmd:
                r = await client.get(f"{base}/voice")
                data = r.json()
                options = data.get("options", [])
                query = cmd.replace("voice", " ").strip()
                # exact id or name substring first
                matches = [v for v in options if query and query in v]
                if not matches:
                    prefix = ""
                    for word, letter in _ACCENT.items():
                        if word in cmd:
                            prefix = letter
                            break
                    gender = ("f" if _re.search(r"\b(female|woman)\b", cmd)
                              else "m" if _re.search(r"\b(male|man)\b", cmd) else "")
                    matches = [v for v in options
                               if (not prefix or v.startswith(prefix))
                               and (not gender or (len(v) > 1 and v[1] == gender))]
                if not matches:
                    return ("I couldn't match that to an installed voice. Try "
                            "an accent and gender, like 'a British male voice'.")
                choice = random.choice(matches)
                res = await client.post(f"{base}/voice", json={"voice": choice})
                if not res.json().get("ok"):
                    return "That voice didn't take, sorry."
                return f"Switched to the voice {choice.replace('_', ' ')}."

            return ("I can adjust 'volume up/down', 'volume 60', or switch "
                    "voice, like 'a British male voice'.")
        except Exception as e:
            return f"I couldn't adjust that ({e})."

    try:
        yield FunctionInfo.from_fn(
            _adjust,
            description=(
                "Adjust the robot's own settings by request: speaker volume "
                "('volume up', 'volume down', 'volume 60', 'quieter', "
                "'louder') or speaking voice ('british male voice', "
                "'american female voice', or a specific voice name). Use "
                "whenever someone asks the robot to speak quieter/louder or "
                "change its voice. Input is the plain request."
            ),
        )
    finally:
        await client.aclose()
