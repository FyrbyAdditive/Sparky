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
                "name from: nod, attentive, intrigued5, antennaSmallWiggle, "
                "antennaLargeWiggle, lookAroundShort, scan, listen1, talking, "
                "talkingLeftShoulder, talkingRightShoulder, wakeUp1, sleep3, focus, "
                "idle3old, takePicture, picturePreparation. Use when the user asks the "
                "robot to move, dance, nod, wiggle its antennas, look around, wake up, "
                "or go to sleep, or to add physical expression to a response."
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
