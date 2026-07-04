"""Timer/reminder tools — thin clients of the bot's /reminders API.

The bot owns the clock and the mouth (its scheduler announces due items
through the robot's speaker); these tools just set, list, and cancel.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class SetReminderConfig(FunctionBaseConfig, name="robot_set_reminder"):
    """Set a timer or reminder on the robot."""
    base_url: str = Field(default="http://localhost:7861",
                          description="Robot API base URL")


@register_function(config_type=SetReminderConfig)
async def robot_set_reminder_fn(config: SetReminderConfig, builder: Builder):
    import httpx

    base = config.base_url.rstrip("/")
    client = httpx.AsyncClient(timeout=10.0)

    async def _set(when_and_label: str) -> str:
        try:
            when, _, label = when_and_label.partition("|")
            r = await client.post(f"{base}/reminders",
                                  json={"when": when.strip(),
                                        "label": label.strip() or "your reminder"})
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.error(f"set_reminder failed: {e}")
            return f"I couldn't set that reminder ({e})."
        if not data.get("ok"):
            return ("I couldn't understand that time. Use a duration like "
                    "'10m' or '1h 30m', or a clock time like '15:00'.")
        item = data["reminder"]
        mins = max(0, round((item["due_ts"] - __import__("time").time()) / 60))
        return (f"Done — I'll remind you about {item['label']} "
                f"in about {mins} minute{'s' if mins != 1 else ''}."
                if mins < 120 else
                f"Done — reminder set for {item['label']}.")

    try:
        yield FunctionInfo.from_fn(
            _set,
            description=(
                "Set a timer, alarm or reminder the robot will announce out "
                "loud when due. Input format: '<when> | <label>' where <when> "
                "is a duration like '10m', '90s', '1h 30m' or a 24h clock "
                "time like '15:00', and <label> is what to say. Examples: "
                "'10m | check the oven', '07:30 | wake up call'."
            ),
        )
    finally:
        await client.aclose()


class ListRemindersConfig(FunctionBaseConfig, name="robot_list_reminders"):
    """List the robot's pending timers and reminders."""
    base_url: str = Field(default="http://localhost:7861",
                          description="Robot API base URL")


@register_function(config_type=ListRemindersConfig)
async def robot_list_reminders_fn(config: ListRemindersConfig, builder: Builder):
    import httpx

    base = config.base_url.rstrip("/")
    client = httpx.AsyncClient(timeout=10.0)

    async def _list(query: str = "") -> str:
        try:
            r = await client.get(f"{base}/reminders")
            r.raise_for_status()
            items = r.json().get("reminders", [])
        except Exception as e:
            return f"I couldn't check the reminders ({e})."
        if not items:
            return "There are no timers or reminders set."
        parts = []
        for it in items:
            secs = it["remaining_secs"]
            when = (f"{secs // 3600} hours {round((secs % 3600) / 60)} minutes"
                    if secs >= 3600 else
                    f"{secs // 60} minutes" if secs >= 60 else f"{secs} seconds")
            parts.append(f"{it['label']} in {when}")
        return "Current reminders: " + "; ".join(parts) + "."

    try:
        yield FunctionInfo.from_fn(
            _list,
            description=("List all pending timers and reminders. Input is "
                         "ignored (pass an empty string)."),
        )
    finally:
        await client.aclose()


class CancelReminderConfig(FunctionBaseConfig, name="robot_cancel_reminder"):
    """Cancel a pending timer or reminder."""
    base_url: str = Field(default="http://localhost:7861",
                          description="Robot API base URL")


@register_function(config_type=CancelReminderConfig)
async def robot_cancel_reminder_fn(config: CancelReminderConfig, builder: Builder):
    import httpx

    base = config.base_url.rstrip("/")
    client = httpx.AsyncClient(timeout=10.0)

    async def _cancel(query: str) -> str:
        try:
            r = await client.delete(f"{base}/reminders/{query.strip()}")
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            return f"I couldn't cancel that ({e})."
        if not data.get("ok"):
            remaining = data.get("reminders", [])
            names = ", ".join(i["label"] for i in remaining) or "none"
            return (f"I couldn't find a reminder matching '{query}'. "
                    f"Current reminders: {names}.")
        return f"Cancelled the reminder for {data['cancelled']['label']}."

    try:
        yield FunctionInfo.from_fn(
            _cancel,
            description=("Cancel a pending timer or reminder. Input is a few "
                         "words from its label, e.g. 'oven' or 'tea'."),
        )
    finally:
        await client.aclose()
