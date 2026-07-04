"""Deterministic clock/date tool — the LLM must never guess the time."""

import logging
from datetime import datetime

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class LocalTimeConfig(FunctionBaseConfig, name="local_time"):
    """Current local time and date."""


@register_function(config_type=LocalTimeConfig)
async def local_time_fn(config: LocalTimeConfig, builder: Builder):

    async def _now(query: str = "") -> str:
        now = datetime.now()
        day = now.day
        suffix = ("th" if 11 <= day % 100 <= 13
                  else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th"))
        return (f"It's {now:%H:%M} on {now:%A} the {day}{suffix} "
                f"of {now:%B %Y}.")

    yield FunctionInfo.from_fn(
        _now,
        description=(
            "The current local time and date. ALWAYS use this for any "
            "question about the time, date, or day of the week — never "
            "answer those from memory. Input is ignored."
        ),
    )
