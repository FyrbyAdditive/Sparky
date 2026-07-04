"""Weather via Open-Meteo — a proper keyless JSON API, not scraping.

Weather questions previously fell to web_search, which choked on
JS-rendered weather sites (weather.com placeholders, AccuWeather 403s).
Open-Meteo is free, keyless, and structured. Internet access, so it sits
behind the same panel gate as web search (fail-closed).
"""

import logging
import time

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from ces_tutorial.functions.web_search import _announce, _gate_enabled

logger = logging.getLogger(__name__)

_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CACHE_TTL = 600.0
_cache: dict[str, tuple[float, str]] = {}

_ANNOUNCE_WEATHER = [
    "Checking the forecast.",
    "Let me look at the weather.",
    "One moment, reading the skies.",
    "Consulting the meteorologists.",
    "Fetching the forecast now.",
    "Let me see what the weather holds.",
]

# WMO weather interpretation codes -> spoken text
_WMO = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "severe thunderstorms with hail",
}


def _wmo(code) -> str:
    return _WMO.get(int(code or 0), "mixed conditions")


class WeatherConfig(FunctionBaseConfig, name="open_meteo_weather"):
    """Weather forecasts from Open-Meteo (keyless, JSON)."""
    robot_api_base_url: str = Field(
        default="http://localhost:7861",
        description="Robot API base URL (optional-tools registry)")
    default_location: str = Field(
        default="", description="Fallback place when none is given")


@register_function(config_type=WeatherConfig)
async def open_meteo_weather_fn(config: WeatherConfig, builder: Builder):
    import os

    import httpx

    client = httpx.AsyncClient(timeout=12.0)
    default_loc = config.default_location or os.getenv(
        "WEATHER_DEFAULT_LOCATION", "Stockholm")

    async def _forecast(query: str) -> str:
        enabled = await _gate_enabled(client, config.robot_api_base_url)
        if enabled is None:
            return ("I couldn't confirm internet access is enabled, so I "
                    "stayed offline. Do not call this tool again for this request.")
        if not enabled:
            return ("Web access is turned off in the robot's control panel, "
                    "so I can't fetch the forecast. Do not call this tool "
                    "again for this request.")

        place, _, scope = query.partition("|")
        place = place.strip() or default_loc
        scope = scope.strip().lower() or "today"
        key = f"{place.lower()}|{scope}"
        hit = _cache.get(key)
        if hit and time.monotonic() - hit[0] < _CACHE_TTL:
            return hit[1]

        await _announce(client, config.robot_api_base_url, _ANNOUNCE_WEATHER)
        try:
            g = await client.get(_GEO_URL, params={"name": place, "count": 1})
            g.raise_for_status()
            hits = g.json().get("results") or []
            if not hits:
                return f"I couldn't find a place called {place}."
            loc = hits[0]
            f = await client.get(_FORECAST_URL, params={
                "latitude": loc["latitude"], "longitude": loc["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m",
                "daily": ("weather_code,temperature_2m_max,temperature_2m_min,"
                          "precipitation_probability_max"),
                "timezone": "auto", "forecast_days": 7,
            })
            f.raise_for_status()
            data = f.json()
        except Exception as e:
            logger.error(f"weather lookup failed: {e}")
            return f"I couldn't reach the weather service ({e})."

        name = loc.get("name", place)
        cur = data.get("current", {})
        daily = data.get("daily", {})
        days = daily.get("time", [])
        out = [f"In {name} right now: {_wmo(cur.get('weather_code'))}, "
               f"{round(cur.get('temperature_2m', 0))} degrees."]

        def day_line(i, label):
            return (f"{label}: {_wmo(daily['weather_code'][i])}, "
                    f"{round(daily['temperature_2m_min'][i])} to "
                    f"{round(daily['temperature_2m_max'][i])} degrees, "
                    f"{daily['precipitation_probability_max'][i] or 0} percent "
                    "chance of precipitation.")

        if days:
            if scope.startswith("week"):
                import datetime as _dt
                for i in range(min(7, len(days))):
                    label = _dt.date.fromisoformat(days[i]).strftime("%A")
                    out.append(day_line(i, label))
            elif scope.startswith("tomorrow") and len(days) > 1:
                out.append(day_line(1, "Tomorrow"))
            else:
                out.append(day_line(0, "Today"))
                if len(days) > 1:
                    out.append(day_line(1, "Tomorrow"))
        result = " ".join(out)
        _cache[key] = (time.monotonic(), result)
        return result

    try:
        yield FunctionInfo.from_fn(
            _forecast,
            description=(
                "Get the weather forecast — ALWAYS prefer this over "
                "web_search for weather questions. Input: '<place>' or "
                "'<place> | tomorrow' or '<place> | week'; an empty place "
                "uses the home location. Returns current conditions plus "
                "today/tomorrow (or the week)."
            ),
        )
    finally:
        await client.aclose()
