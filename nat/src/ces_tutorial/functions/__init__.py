# Import functions to ensure registration happens
from ces_tutorial.functions.router import router_fn
from ces_tutorial.functions.router_agent import router_agent_fn
from ces_tutorial.functions.wiki_offline import wiki_search_offline_fn
from ces_tutorial.functions.robot_tools import (
    robot_play_animation_fn,
    robot_look_at_fn,
    robot_remember_speaker_name_fn,
)
from ces_tutorial.functions.web_search import (
    web_search_fn,
    web_read_page_fn,
)
from ces_tutorial.functions.react_stream import streaming_react_agent_workflow
from ces_tutorial.functions.robot_tools import (
    robot_take_photo_fn,
    robot_adjust_fn,
)
from ces_tutorial.functions.reminder_tools import (
    robot_set_reminder_fn,
    robot_list_reminders_fn,
    robot_cancel_reminder_fn,
)
from ces_tutorial.functions.weather import open_meteo_weather_fn
from ces_tutorial.functions.local_time import local_time_fn

__all__ = [
    "router_fn",
    "router_agent_fn",
    "wiki_search_offline_fn",
    "robot_play_animation_fn",
    "robot_look_at_fn",
    "robot_remember_speaker_name_fn",
    "web_search_fn",
    "web_read_page_fn",
    "streaming_react_agent_workflow",
    "robot_take_photo_fn",
    "robot_adjust_fn",
    "robot_set_reminder_fn",
    "robot_list_reminders_fn",
    "robot_cancel_reminder_fn",
    "open_meteo_weather_fn",
    "local_time_fn",
]
