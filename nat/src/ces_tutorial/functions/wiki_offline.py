"""Offline Wikipedia search tool.

Drop-in replacement for NAT's stock `wiki_search` (which calls wikipedia.org)
backed by the local txtai Wikipedia service in deploy/wiki-offline. Same
string-in/string-out shape, so the react_agent uses it identically.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class WikiSearchOfflineConfig(FunctionBaseConfig, name="wiki_search_offline"):
    """Search a local offline Wikipedia index (no internet access needed)."""
    base_url: str = Field(
        default="http://localhost:8040",
        description="Base URL of the wiki-offline service",
    )
    robot_api_base_url: str = Field(
        default="http://localhost:7861",
        description="Robot API base URL (spoken search narration)",
    )
    max_results: int = Field(default=2, description="Number of articles to return")


# Spoken narration while the lookup runs (see web_search._announce).
_ANNOUNCE_WIKI = [
    "Let me check Wikipedia.",
    "Consulting my encyclopedia.",
    "Flipping through Wikipedia.",
    "One moment, checking Wikipedia.",
    "Let me look that up in the encyclopedia.",
    "Checking my knowledge base.",
    "Wikipedia should know this.",
    "Paging through the encyclopedia.",
    "Let me consult the archives.",
    "Looking through Wikipedia now.",
    "A quick encyclopedia check.",
    "Let me verify that in Wikipedia.",
    "Searching my offline library.",
    "Digging into the encyclopedia.",
    "Give me a second with Wikipedia.",
    "Thumbing through my reference books.",
    "The encyclopedia will settle this.",
    "Checking the facts in Wikipedia.",
    "Let me pull up the article.",
    "Consulting the collected knowledge.",
]


@register_function(config_type=WikiSearchOfflineConfig)
async def wiki_search_offline_fn(config: WikiSearchOfflineConfig, builder: Builder):
    import httpx

    base_url = config.base_url.rstrip("/")
    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=30.0)

    async def _search(query: str) -> str:
        from ces_tutorial.functions.web_search import _announce

        await _announce(client, config.robot_api_base_url, _ANNOUNCE_WIKI)
        try:
            response = await client.get(
                f"{base_url}/search",
                params={"q": query, "n": config.max_results},
            )
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            logger.error(f"Offline wiki search failed: {e}")
            return f"Wikipedia search is currently unavailable ({e})."

        results = data.get("results", [])
        if not results:
            return f"No Wikipedia results found for '{query}'."

        return "\n\n".join(f"{r['id']}: {r['text']}" for r in results)

    try:
        yield FunctionInfo.from_fn(
            _search,
            description=(
                "Search a local offline copy of Wikipedia for factual information "
                "about people, places, events, and concepts. Input is a search query string."
            ),
        )
    finally:
        await client.aclose()
