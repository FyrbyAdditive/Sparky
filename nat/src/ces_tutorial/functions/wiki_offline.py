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
    max_results: int = Field(default=2, description="Number of articles to return")


@register_function(config_type=WikiSearchOfflineConfig)
async def wiki_search_offline_fn(config: WikiSearchOfflineConfig, builder: Builder):
    import httpx

    base_url = config.base_url.rstrip("/")

    async def _search(query: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
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

    yield FunctionInfo.from_fn(
        _search,
        description=(
            "Search a local offline copy of Wikipedia for factual information "
            "about people, places, events, and concepts. Input is a search query string."
        ),
    )
