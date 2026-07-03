"""Live web search + page reading via DuckDuckGo's HTML endpoint.

These are OPTIONAL tools: before touching the network they ask the robot
API's optional-tools registry (panel > Tools) whether "web_search" is
enabled, and fail CLOSED — if the registry is unreachable or says no, they
return a refusal string instead of going online. That keeps the panel
toggle authoritative and instant (no NAT restart) and preserves the
project's offline-first posture.

No search API key and no new dependencies: DuckDuckGo's HTML endpoint is
fetched with httpx and parsed with beautifulsoup4 (both already shipped
transitively via nvidia-nat[langchain], and now declared in pyproject).
"""

import logging
from urllib.parse import parse_qs, urlparse

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)

# The single registry entry gating BOTH tools: search and page reading are
# one capability ("the robot may use the internet") with one panel switch.
GATE_TOOL = "web_search"

# ReAct agents retry failing tools (parse_agent_response_max_retries: 3),
# so the refusal must tell the model not to try again this turn.
DISABLED_MSG = (
    "Web search is turned off in the robot's control panel. Do not call "
    "this tool again for this request; answer from what you already know "
    "and mention that web search is disabled."
)
GATE_UNREACHABLE_MSG = (
    "I couldn't confirm that web search is enabled right now, so I stayed "
    "offline. Do not call this tool again for this request."
)

# DDG's HTML endpoint serves a bot-check page to clients without a
# plausible browser User-Agent.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
}


async def _gate_enabled(client, robot_api_base_url: str) -> bool | None:
    """True/False from the registry; None when the registry is unreachable
    (callers fail closed on None)."""
    try:
        r = await client.get(f"{robot_api_base_url.rstrip('/')}/tools", timeout=3.0)
        r.raise_for_status()
        tool = r.json().get("tools", {}).get(GATE_TOOL, {})
        return bool(tool.get("enabled"))
    except Exception as e:
        logger.warning(f"web tools: registry check failed ({e}) — staying offline")
        return None


def _unwrap_ddg_url(href: str) -> str:
    """DDG wraps result links as /l/?uddg=<urlencoded-url>&rut=..."""
    try:
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            return qs.get("uddg", [href])[0]
    except Exception:
        pass
    return href


class WebSearchConfig(FunctionBaseConfig, name="web_search_duckduckgo"):
    """Search the live web with DuckDuckGo (optional tool, panel-gated)."""
    base_url: str = Field(
        default="https://html.duckduckgo.com/html/",
        description="DuckDuckGo HTML search endpoint",
    )
    robot_api_base_url: str = Field(
        default="http://localhost:7861",
        description="Robot API base URL (optional-tools registry)",
    )
    max_results: int = Field(default=5, description="Number of results to return")


@register_function(config_type=WebSearchConfig)
async def web_search_fn(config: WebSearchConfig, builder: Builder):
    import httpx
    from bs4 import BeautifulSoup

    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=15.0, headers=_HEADERS, follow_redirects=True)

    async def _search(query: str) -> str:
        enabled = await _gate_enabled(client, config.robot_api_base_url)
        if enabled is None:
            return GATE_UNREACHABLE_MSG
        if not enabled:
            return DISABLED_MSG

        try:
            r = await client.get(config.base_url, params={"q": query})
            r.raise_for_status()
        except Exception as e:
            logger.error(f"web search failed: {e}")
            return f"I couldn't reach the internet to search ({e})."

        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for div in soup.select("div.result"):
            if "result--ad" in div.get("class", []):
                continue
            a = div.select_one("a.result__a")
            if a is None or not a.get_text(strip=True):
                continue
            url = _unwrap_ddg_url(a.get("href", ""))
            snippet_el = div.select_one(".result__snippet")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            results.append((a.get_text(strip=True), url, snippet))
            if len(results) >= config.max_results:
                break

        if not results:
            # a 200 with zero results is usually DDG's bot-check page
            return (f"No web results found for '{query}' (the search service "
                    "may be rate-limiting; try again in a moment).")

        return "\n\n".join(
            f"{i}. {title}\n   {url}\n   {snippet}"
            for i, (title, url, snippet) in enumerate(results, 1))

    try:
        yield FunctionInfo.from_fn(
            _search,
            description=(
                "Search the live internet with DuckDuckGo for current events, "
                "news, weather, prices, or anything not in the offline "
                "Wikipedia. Input is a search query string; returns numbered "
                "results with titles, URLs and snippets. Use read_web_page to "
                "open a promising result URL."
            ),
        )
    finally:
        await client.aclose()


class WebReadPageConfig(FunctionBaseConfig, name="web_read_page"):
    """Fetch and read a web page's text (optional tool, panel-gated)."""
    robot_api_base_url: str = Field(
        default="http://localhost:7861",
        description="Robot API base URL (optional-tools registry)",
    )
    max_chars: int = Field(default=6000, description="Truncate extracted text here")


@register_function(config_type=WebReadPageConfig)
async def web_read_page_fn(config: WebReadPageConfig, builder: Builder):
    import httpx
    from bs4 import BeautifulSoup

    client = httpx.AsyncClient(timeout=20.0, headers=_HEADERS, follow_redirects=True)

    async def _read(url: str) -> str:
        enabled = await _gate_enabled(client, config.robot_api_base_url)
        if enabled is None:
            return GATE_UNREACHABLE_MSG
        if not enabled:
            return DISABLED_MSG

        url = url.strip().strip("'\"")
        if not url.lower().startswith(("http://", "https://")):
            return "That doesn't look like a web URL I can open (need http/https)."

        try:
            r = await client.get(url)
            r.raise_for_status()
        except Exception as e:
            logger.error(f"web page read failed for {url}: {e}")
            return (f"That page couldn't be fetched ({e}). Try a different "
                    "result URL, or answer from the search snippets you "
                    "already have.")

        ctype = r.headers.get("content-type", "")
        if "html" not in ctype and not ctype.startswith("text/"):
            return f"That page isn't readable text (content-type {ctype or 'unknown'})."

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header",
                         "footer", "aside", "form", "svg"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        lines = [ln.strip() for ln in root.get_text("\n").splitlines()]
        text = "\n".join(ln for ln in lines if ln)
        if not text:
            return ("That page had no readable text (it may need JavaScript). "
                    "Try a different result URL, or answer from the search "
                    "snippets you already have.")
        if len(text) > config.max_chars:
            text = text[:config.max_chars] + "\n...(truncated)"
        return text

    try:
        yield FunctionInfo.from_fn(
            _read,
            description=(
                "Fetch a web page and return its readable text. Input is a "
                "full URL, usually taken from a previous web_search result. "
                "Use it to read details behind a search result."
            ),
        )
    finally:
        await client.aclose()
