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

import asyncio
import logging
import os
import time
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
    """Search the live web (optional tool, panel-gated). Primary engine is
    the self-hosted SearXNG aggregator; direct DuckDuckGo/Mojeek scraping
    remains as fallback when SearXNG is down."""
    searxng_url: str = Field(
        default="http://magi:8888",
        description="Self-hosted SearXNG base URL (JSON API)",
    )
    base_url: str = Field(
        default="https://html.duckduckgo.com/html/",
        description="DuckDuckGo HTML search endpoint (fallback)",
    )
    robot_api_base_url: str = Field(
        default="http://localhost:7861",
        description="Robot API base URL (optional-tools registry)",
    )
    max_results: int = Field(default=5, description="Number of results to return")


# Fallback engine: DDG's anomaly detection flags an IP after bursts of
# automated queries (observed live: HTTP 202 challenge pages from every
# DDG endpoint after one chatty evening). Mojeek runs its own independent
# index, tolerates polite scraping, and serves direct result URLs.
_MOJEEK_URL = "https://www.mojeek.com/search"

# Client-side politeness (what got us flagged: the ReAct agent fired four
# searches inside one turn). Same practice as the ddgs library's ~1-2s
# inter-request throttle, plus a short-TTL cache so retried/repeated
# queries within a conversation never hit the engines twice. Shared
# across web_search AND read_web_page (one outbound budget).
_MIN_INTERVAL_SECS = float(os.getenv("SEARCH_MIN_INTERVAL_SECS", "1.5"))
_CACHE_TTL_SECS = float(os.getenv("SEARCH_CACHE_TTL_SECS", "300"))
_CACHE_MAX = 64
_throttle_lock = asyncio.Lock()
_last_request_ts = 0.0
_result_cache: dict[str, tuple[float, str]] = {}


async def _polite_slot():
    """Serialize outbound requests and enforce the minimum interval."""
    global _last_request_ts
    async with _throttle_lock:
        wait = _last_request_ts + _MIN_INTERVAL_SECS - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_ts = time.monotonic()


def _cache_get(key: str) -> str | None:
    hit = _result_cache.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL_SECS:
        return hit[1]
    return None


def _cache_put(key: str, value: str):
    if len(_result_cache) >= _CACHE_MAX:
        oldest = min(_result_cache, key=lambda k: _result_cache[k][0])
        del _result_cache[oldest]
    _result_cache[key] = (time.monotonic(), value)


@register_function(config_type=WebSearchConfig)
async def web_search_fn(config: WebSearchConfig, builder: Builder):
    import httpx
    from bs4 import BeautifulSoup

    # one client for the workflow's lifetime — per-call clients redo TCP setup
    client = httpx.AsyncClient(timeout=15.0, headers=_HEADERS, follow_redirects=True)

    async def _searxng_results(query: str) -> list[tuple[str, str, str]]:
        # aggregates many engines server-side and returns clean JSON — no
        # HTML scraping, and immune to any single engine blocking us
        r = await client.get(f"{config.searxng_url.rstrip('/')}/search",
                             params={"q": query, "format": "json"})
        r.raise_for_status()
        return [(res.get("title", ""), res.get("url", ""),
                 res.get("content", "") or "")
                for res in r.json().get("results", [])[:config.max_results]
                if res.get("url")]

    async def _ddg_results(query: str) -> list[tuple[str, str, str]]:
        r = await client.get(config.base_url, params={"q": query})
        # DDG signals "prove you're human" with a 202 challenge page (and
        # sometimes a 200 with no result nodes) — treat both as no results
        # so the fallback engine takes over
        if r.status_code != 200:
            logger.warning(f"web search: DDG returned {r.status_code}, "
                           "likely rate-limited — trying fallback engine")
            return []
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
        return results

    async def _mojeek_results(query: str) -> list[tuple[str, str, str]]:
        r = await client.get(_MOJEEK_URL, params={"q": query})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        results = []
        for li in soup.select("ul.results-standard li"):
            a = li.select_one("h2 a") or li.find("a")
            if a is None or not a.get_text(strip=True):
                continue
            snippet_el = li.select_one("p.s")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            results.append((a.get_text(strip=True), a.get("href", ""), snippet))
            if len(results) >= config.max_results:
                break
        return results

    async def _search(query: str) -> str:
        enabled = await _gate_enabled(client, config.robot_api_base_url)
        if enabled is None:
            return GATE_UNREACHABLE_MSG
        if not enabled:
            return DISABLED_MSG

        cache_key = f"search:{query.strip().lower()}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        await _polite_slot()

        results = []
        errors = []
        for name, engine in (("SearXNG", _searxng_results),
                             ("DuckDuckGo", _ddg_results),
                             ("Mojeek", _mojeek_results)):
            try:
                results = await engine(query)
            except Exception as e:
                logger.error(f"web search via {name} failed: {e}")
                errors.append(f"{name}: {e}")
            if results:
                break

        if not results:
            if errors and len(errors) == 3:
                return f"I couldn't reach the internet to search ({errors[0]})."
            return (f"No web results found for '{query}' right now (the "
                    "search services may be rate-limiting; try again in a "
                    "few minutes, or answer from what you already know).")

        out = "\n\n".join(
            f"{i}. {title}\n   {url}\n   {snippet}"
            for i, (title, url, snippet) in enumerate(results, 1))
        _cache_put(cache_key, out)
        return out

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

        cache_key = f"read:{url}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached
        await _polite_slot()

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
        _cache_put(cache_key, text)
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
