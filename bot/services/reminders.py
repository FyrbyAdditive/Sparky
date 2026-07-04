"""Timers and reminders: local scheduler with spoken announcements.

State lives here (the bot owns the clock and the mouth); the NAT tools
and the panel talk to it through robot_api's /reminders endpoints. Due
items speak "Reminder: <label>" through the direct-TTS path (plays even
mid-turn), chime, and wiggle for attention. Everything persists in
~/.sparky/reminders.json so wall-clock alarms survive restarts.

Times are stored as wall-clock epoch seconds (time.time) — monotonic
clocks don't survive restarts and alarms must.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_FILE = Path.home() / ".sparky" / "reminders.json"

# {id: {"id", "label", "due_ts", "created_ts"}} — plain dict, only touched
# from the robot-api event loop (endpoints + scheduler tick)
REMINDERS: dict[str, dict] = {}
_next_id = {"n": 1}

_DURATION_RE = re.compile(
    r"(?:(\d+)\s*(?:h|hr|hour)s?)?\s*(?:(\d+)\s*(?:m|min|minute)s?)?"
    r"\s*(?:(\d+)\s*(?:s|sec|second)s?)?$", re.I)
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_when(text: str) -> float | None:
    """'10m' / '1h 30m' / '90s' → now+duration; '15:00' → next occurrence.
    Returns a wall-clock due timestamp, or None if unparseable."""
    text = text.strip().lower().replace("in ", "").replace("at ", "")
    m = _CLOCK_RE.match(text)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if hh > 23 or mm > 59:
            return None
        now = datetime.now()
        due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if due <= now:
            due += timedelta(days=1)  # next occurrence
        return due.timestamp()
    m = _DURATION_RE.match(text)
    if m and any(m.groups()):
        h, mi, s = (int(g) if g else 0 for g in m.groups())
        secs = h * 3600 + mi * 60 + s
        return time.time() + secs if secs > 0 else None
    return None


def _save():
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(list(REMINDERS.values()), indent=1))
        os.replace(tmp, _FILE)
    except Exception as e:
        logger.warning(f"reminders: could not persist: {e}")


def load():
    """Restore pending reminders; announce-on-startup any missed while down."""
    try:
        for item in json.loads(_FILE.read_text()):
            if isinstance(item, dict) and "id" in item and "due_ts" in item:
                REMINDERS[item["id"]] = item
                n = int(re.sub(r"\D", "", item["id"]) or 0)
                _next_id["n"] = max(_next_id["n"], n + 1)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"reminders: could not load state: {e}")


def add(label: str, due_ts: float) -> dict:
    rid = f"r{_next_id['n']}"
    _next_id["n"] += 1
    item = {"id": rid, "label": label.strip() or "your reminder",
            "due_ts": due_ts, "created_ts": time.time()}
    REMINDERS[rid] = item
    _save()
    logger.info(f"reminder set: {rid} '{item['label']}' due "
                f"{datetime.fromtimestamp(due_ts):%H:%M:%S}")
    return item


def cancel(query: str) -> dict | None:
    """Cancel by id or label substring (most-recently-created match)."""
    q = query.strip().lower()
    if q in REMINDERS:
        item = REMINDERS.pop(q)
        _save()
        return item
    matches = [r for r in REMINDERS.values() if q in r["label"].lower()]
    if not matches:
        return None
    item = max(matches, key=lambda r: r["created_ts"])
    REMINDERS.pop(item["id"])
    _save()
    return item


def listing() -> list[dict]:
    now = time.time()
    return [{**r, "remaining_secs": max(0, round(r["due_ts"] - now))}
            for r in sorted(REMINDERS.values(), key=lambda r: r["due_ts"])]


async def scheduler(speak, chime, wiggle):
    """1s tick on the robot-api loop. Callbacks keep this module decoupled:
    speak(text), chime(), wiggle() are injected by robot_api."""
    load()
    missed = [r for r in REMINDERS.values() if r["due_ts"] < time.time() - 5]
    while True:
        await asyncio.sleep(1.0)
        try:
            now = time.time()
            if missed:
                names = ", ".join(r["label"] for r in missed)
                speak(f"While I was off I missed reminding you about: {names}.")
                for r in missed:
                    REMINDERS.pop(r["id"], None)
                missed = []
                _save()
            due = [r for r in REMINDERS.values() if r["due_ts"] <= now]
            for r in due:
                REMINDERS.pop(r["id"], None)
                logger.info(f"reminder due: {r['id']} '{r['label']}'")
                chime()
                speak(f"Reminder: {r['label']}.")
                wiggle()
            if due:
                _save()
        except Exception as e:
            logger.warning(f"reminder scheduler: {e}")
