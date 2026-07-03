#!/usr/bin/env python3
"""Sparky remote-client launcher.

Plug the Reachy Mini into this machine (macOS or Linux), run this, talk.
Starts the same trio that runs on the Spark robot host — reachy daemon,
NAT agent server, and the bot — with all inference endpoints pointing at
the Sparks over the LAN. Opens the control panel in your browser.

First run walks through a tiny wizard and writes ~/.sparky/remote.env.
If the Spark's own bot is running (it can't work without the robot), the
launcher offers to pause it over SSH and restores it on quit.

Run me via:  uv run --project bot python app/launcher.py
(or just double-click Sparky.app / the sparky desktop entry).
"""

import atexit
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path.home() / ".sparky"
ENV_FILE = CONFIG_DIR / "remote.env"
TEMPLATE = REPO / "deploy" / "profiles" / "remote-client.bot.env"
UV = shutil.which("uv") or str(Path.home() / ".local/bin/uv")

children: list[tuple[str, subprocess.Popen]] = []
paused_spark_bot = {"host": None, "user": None}


def say(msg):
    print(f"\033[1;36m[sparky]\033[0m {msg}")


def http_ok(url, timeout=3) -> bool:
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def tcp_ok(host, port, timeout=3) -> bool:
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def wizard():
    say("First-time setup — where are your Sparks?")
    audio_host = input("  Spark hosting speech services (Riva/Kokoro) [magi]: ").strip() or "magi"
    llm_host = input("  Spark hosting the LLM + wiki [shodan]: ").strip() or "shodan"
    ssh_user = input(f"  SSH user on {audio_host} for handoff [tim]: ").strip() or "tim"

    say("Probing endpoints...")
    checks = [
        ("Riva STT", tcp_ok(audio_host, 50051)),
        ("Kokoro TTS", http_ok(f"http://{audio_host}:8880/v1/models")),
        ("Router LLM", http_ok(f"http://{audio_host}:8030/health")),
        ("Agent LLM", http_ok(f"http://{llm_host}:8010/health")),
        ("Wikipedia", http_ok(f"http://{llm_host}:8040/health")),
    ]
    for name, ok in checks:
        print(f"    {'✓' if ok else '✗'} {name}")
    if not all(ok for _, ok in checks):
        say("Some services are unreachable — you can continue, but those features will fail.")
        if input("  Continue anyway? [y/N]: ").strip().lower() != "y":
            sys.exit(1)

    CONFIG_DIR.mkdir(exist_ok=True)
    env = TEMPLATE.read_text().replace("@AUDIO_HOST@", audio_host).replace("@LLM_HOST@", llm_host)
    env += f"\n# handoff\nSPARK_SSH={ssh_user}@{audio_host}\n"
    ENV_FILE.write_text(env)
    say(f"Config written to {ENV_FILE}")


def read_env() -> dict:
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def robot_present() -> bool:
    try:
        result = subprocess.run(
            [UV, "run", "--project", str(REPO / "bot"), "python", "-c",
             "import pyaudio; pa = pyaudio.PyAudio(); "
             "print(any('reachy' in str(pa.get_device_info_by_index(i).get('name','')).lower() "
             "for i in range(pa.get_device_count())))"],
            capture_output=True, text=True, timeout=60, cwd=REPO / "bot",
        )
        return "True" in result.stdout
    except Exception:
        return False


def spark_handoff(ssh_target: str):
    """Pause the Spark's bot if it's running there (it has no robot now)."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_target,
             "systemctl --user is-active sparky-bot"],
            capture_output=True, text=True, timeout=15,
        )
        if r.stdout.strip() != "active":
            return
    except Exception:
        say(f"Couldn't check {ssh_target} over SSH — if the Spark bot is running, stop it manually.")
        return

    ans = input(f"  The Spark's bot is running on {ssh_target}. Pause it while you use the robot here? [Y/n]: ")
    if ans.strip().lower() in ("", "y", "yes"):
        subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_target,
                        "systemctl --user stop sparky-bot reachy-daemon"], timeout=20)
        user, host = ssh_target.split("@")
        paused_spark_bot.update(host=host, user=user)
        say(f"Paused the bot on {host} — it will be restored when you quit.")


def restore_spark_bot():
    if paused_spark_bot["host"]:
        target = f"{paused_spark_bot['user']}@{paused_spark_bot['host']}"
        say(f"Restoring the bot on {paused_spark_bot['host']}...")
        subprocess.run(["ssh", "-o", "BatchMode=yes", target,
                        "systemctl --user start reachy-daemon sparky-bot"], timeout=30)
        paused_spark_bot["host"] = None


def start_child(name, cmd, cwd, health_url, timeout=180) -> subprocess.Popen:
    say(f"Starting {name}...")
    log = open(CONFIG_DIR / f"{name}.log", "a")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    children.append((name, proc))
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited during startup — see {CONFIG_DIR}/{name}.log")
        if http_ok(health_url):
            say(f"  {name} ready")
            return proc
        time.sleep(2)
    raise RuntimeError(f"{name} did not become ready — see {CONFIG_DIR}/{name}.log")


def stop_children():
    for name, proc in reversed(children):
        if proc.poll() is None:
            say(f"Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    children.clear()


def main():
    say("Sparky remote client")
    if not ENV_FILE.exists():
        wizard()
    env = read_env()

    if not robot_present():
        say("⚠ No Reachy Mini detected on this machine (USB).")
        say("  Sound only goes in and out of the robot - without it there is no voice.")
        say("  Plug the robot in and relaunch (or continue for panel-only testing).")
        if input("  Continue anyway? [y/N]: ").strip().lower() != "y":
            sys.exit(1)

    if env.get("SPARK_SSH"):
        spark_handoff(env["SPARK_SSH"])
    atexit.register(restore_spark_bot)
    atexit.register(stop_children)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    env_file = str(ENV_FILE)
    start_child("daemon",
                [UV, "run", "-m", "reachy_mini.daemon.app.main", "--no-localhost-only"],
                REPO / "bot", "http://127.0.0.1:8000/")
    start_child("nat",
                [UV, "run", "--env-file", env_file, "nat", "serve",
                 "--config_file", "src/ces_tutorial/config.yml", "--port", "8001"],
                REPO / "nat", "http://127.0.0.1:8001/docs")
    start_child("bot",
                [UV, "run", "--env-file", env_file, "python", "main.py"],
                REPO / "bot", "http://127.0.0.1:7861/health")

    say("All up — opening the control panel. The robot should greet you.")
    webbrowser.open("http://localhost:7861/")
    say("Press Ctrl+C (or close this window) to quit and hand the robot back.")

    # Supervise: restart crashed children (except repeated fast crashes)
    crash_counts: dict[str, int] = {}
    try:
        while True:
            time.sleep(3)
            for i, (name, proc) in enumerate(list(children)):
                if proc.poll() is not None:
                    crash_counts[name] = crash_counts.get(name, 0) + 1
                    if crash_counts[name] > 3:
                        raise RuntimeError(f"{name} keeps crashing — see {CONFIG_DIR}/{name}.log")
                    say(f"⚠ {name} exited — restarting ({crash_counts[name]}/3)")
                    children.pop(i)
                    if name == "daemon":
                        start_child(name, [UV, "run", "-m", "reachy_mini.daemon.app.main",
                                           "--no-localhost-only"], REPO / "bot", "http://127.0.0.1:8000/")
                    elif name == "nat":
                        start_child(name, [UV, "run", "--env-file", env_file, "nat", "serve",
                                           "--config_file", "src/ces_tutorial/config.yml",
                                           "--port", "8001"], REPO / "nat", "http://127.0.0.1:8001/docs")
                    elif name == "bot":
                        start_child(name, [UV, "run", "--env-file", env_file, "python", "main.py"],
                                    REPO / "bot", "http://127.0.0.1:7861/health")
    except KeyboardInterrupt:
        say("Shutting down...")


if __name__ == "__main__":
    main()
