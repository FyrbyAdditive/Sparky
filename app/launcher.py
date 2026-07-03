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
_local_uv = Path.home() / ".local/bin/uv"
UV = str(_local_uv) if _local_uv.exists() else (shutil.which("uv") or "uv")

children: list[tuple[str, subprocess.Popen]] = []
paused_spark_bot = {"host": None, "user": None}


def ask(prompt: str, default_yes: bool = True) -> bool:
    """Prompt a human; take the default when stdin isn't interactive."""
    if not sys.stdin.isatty():
        say(f"{prompt} -> auto '{'Y' if default_yes else 'N'}' (non-interactive)")
        return default_yes
    ans = input(prompt).strip().lower()
    if not ans:
        return default_yes
    return ans in ("y", "yes")


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
            env[k] = v.strip().strip("\'\"")
    return env


def robot_present() -> bool:
    # The synced venv's python answers in well under a second; a cold
    # `uv run` first resolves the whole environment (seconds at launch).
    venv_python = REPO / "bot" / ".venv" / "bin" / "python"
    if venv_python.exists():
        probe_cmd = [str(venv_python), "-c"]
    else:
        probe_cmd = [UV, "run", "--project", str(REPO / "bot"), "python", "-c"]
    try:
        result = subprocess.run(
            probe_cmd +
            ["import pyaudio; pa = pyaudio.PyAudio(); "
             "print(any('reachy' in str(pa.get_device_info_by_index(i).get('name','')).lower() "
             "for i in range(pa.get_device_count())))"],
            capture_output=True, text=True, timeout=60, cwd=REPO / "bot",
            stdin=subprocess.DEVNULL,
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
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
        )
        if r.stdout.strip() != "active":
            return
    except Exception:
        say(f"Couldn't check {ssh_target} over SSH — if the Spark bot is running, stop it manually.")
        return

    if ask(f"  The Spark's bot is running on {ssh_target}. Pause it while you use the robot here? [Y/n]: "):
        subprocess.run(["ssh", "-o", "BatchMode=yes", ssh_target,
                        "systemctl --user stop sparky-bot reachy-daemon"],
                       timeout=20, stdin=subprocess.DEVNULL)
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


def start_child(name, cmd, cwd, health_url, timeout=180, extra_env=None) -> subprocess.Popen:
    say(f"Starting {name}...")
    log = open(CONFIG_DIR / f"{name}.log", "a")
    child_env = dict(os.environ)
    if extra_env:
        child_env.update(extra_env)
    # New session per child: the real daemon/nat/bot are *grandchildren*
    # under the uv wrapper, so teardown must signal the whole process group
    # or they survive the launcher (two-instance hazard).
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                            env=child_env, stdin=subprocess.DEVNULL,
                            start_new_session=True)
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


def _signal_group(proc: subprocess.Popen, sig: signal.Signals):
    """Signal the child's whole process group (uv wrapper + grandchildren)."""
    try:
        os.killpg(proc.pid, sig)  # start_new_session makes pgid == child pid
    except (ProcessLookupError, PermissionError):
        if proc.poll() is None:
            proc.send_signal(sig)


def stop_children():
    for name, proc in reversed(children):
        if proc.poll() is None:
            say(f"Stopping {name}...")
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
    children.clear()


def sweep_leftovers():
    """Kill any stray trio processes from previous runs — duplicate
    instances fight over the robot and can grab the wrong audio device."""
    import re

    patterns = ["reachy_mini.daemon", "bot/main.py", "python main.py", "nat serve"]
    out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    me = os.getpid()
    for line in out.splitlines():
        if any(p in line for p in patterns) and "launcher.py" not in line:
            m = re.match(r"\s*(\d+)", line)
            if m and int(m.group(1)) != me:
                say(f"Sweeping leftover process {m.group(1)}: {line.strip()[:80]}")
                try:
                    os.kill(int(m.group(1)), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main():
    say("Sparky remote client")
    sweep_leftovers()
    if not ENV_FILE.exists():
        wizard()
    env = read_env()

    if not robot_present():
        say("⚠ No Reachy Mini detected on this machine (USB).")
        say("  Sound only goes in and out of the robot - without it there is no voice.")
        say("  Plug the robot in and relaunch (or continue for panel-only testing).")
        if not ask("  Continue anyway? [y/N]: ", default_yes=False):
            sys.exit(1)

    if env.get("SPARK_SSH"):
        spark_handoff(env["SPARK_SSH"])
    atexit.register(restore_spark_bot)
    atexit.register(stop_children)

    def _on_sigterm(*_):
        # SystemExit unwinds the main loop and runs the atexit teardown
        # (stop_children then restore_spark_bot) deterministically.
        say("SIGTERM — shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    env_file = str(ENV_FILE)
    # --deactivate-audio: the BOT owns the robot's mic/speaker exclusively;
    # without it the daemon plays wake sounds through the host's DEFAULT
    # output — i.e. the computer's speakers on a laptop.
    start_child("daemon",
                [UV, "run", "-m", "reachy_mini.daemon.app.main",
                 "--no-localhost-only", "--deactivate-audio"],
                REPO / "bot", "http://127.0.0.1:8000/")
    start_child("nat",
                [UV, "run", "nat", "serve",
                 "--config_file", "src/ces_tutorial/config.yml", "--port", "8001"],
                REPO / "nat", "http://127.0.0.1:8001/docs", extra_env=env)
    start_child("bot",
                [UV, "run", "python", "main.py"],
                REPO / "bot", "http://127.0.0.1:7861/health", extra_env=env)

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
                    # SAFETY: never auto-restart into live hardware. A crashed
                    # component means an unknown robot state — stop the world
                    # and let the human relaunch deliberately.
                    raise RuntimeError(
                        f"{name} exited unexpectedly — stopping everything for safety. "
                        f"See {CONFIG_DIR}/{name}.log, then relaunch Sparky.")
    except KeyboardInterrupt:
        say("Shutting down...")


if __name__ == "__main__":
    main()
