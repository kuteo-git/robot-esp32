"""
r1_watchdog — auto-recovers a PHICOMM R1 running the self-built Android app when it crashes/hangs.
The app can die during state transitions; this watchdog detects that and RESTARTS THE APP over the
port-8080 shell (independent of the app, so it keeps working even when the app is dead).

MULTIPLE SPEAKERS: each IP is watched independently (its own state), so one failing device only
restarts that device. Configure via env R1_IPS="ip1,ip2,..." (or a single R1_IP). Required — set it
to your speaker's LAN address(es); there is no default.

Detecting "app dead": the R1 still answers PING (Android is up) but the app process is gone. Liveness
is checked by looking for the app's process via `ps` over the shell on ws://<ip>:8080 (subprotocol v1).
Recovery: `am force-stop <pkg>; am start -n <pkg>/<activity>` over the same shell.
  (No device reboot: the uid=system shell is blocked by SELinux from reboot/setprop; an app restart
  brings it back in ~4s.)
"""
import os
import re
import time
import json
import asyncio
import subprocess
from datetime import datetime

import websockets

R1_IPS = [ip.strip() for ip in os.environ.get(
    "R1_IPS", os.environ.get("R1_IP", "")).split(",") if ip.strip()]
CHECK_SEC = int(os.environ.get("R1_WATCHDOG_CHECK_SEC", "30"))
COOLDOWN_SEC = int(os.environ.get("R1_WATCHDOG_COOLDOWN_SEC", "120"))  # after a restart, wait for the app to re-init
FAIL_THRESHOLD = int(os.environ.get("R1_WATCHDOG_FAILS", "2"))         # consecutive dead checks before recovering
# Self-built Android app package/activity (override via env if you renamed them).
APP_PACKAGE = os.environ.get("R1_APP_PACKAGE", "info.dourok.voicebot.dev")
APP_ACTIVITY = os.environ.get("R1_APP_ACTIVITY", "info.dourok.voicebot.MainActivity")
# Device reboot is blocked (uid=system, SELinux). Restart the app via `am` (verified, ~4s recovery).
RECOVER_CMD = f"am force-stop {APP_PACKAGE}; am start -n {APP_PACKAGE}/{APP_ACTIVITY}"


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} [r1-watchdog] {msg}", flush=True)


def ping_ok(ip):
    try:
        return subprocess.run(["ping", "-c1", "-t2", ip], capture_output=True).returncode == 0
    except Exception:
        return False


async def _shell(ip, cmd, drain=6):
    async with websockets.connect(f"ws://{ip}:8080", subprotocols=["v1"], open_timeout=6) as ws:
        await ws.send(json.dumps({"type": "shell", "type_id": "myshell", "shell": cmd}))
        out, end = [], asyncio.get_event_loop().time() + drain
        while asyncio.get_event_loop().time() < end:
            try:
                m = await asyncio.wait_for(ws.recv(), end - asyncio.get_event_loop().time())
            except Exception:
                break
            try:
                out.append(json.loads(m).get("data", ""))
            except Exception:
                out.append(str(m))
        return "\n".join(out)


def app_alive(ip):
    # The self-built app has no HTTP control port -> check for its PROCESS via the shell on 8080.
    try:
        out = asyncio.run(_shell(ip, f"echo A=$(ps | grep {APP_PACKAGE} | grep -v grep | busybox wc -l)", 8))
        m = re.search(r"A=(\d+)", out)
        return bool(m) and int(m.group(1)) > 0
    except Exception:
        return False


async def _send_recover(ip):
    async with websockets.connect(f"ws://{ip}:8080", subprotocols=["v1"], open_timeout=6) as ws:
        await ws.send(json.dumps({"type": "shell", "type_id": "myshell", "shell": RECOVER_CMD}))
        try:
            return (await asyncio.wait_for(ws.recv(), 5)).strip()
        except Exception:
            return ""


def recover_app(ip):
    """Restart the app on one speaker over the shell on 8080 (force-stop + am start). ~4s recovery."""
    try:
        out = asyncio.run(_send_recover(ip))
        return "Starting: Intent" in out or out != ""
    except Exception as e:
        log(f"[{ip}] app restart FAILED (shell 8080 unreachable?): {e}")
        return False


def main():
    if not R1_IPS:
        log("no speaker IPs configured — set R1_IPS=\"ip1,ip2,...\" (or R1_IP) and restart. Exiting.")
        return
    log(f"starting — speakers={R1_IPS}, check={CHECK_SEC}s, cooldown={COOLDOWN_SEC}s, threshold={FAIL_THRESHOLD}")
    now0 = time.time()
    # per-speaker state; initial grace = cooldown (avoid acting during boot/recovery)
    state = {ip: {"fails": 0, "last_recover": now0} for ip in R1_IPS}
    log(f"initial grace {COOLDOWN_SEC}s before watching")
    while True:
        time.sleep(CHECK_SEC)
        for ip in R1_IPS:
            st = state[ip]
            if time.time() - st["last_recover"] < COOLDOWN_SEC:
                continue  # just recovered this one -> wait out the cooldown

            if not ping_ok(ip):
                if st["fails"]:
                    log(f"[{ip}] no ping (power/wifi down?) — can't recover, waiting")
                st["fails"] = 0
                continue

            if app_alive(ip):
                if st["fails"]:
                    log(f"[{ip}] app is back — resetting counter")
                st["fails"] = 0
                continue

            # ping OK but the app process is gone -> suspected crash
            st["fails"] += 1
            log(f"[{ip}] app DEAD (ping OK, process not found) — {st['fails']}/{FAIL_THRESHOLD}")
            if st["fails"] >= FAIL_THRESHOLD:
                log(f"[{ip}] => RESTARTING APP over shell 8080 (am force-stop + am start)")
                ok = recover_app(ip)
                log(f"[{ip}] restart sent, waiting for app init (~a few seconds)" if ok else f"[{ip}] RESTART FAILED")
                st["last_recover"] = time.time()
                st["fails"] = 0


if __name__ == "__main__":
    main()
