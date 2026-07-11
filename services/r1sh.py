#!/usr/bin/env python3
"""Run a single shell command on the R1 speaker over the WebSocket shell on port 8080 (uid=system).

The R1 has no convenient adb-shell and no wget/curl/busybox, but it does expose a WebSocket shell on
port 8080 (subprotocol "v1"). Use it to: install apks (`pm install`), recover the app (`am ...`),
read state (`dumpsys`, `ps`, `getprop`), etc.

  python services/r1sh.py '<command>' [drain_seconds]   # drain defaults to 8s (pm install needs >45)
  R1_IP=<speaker-ip> python services/r1sh.py 'id'

Requires: python with `websockets` (use services/.venv/bin/python). See SETUP.md.
The variant used inside the watchdog is r1_watchdog.py::_shell.
"""
import asyncio
import json
import os
import sys

import websockets

R1_IP = os.environ.get("R1_IP", "")  # set R1_IP env to your speaker's LAN address


async def run(cmd: str, drain: float) -> str:
    async with websockets.connect(
        f"ws://{R1_IP}:8080", subprotocols=["v1"], open_timeout=6
    ) as ws:
        await ws.send(json.dumps({"type": "shell", "type_id": "myshell", "shell": cmd}))
        out, end = [], asyncio.get_event_loop().time() + drain
        while asyncio.get_event_loop().time() < end:
            try:
                m = await asyncio.wait_for(ws.recv(), end - asyncio.get_event_loop().time())
            except asyncio.TimeoutError:
                break
            try:
                out.append(json.loads(m).get("data", ""))
            except Exception:
                out.append(str(m))
        return "".join(out)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    if not R1_IP:
        print("Set the R1_IP environment variable to your speaker's LAN address, e.g.\n"
              "  R1_IP=192.168.1.50 python services/r1sh.py 'id'", file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    drain = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    print(asyncio.run(run(cmd, drain)), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
