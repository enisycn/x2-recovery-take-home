"""Small local IPC client joining ROS Humble to the Isaac Lab policy process."""

from __future__ import annotations

import json
import socket
from typing import Callable


class IsaacIpcClient:
    """Consume newline-delimited state events from a repo-local Unix socket."""

    def __init__(self, socket_path: str, timeout_sec: float) -> None:
        self.socket_path = socket_path
        self.timeout_sec = float(timeout_sec)

    def run(
        self,
        *,
        seed: int,
        real_time: bool,
        on_step: Callable[[dict], None] | None,
    ) -> dict:
        request = {
            "command": "start",
            "seed": int(seed),
            "timeout_sec": self.timeout_sec,
            "real_time": bool(real_time),
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout_sec + 10.0)
            connection.connect(self.socket_path)
            stream = connection.makefile("rwb")
            stream.write((json.dumps(request) + "\n").encode("utf-8"))
            stream.flush()
            for raw_line in stream:
                event = json.loads(raw_line.decode("utf-8"))
                event_type = event.get("type")
                if event_type == "step":
                    if on_step is not None:
                        on_step(event)
                    continue
                if event_type == "result":
                    return event
                raise RuntimeError(f"unexpected Isaac IPC event: {event_type!r}")
        raise RuntimeError("Isaac policy server disconnected without a result")
