# Copyright 2026 Individual Contributor
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Async client for ``sim_bridge.py``: one simulator subprocess per episode, JSON lines over its pipes."""

import asyncio
import collections
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

BRIDGE_SCRIPT = Path(__file__).resolve().parent / "sim_bridge.py"
STREAM_LIMIT = 256 * 1024 * 1024  # one reply carries three base64 camera frames on a single line


class SimBridgeError(RuntimeError):
    """The bridge process died, timed out, or answered ``ok: false``."""


def fake_sim_command() -> list[str]:
    """The command that runs the fake simulator with the current interpreter."""
    return [sys.executable, str(BRIDGE_SCRIPT), "--backend", "fake"]


class SimBridgeClient:
    """Spawns a bridge process and exchanges requests with it.

    Args:
        command: The bridge command line, e.g. ``[<sim venv python>, "sim_bridge.py", "--backend", "driver", ...]``.
            ``{bridge}`` in an argument is replaced by the path of the recipe's ``sim_bridge.py``.
        cwd: Working directory of the process (the ABC simulator must start in its own directory).
        env: Environment variables added to the current environment (``MUJOCO_GL``, ``PYTHONPATH``, ...);
            a ``None`` value removes the variable.
        timeout: Seconds to wait for one reply (a reset builds the scene and runs the scripted expert).
    """

    def __init__(
        self,
        command: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict[str, Any]] = None,
        timeout: float = 600.0,
    ):
        self.command = [str(part).replace("{bridge}", str(BRIDGE_SCRIPT)) for part in command]
        self.cwd = os.path.expanduser(cwd) if cwd else None
        self.env = dict(os.environ)
        for key, value in (env or {}).items():
            if value is None:
                self.env.pop(str(key), None)
            else:
                self.env[str(key)] = str(value)
        self.timeout = float(timeout)
        self.process: Optional[asyncio.subprocess.Process] = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=40)
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the bridge process."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=self.env,
            limit=STREAM_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while True:
            try:
                line = await self.process.stderr.readline()
            except ValueError:  # an overlong line: drop it
                continue
            if not line:
                return
            self._stderr_tail.append(line.decode(errors="replace").rstrip())

    def _failure(self, what: str) -> SimBridgeError:
        tail = "\n".join(self._stderr_tail)
        return SimBridgeError(f"{what} (command: {' '.join(self.command)})\n--- bridge stderr tail ---\n{tail}")

    async def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its reply; raises :class:`SimBridgeError` on any failure."""
        if self.process is None:
            await self.start()
        assert self.process is not None and self.process.stdin is not None and self.process.stdout is not None
        try:
            self.process.stdin.write((json.dumps(payload) + "\n").encode())
            await self.process.stdin.drain()
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise self._failure(f"no reply to {payload.get('cmd')!r} within {self.timeout:.0f} s") from None
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise self._failure(f"bridge pipe closed during {payload.get('cmd')!r}: {exc}") from None
        if not line:
            await asyncio.sleep(0.05)  # let the stderr reader catch the traceback
            raise self._failure(f"bridge exited during {payload.get('cmd')!r}")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise SimBridgeError(
                f"bridge error on {payload.get('cmd')!r}: {reply.get('error')}\n{reply.get('traceback', '')}"
            )
        return reply

    async def reset(
        self, sim: str, task: str, seed: int, prepose: str, episode_args: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Start an episode: scene from ``seed``, start state from ``prepose`` (``"x"`` = Home)."""
        return await self.request(
            {
                "cmd": "reset",
                "sim": sim,
                "task": task,
                "seed": int(seed),
                "prepose": prepose,
                "episode_args": dict(episode_args or {}),
            }
        )

    async def step(self, action: dict[str, Any]) -> dict[str, Any]:
        """Carry out one action."""
        return await self.request({"cmd": "step", "action": action})

    async def close(self) -> None:
        """Ask the bridge to exit, then make sure the process is gone."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            if process.returncode is None and process.stdin is not None:
                process.stdin.write(b'{"cmd": "close"}\n')
                await process.stdin.drain()
                process.stdin.close()
                await asyncio.wait_for(process.wait(), timeout=10)
        except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if self._stderr_task is not None:
                self._stderr_task.cancel()

    async def __aenter__(self) -> "SimBridgeClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()
