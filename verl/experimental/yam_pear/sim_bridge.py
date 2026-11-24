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
"""Simulator bridge: one YAM-station episode behind a line-delimited JSON protocol on stdin/stdout.

This file is STANDALONE: standard library only, no verl imports, python >= 3.10. It is meant to be copied
to (or run in place from) a simulator's own virtual environment, which is not the verl environment:

    <sim venv>/bin/python sim_bridge.py --backend driver --driver-dir <dir with raiden_pear.py> \
        --driver-module raiden_pear
    python sim_bridge.py --backend fake            # deterministic pure-Python simulator, for tests

One process serves one episode at a time. Every request is one JSON object on one line; every reply is one
JSON object on one line with ``"ok": true`` or ``{"ok": false, "error": "..."}``.

    {"cmd": "reset", "sim": "raiden", "task": "pick_cube", "seed": 7, "prepose": "0,6,grasped",
     "episode_args": {...}}
        -> {"ok": true, "instruction": str, "prepose": any, "observation": OBS,
            "reward": 0.0, "success": false, "done": false}
    {"cmd": "step", "action": {"left_arm": {...}, "right_arm": {...}, "left_gripper": {...}, ...}}
        -> {"ok": true, "feedback": str | null, "result": {...}, "observation": OBS,
            "reward": float, "success": bool, "done": bool}
    {"cmd": "close"}
        -> {"ok": true}            (the process then exits)

``OBS`` is ``{"state": {part: {quantity: value}}, "images": {camera: base64 JPEG}}`` with the cameras in the
order the station sends them. ``reward`` is the episode's best shaped reward so far (1 = success), i.e. the
driver's ``Episode.max_reward``. ``prepose`` is the start state for the scripted expert (``"R,H[,stage]"``),
or ``"x"`` / ``""`` for the station's Home.

The simulators and their Python packages write to stdout freely, so the protocol does not use file
descriptor 1: at start the real stdout is duplicated to a private descriptor and fd 1 is pointed at stderr.
"""

import argparse
import base64
import importlib
import json
import math
import os
import random
import sys
import traceback
from types import SimpleNamespace

PROTOCOL_VERSION = 1
ARMS = ("left", "right")
ARM_Q = ("x", "y", "z", "roll", "pitch", "yaw")
CAMERAS = ("head", "left_wrist", "right_wrist")
NEXT_OBSERVATION = " Use the new measured observation."

# A 32x32 grey JPEG: the fake simulator's camera frame (so that it needs no imaging library).
_FAKE_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDABALDA4MChAODQ4SERATGCgaGBYWGDEjJR0oOjM9PDkzODdASFxOQERXRTc4UG1RV19iZ2hn"
    "Pk1xeXBkeFxlZ2P/2wBDARESEhgVGC8aGi9jQjhCY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2Nj"
    "Y2NjY2P/wAARCAAgACADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9"
    "AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZn"
    "aGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP0"
    "9fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMi"
    "MoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaH"
    "iImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEA"
    "PwCrRRRWpmFFFFABRRRQAUUUUAf/2Q=="
)


class FakeEpisode:
    """A deterministic stand-in for a driver ``Episode``, with the same surface the bridge uses.

    The scene is one number: how many 5 cm descents of the left arm are still missing. An action whose
    ``left_arm.z`` is at most -5 makes one descent; anything else is carried out without progress and yields
    station-style feedback. The distance from Home is fixed by (task, seed); a prepose start state shortens
    it the way the skill ladder does (``grasped`` leaves one descent, ``around`` two, ``R,H`` more as R grows).
    The shaped reward is the fraction of the distance covered, the success reward is 1.
    """

    def __init__(self, args):
        rng = random.Random(f"{args.task}:{args.seed}")
        self.total = rng.randint(4, 8)
        prepose = getattr(args, "prepose", "") or ""
        self.prepose = None
        self.remaining = self.total
        if prepose:
            parts = prepose.split(",")
            radius, height = float(parts[0]), float(parts[1])
            stage = parts[2] if len(parts) > 2 else "above"
            if stage == "grasped":
                self.remaining = 1
            elif stage == "around":
                self.remaining = 2
            elif radius >= 0:
                self.remaining = min(self.total, 3 + int(radius // 4))
            self.prepose = {"radius": radius, "height": height, "stage": stage}
        self.start = self.remaining
        self.instruction = f"pick up the cube ({args.task})"
        self.success = False
        self.max_reward = 0.0
        self.steps = 0

    def quantities(self):
        """The station's state: every part's quantities (cm, rad)."""
        out = {}
        for i, arm in enumerate(ARMS):
            z = 5.0 * self.remaining if i == 0 else 20.0
            out[arm + "_arm"] = {"x": 30.0, "y": 10.0 - 20.0 * i, "z": z, "roll": 0.0, "pitch": 1.57, "yaw": 0.0}
        for arm in ARMS:
            out[arm + "_gripper"] = {"width": 9.5}
        return out

    def images(self):
        """Camera name -> JPEG bytes."""
        data = base64.b64decode(_FAKE_JPEG_B64)
        return {cam: data for cam in CAMERAS}

    def act(self, action):
        """Carry out a delta action; returns (feedback text or None, result)."""
        self.steps += 1
        dz = float(action["left_arm"]["z"])
        if dz <= -5.0 and self.remaining > 0:
            self.remaining -= 1
            covered = self.start - self.remaining
            self.max_reward = max(self.max_reward, covered / self.start)
            if self.remaining == 0:
                self.success = True
                self.max_reward = 1.0
            return None, {"status": "completed", "settled": True, "reached": True, "grippers": {}, "done": self.success}
        feedback = {"status": "completed", "execution_requested": True, "settled": True, "reached": False}
        result = {"status": "completed", "settled": True, "reached": False, "grippers": {}, "done": False}
        return json.dumps(feedback) + NEXT_OBSERVATION, result


class FakeBackend:
    """Builds :class:`FakeEpisode` objects."""

    def make(self, request):
        """Create the episode for a reset request."""
        if not isinstance(request.get("seed"), int):
            raise ValueError("reset needs an integer seed")
        prepose = request.get("prepose") or ""
        return FakeEpisode(
            SimpleNamespace(task=request["task"], seed=request["seed"], prepose="" if prepose == "x" else prepose)
        )

    def encode_images(self, episode):
        """Camera name -> base64 JPEG."""
        return {cam: base64.b64encode(data).decode() for cam, data in episode.images().items()}


class DriverBackend:
    """Wraps the ``Episode`` class of an existing station driver (``raiden_pear.py``, ``abc_pear_cur.py``)."""

    def __init__(self, driver_dir, driver_module):
        if driver_dir:
            sys.path.insert(0, os.path.abspath(driver_dir))
        self.module = importlib.import_module(driver_module)

    def make(self, request):
        """Create the driver's Episode with the arguments its command line would have given it."""
        prepose = request.get("prepose") or ""
        args = {
            "task": request["task"],
            "seed": int(request["seed"]),
            "prepose": "" if prepose == "x" else prepose,
            "start": "home",
            "randomization": "",
            "grasp_depth": -1.0,
            "grasp_yaw_offset": 0.0,
            "lift": 0.0,
            "lift_pick": -1,
            "record_dense": False,
        }
        args.update(request.get("episode_args") or {})
        return self.module.Episode(argparse.Namespace(**args))

    def encode_images(self, episode):
        """Camera name -> base64 JPEG, encoded exactly as the driver encodes what it sends the model."""
        return {
            cam: base64.b64encode(self.module.jpeg_bytes(image)).decode() for cam, image in episode.images().items()
        }


def _finite_reward(value):
    value = float(value or 0.0)
    return value if math.isfinite(value) else 0.0


def serve(backend, reader, writer):
    """Run the request loop until ``close`` or end of input."""
    episode = None

    def observation():
        return {"state": episode.quantities(), "images": backend.encode_images(episode)}

    def status():
        success = bool(episode.success)
        return {"reward": _finite_reward(episode.max_reward), "success": success, "done": success}

    for line in reader:
        line = line.strip()
        if not line:
            continue
        closing = False
        try:
            request = json.loads(line)
            cmd = request.get("cmd")
            if cmd == "hello":
                reply = {"ok": True, "protocol": PROTOCOL_VERSION}
            elif cmd == "reset":
                episode = backend.make(request)
                reply = {
                    "ok": True,
                    "instruction": episode.instruction,
                    "prepose": getattr(episode, "prepose", None),
                    "observation": observation(),
                    **status(),
                }
            elif cmd == "step":
                if episode is None:
                    raise RuntimeError("step before reset")
                feedback, result = episode.act(request["action"])
                reply = {"ok": True, "feedback": feedback, "result": result, "observation": observation(), **status()}
            elif cmd == "close":
                reply, closing = {"ok": True}, True
            else:
                raise ValueError(f"unknown cmd {cmd!r}")
        except Exception as exc:  # the caller decides what an error means; the bridge keeps serving
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
        writer.write(json.dumps(reply, default=str) + "\n")
        writer.flush()
        if closing:
            return


def main(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backend", choices=("fake", "driver"), default="driver")
    parser.add_argument("--driver-dir", default="", help="directory holding the driver module (put on sys.path)")
    parser.add_argument("--driver-module", default="raiden_pear", help="module that defines Episode and jpeg_bytes")
    args = parser.parse_args(argv)

    # Keep the protocol channel private: anything the simulator prints goes to stderr.
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1, encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    backend = FakeBackend() if args.backend == "fake" else DriverBackend(args.driver_dir, args.driver_module)
    serve(backend, sys.stdin, protocol_out)


if __name__ == "__main__":
    main()
