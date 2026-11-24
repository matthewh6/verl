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
"""The simulator bridge protocol, exercised against the fake simulator in a real subprocess."""

import ast
import asyncio
import base64
import io
import json
import subprocess
import sys

import pytest
from PIL import Image

from verl.experimental.yam_pear.sim_client import BRIDGE_SCRIPT, SimBridgeClient, SimBridgeError, fake_sim_command

DESCEND = {
    "left_arm": {"x": 0, "y": 0, "z": -5, "roll": 0, "pitch": 0, "yaw": 0},
    "right_arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
    "left_gripper": {"width": 0},
    "right_gripper": {"width": 0},
}
STAY = {**DESCEND, "left_arm": {**DESCEND["left_arm"], "z": 0}}


def test_bridge_is_standalone():
    """The bridge runs inside a simulator's venv: standard library only, no verl."""
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(ast.parse(BRIDGE_SCRIPT.read_text())):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert name.split(".")[0] in stdlib, f"sim_bridge.py imports {name}"


@pytest.mark.asyncio
async def test_reset_observation_and_determinism():
    replies = []
    for _ in range(2):
        async with SimBridgeClient(fake_sim_command(), timeout=30) as client:
            replies.append(await client.reset("fake", "pick_cube", 7, "x"))
    first = replies[0]
    assert first == replies[1]
    assert first["ok"] and first["reward"] == 0.0 and first["success"] is False and first["done"] is False
    assert isinstance(first["instruction"], str) and first["instruction"]
    obs = first["observation"]
    assert set(obs["state"]) == {"left_arm", "right_arm", "left_gripper", "right_gripper"}
    assert list(obs["images"]) == ["head", "left_wrist", "right_wrist"]
    for data in obs["images"].values():
        image = Image.open(io.BytesIO(base64.b64decode(data)))
        assert image.format == "JPEG"


@pytest.mark.asyncio
async def test_step_reward_success_and_feedback():
    async with SimBridgeClient(fake_sim_command(), timeout=30) as client:
        reset = await client.reset("fake", "pick_cube", 3, "0,6,around")  # two descents left
        assert reset["prepose"]["stage"] == "around"
        idle = await client.step(STAY)
        assert idle["feedback"].endswith(" Use the new measured observation.")
        assert json.loads(idle["feedback"].rsplit(" Use the new", 1)[0])["reached"] is False
        assert idle["reward"] == 0.0 and not idle["done"]
        half = await client.step(DESCEND)
        assert half["feedback"] is None and half["reward"] == pytest.approx(0.5) and not half["success"]
        assert half["observation"]["state"]["left_arm"]["z"] < reset["observation"]["state"]["left_arm"]["z"]
        done = await client.step(DESCEND)
        assert done["success"] and done["done"] and done["reward"] == 1.0
        assert set(done) >= {"ok", "feedback", "result", "observation", "reward", "success", "done"}


@pytest.mark.asyncio
async def test_rungs_shorten_the_episode():
    remaining = {}
    for prepose in ("0,6,grasped", "0,6,around", "4,8", "x"):
        async with SimBridgeClient(fake_sim_command(), timeout=30) as client:
            reply = await client.reset("fake", "pick_cube", 11, prepose)
            remaining[prepose] = reply["observation"]["state"]["left_arm"]["z"]
    assert remaining["0,6,grasped"] < remaining["0,6,around"] < remaining["4,8"] <= remaining["x"]


@pytest.mark.asyncio
async def test_errors_are_replies_and_the_bridge_keeps_serving():
    async with SimBridgeClient(fake_sim_command(), timeout=30) as client:
        with pytest.raises(SimBridgeError, match="step before reset"):
            await client.step(DESCEND)
        with pytest.raises(SimBridgeError, match="unknown cmd"):
            await client.request({"cmd": "dance"})
        with pytest.raises(SimBridgeError, match="integer seed"):
            await client.request({"cmd": "reset", "sim": "fake", "task": "t", "seed": "7"})
        assert (await client.reset("fake", "pick_cube", 0, "x"))["ok"]


@pytest.mark.asyncio
async def test_dead_bridge_raises_with_stderr():
    client = SimBridgeClient([sys.executable, "-c", "import sys; sys.stderr.write('boom-marker\\n'); sys.exit(3)"])
    with pytest.raises(SimBridgeError, match="boom-marker"):
        await client.reset("fake", "pick_cube", 0, "x")
    await client.close()


@pytest.mark.asyncio
async def test_timeout_raises():
    client = SimBridgeClient([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.3)
    with pytest.raises(SimBridgeError, match="no reply"):
        await client.reset("fake", "pick_cube", 0, "x")
    await client.close()
    assert client.process is None


def test_stdout_carries_only_protocol_lines():
    """Whatever a simulator prints must not corrupt the protocol: fd 1 is redirected to stderr."""
    code = (
        "import runpy, sys, builtins\n"
        f"sys.argv = [{str(BRIDGE_SCRIPT)!r}, '--backend', 'fake']\n"
        "import json as _json\n"
        "_dumps = _json.dumps\n"
        "def noisy(*a, **k):\n"
        "    print('SIM NOISE')\n"
        "    return _dumps(*a, **k)\n"
        "_json.dumps = noisy\n"
        f"runpy.run_path({str(BRIDGE_SCRIPT)!r}, run_name='__main__')\n"
    )
    requests = (
        '{"cmd": "hello"}\n{"cmd": "reset", "sim": "fake", "task": "t", "seed": 1, "prepose": "x"}\n{"cmd": "close"}\n'
    )
    done = subprocess.run([sys.executable, "-c", code], input=requests, capture_output=True, text=True, timeout=60)
    lines = done.stdout.splitlines()
    assert len(lines) == 3 and all(json.loads(line)["ok"] for line in lines)
    assert json.loads(lines[0])["protocol"] == 1
    assert "SIM NOISE" in done.stderr and "SIM NOISE" not in done.stdout


def test_client_env_none_removes_variable(monkeypatch):
    monkeypatch.setenv("YAM_PEAR_TEST_VAR", "1")
    client = SimBridgeClient(["{bridge}"], env={"YAM_PEAR_TEST_VAR": None, "OTHER": 3})
    assert "YAM_PEAR_TEST_VAR" not in client.env and client.env["OTHER"] == "3"
    assert client.command == [str(BRIDGE_SCRIPT)]
    assert asyncio.run(client.close()) is None


STUB_DRIVER = '''
import json

class Episode:
    """The surface of raiden_pear.Episode / abc_pear_cur.Episode that the bridge relies on."""
    def __init__(self, args):
        print("driver chatter on stdout")  # the real simulators print while loading
        self.args = vars(args)
        self.instruction = "stub " + args.task
        self.prepose = {"given": args.prepose}
        self.success, self.max_reward, self.n = False, 0.0, 0
    def quantities(self):
        return {"left_arm": {"z": float(self.n)}}
    def images(self):
        return {"head": "H", "left_wrist": "L", "right_wrist": "R"}
    def act(self, action):
        self.n += 1
        self.max_reward = max(self.max_reward, float("nan") if action.get("nan") else 0.25 * self.n)
        self.success = self.n >= 2
        return (None if self.success else "feedback"), {"args": self.args}

def jpeg_bytes(image):
    return ("jpeg:" + image).encode()
'''


@pytest.mark.asyncio
async def test_driver_backend_wraps_an_episode_class(tmp_path):
    (tmp_path / "stub_driver.py").write_text(STUB_DRIVER)
    command = [
        sys.executable,
        "{bridge}",
        "--backend",
        "driver",
        "--driver-dir",
        str(tmp_path),
        "--driver-module",
        "stub_driver",
    ]
    async with SimBridgeClient(command, timeout=30) as client:
        reset = await client.reset("abc", "put_relative", 42, "x", {"lift": 0.08, "lift_pick": 0})
        assert reset["instruction"] == "stub put_relative" and reset["prepose"] == {"given": ""}  # "x" = Home
        assert reset["observation"]["images"] == {
            cam: base64.b64encode(f"jpeg:{tag}".encode()).decode()
            for cam, tag in (("head", "H"), ("left_wrist", "L"), ("right_wrist", "R"))
        }
        step = await client.step({"nan": True})
        args = step["result"]["args"]
        # what the drivers' command lines give Episode, plus the per-simulator episode_args
        assert args == {
            "task": "put_relative",
            "seed": 42,
            "prepose": "",
            "start": "home",
            "randomization": "",
            "grasp_depth": -1.0,
            "grasp_yaw_offset": 0.0,
            "lift": 0.08,
            "lift_pick": 0,
            "record_dense": False,
        }
        assert step["feedback"] == "feedback" and step["reward"] == 0.0  # a non-finite reward is not passed on
        last = await client.step({})
        assert last["success"] and last["done"] and last["reward"] == 0.5
    async with SimBridgeClient(command, timeout=30) as client:
        assert (await client.reset("abc", "t", 1, "2.5,7"))["prepose"] == {"given": "2.5,7"}
