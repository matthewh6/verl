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
"""Episode artifacts: records in the old drivers' layout for every episode, frames for a sample."""

import importlib.util
import json
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tests.experimental.yam_pear.test_agent_loop_on_cpu import (
    SAMPLING,
    CharTokenizer,
    MockServer,
    action,
    make_loop,
    reply,
    row_kwargs,
)
from verl.experimental.yam_pear import protocol
from verl.experimental.yam_pear.artifacts import EpisodeArtifacts
from verl.experimental.yam_pear.dataloader import make_row, parse_streams, probe_rows
from verl.experimental.yam_pear.sim_client import BRIDGE_SCRIPT

STREAMS = "raiden:pick_cube:5,raiden:pick_banana:5,abc:put_relative:6"
# what raiden_pear.py main() writes, minus token ids / log-probs and simulator internals
DRIVER_EPISODE_KEYS = {
    "task", "seed", "randomization_request", "start", "instruction", "model", "server", "effort", "history_k",
    "object_state", "prepose", "max_tokens", "end_reason", "success", "max_reward", "num_turns", "wall_s",
    "completion_tokens",
}  # fmt: skip
DRIVER_TURN_KEYS = {
    "turn", "state", "user_text", "images", "model_seconds", "reasoning", "answer", "finish_reason", "usage",
    "action", "feedback", "result", "sim_seconds", "success", "max_reward",
}  # fmt: skip


def load_bridge():
    spec = importlib.util.spec_from_file_location("yam_pear_sim_bridge", BRIDGE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("*/*") if p.is_dir())


def test_disabled_by_default(monkeypatch):
    artifacts = EpisodeArtifacts()
    assert not artifacts.enabled and artifacts.frames == "none" and not artifacts.val_frames
    assert artifacts.write("step_000", "s1_g0", {}, [], "") is None
    assert not artifacts.wants_frames(make_row("fake", "t", 1, 0, split="train", index=0)["extra_info"])
    for name in ("YAM_PEAR_ARTIFACTS_DIR", "YAM_PEAR_ARTIFACT_FRAMES", "YAM_PEAR_ARTIFACT_VAL_FRAMES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PEAR_ROOT", "/pear")
    monkeypatch.setenv("ABC_SIM_DIR", "/abc")
    for name in ("agent_loop.yaml", "agent_loop_fake.yaml"):  # the shipped configs: off unless a directory is given
        (config,) = OmegaConf.load(protocol.PROMPT_DIR.parent / "config" / name)
        block = OmegaConf.to_container(config.artifacts, resolve=True)
        assert block["dir"] is None and block["frames"] == "sample"
        shipped = EpisodeArtifacts(**block)
        assert not shipped.enabled and shipped.val_frames is False
    # values that came through environment variables are text
    from_env = EpisodeArtifacts(dir="null", frames="sample", val_frames="False")
    assert not from_env.enabled and from_env.val_frames is False
    assert EpisodeArtifacts(dir="/x", val_frames="true").val_frames is True
    with pytest.raises(ValueError, match="val_frames"):
        EpisodeArtifacts(dir="/x", val_frames="maybe")
    with pytest.raises(ValueError, match="artifacts.frames"):
        EpisodeArtifacts(dir="/x", frames="some")


@pytest.mark.asyncio
async def test_no_artifacts_are_written_when_disabled(tmp_path):
    loop = make_loop(tmp_path, MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0)))
    await loop.run(dict(SAMPLING), **row_kwargs(rung=0))
    assert [p.name for p in tmp_path.iterdir()] == ["outcomes"]


@pytest.mark.asyncio
async def test_record_is_complete_and_resimulates(tmp_path):
    def policy(prompt, i):
        return reply(0.0 if i == 1 else -5.0, reasoning=f"thinking about turn {i}")

    root = tmp_path / "episodes"
    loop = make_loop(tmp_path, MockServer(CharTokenizer(), policy), artifacts={"dir": str(root), "frames": "all"})
    kwargs = {**row_kwargs(rung=3, seed=5, uid="uid-7", session_id=2), "global_steps": 7}
    outputs = await loop.run(dict(SAMPLING), **kwargs)

    (episode_dir,) = episode_dirs(root)
    assert episode_dir == root / "step_007" / "s5_g2"  # <dir>/step_<NNN>/s<seed>_g<i>/
    assert not list(root.rglob(".tmp-*"))
    episode = json.loads((episode_dir / "episode.json").read_text())
    turns = json.loads((episode_dir / "turns.json").read_text())
    assert DRIVER_EPISODE_KEYS <= set(episode)
    assert (episode["task"], episode["seed"], episode["start"], episode["randomization_request"]) == (
        "pick_cube",
        5,
        "home",
        "",
    )
    assert episode["prepose"] == {"radius": 1.0, "height": 6.0, "stage": "above"}  # what the simulator reports
    assert (episode["end_reason"], episode["success"], episode["max_reward"]) == ("success", True, 1.0)
    assert episode["num_turns"] == len(turns) == len(outputs) == 4
    assert (episode["sim"], episode["rung"], episode["split"], episode["global_steps"]) == ("fake", 3, "train", 7)
    assert (episode["uid"], episode["session_id"], episode["plan_id"]) == ("uid-7", 2, 9)
    assert (episode["model"], episode["effort"], episode["history_k"]) == ("Qwen3.8-27B-PEAR", "low", 2)
    assert (episode_dir / "system_prompt.md").read_text().startswith("You are controlling a robot.")

    rewards = []
    for i, turn in enumerate(turns):
        assert DRIVER_TURN_KEYS <= set(turn) and turn["turn"] == i
        assert (
            turn["reasoning"] == f"thinking about turn {i}" and protocol.parse_action(turn["answer"]) == turn["action"]
        )
        assert turn["action"] == action(0.0 if i == 1 else -5.0)
        assert turn["user_text"].startswith(json.dumps(turn["state"]))
        assert turn["finish_reason"] == "stop" and turn["usage"]["completion_tokens"] > 0
        assert isinstance(turn["result"], dict) and turn["result"]["status"] == "completed"
        rewards.append(turn["max_reward"])
        assert turn["images"] == [f"turn{i:02d}_{cam}.jpg" for cam in ("head", "left_wrist", "right_wrist")]
        for name in turn["images"]:
            assert (episode_dir / name).read_bytes()[:3] == b"\xff\xd8\xff"  # the simulator's JPEG bytes
    assert turns[1]["feedback"].endswith("Use the new measured observation.") and turns[0]["feedback"] is None
    assert rewards == sorted(rewards) and rewards[-1] == 1.0 and turns[-1]["success"] is True
    assert sorted(p.name for p in episode_dir.glob("final_*.jpg")) == [
        "final_head.jpg",
        "final_left_wrist.jpg",
        "final_right_wrist.jpg",
    ]

    # the report tools' reading (behavior_report/extract_inc.py: action -> "a", result -> "r")
    assert all(isinstance(t.get("action"), dict) and isinstance(t.get("result"), dict) for t in turns)
    # ... and replay_video.py's: rebuild the start state from episode.json, replay the recorded actions
    p = episode["prepose"]
    prepose = f"{p['radius']},{p['height']}" + (f",{p['stage']}" if p.get("stage") else "")
    replay = load_bridge().FakeEpisode(
        types.SimpleNamespace(task=episode["task"], seed=int(episode["seed"]), prepose=prepose)
    )
    for turn in turns:
        replay.act(turn["action"])
    assert (replay.success, replay.max_reward) == (episode["success"], episode["max_reward"])


@pytest.mark.asyncio
async def test_records_without_frames_and_unfinished_turns(tmp_path):
    root = tmp_path / "episodes"
    loop = make_loop(tmp_path, MockServer(CharTokenizer(), lambda prompt, i: "no action"), artifacts={"dir": str(root)})
    await loop.run(dict(SAMPLING), **{**row_kwargs(rung=0, seed=4), "global_steps": 12})
    episode_dir = root / "step_012" / "s4_g0"
    assert sorted(p.name for p in episode_dir.iterdir()) == [
        "episode.json",
        "system_prompt.md",
        "turns.json",
    ]  # no frames
    episode = json.loads((episode_dir / "episode.json").read_text())
    (turn,) = json.loads((episode_dir / "turns.json").read_text())
    assert episode["end_reason"] == "parse" and episode["num_turns"] == 1 and episode["frames"] is False
    assert turn["images"] == [] and turn["answer"] == "no action" and "action" not in turn
    assert turn["parse_error"] == "Final answer must end with a fenced json action"

    cut = make_loop(
        tmp_path,
        MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0), stop_reason="length"),
        artifacts={"dir": str(root)},
    )
    await cut.run(dict(SAMPLING), **{**row_kwargs(rung=0, seed=6), "global_steps": 12})
    (turn,) = json.loads((root / "step_012" / "s6_g0" / "turns.json").read_text())
    assert turn["finish_reason"] == "length" and "token limit" in turn["parse_error"]

    failing = make_loop(
        tmp_path,
        MockServer(CharTokenizer(), lambda prompt, i: reply(-5.0), stop_reason="aborted"),
        artifacts={"dir": str(root)},
    )
    with pytest.raises(RuntimeError):
        await failing.run(dict(SAMPLING), **{**row_kwargs(rung=0, seed=8), "global_steps": 12})
    assert not (root / "step_012" / "s8_g0").exists()  # a failed rollout is not an episode


def test_frame_sample_is_exactly_one_group_per_stream_per_step(tmp_path):
    from tests.experimental.yam_pear.test_dataloader_on_cpu import make_loader

    loader = make_loader(tmp_path, "train", calibrate=False)
    workers = [EpisodeArtifacts(dir=str(tmp_path / "a"), frames="sample") for _ in range(3)]  # separate processes
    it = iter(loader)
    positions = {stream: set() for stream in ("raiden:pick_cube", "raiden:pick_banana", "abc:put_relative")}
    for step in range(40):
        infos = list(next(it)["extra_info"])
        assert len(infos) == 16
        chosen = [info for info in infos if workers[0].wants_frames(info)]
        assert all([i for i in infos if w.wants_frames(i)] == chosen for w in workers[1:])  # all workers agree
        assert sorted(f"{i['sim']}:{i['task']}" for i in chosen) == sorted(positions)  # one group per stream
        for info in chosen:
            positions[f"{info['sim']}:{info['task']}"].add(info["group_index"])
    assert all(len(seen) >= 3 for seen in positions.values())  # not always the same group (e.g. the probe)


def test_frame_modes_and_validation_flag():
    planned = make_row("fake", "t", 1, 0, split="train", index=0, plan_id=3, group_index=0, groups_in_stream=1)[
        "extra_info"
    ]
    static = make_row("fake", "t", 1, 0, split="train", index=0)["extra_info"]  # not planned by the curriculum
    val = probe_rows(parse_streams(STREAMS), [3], 100, 1)[0]["extra_info"]
    cal = probe_rows(parse_streams(STREAMS), [3], 500, 1, calibration=True)[0]["extra_info"]
    wants = {
        mode: [EpisodeArtifacts(dir="/x", frames=mode).wants_frames(i) for i in (planned, static, val, cal)]
        for mode in ("none", "sample", "all")
    }
    assert wants == {"none": [False] * 4, "sample": [True, False, False, False], "all": [True, True, False, False]}
    with_val = EpisodeArtifacts(dir="/x", frames="none", val_frames=True)
    assert [with_val.wants_frames(i) for i in (planned, static, val, cal)] == [False, False, True, True]


def test_locations_are_unique_per_episode():
    rows = probe_rows(parse_streams(STREAMS), [3, 7, 11], 100, 4)
    names = [EpisodeArtifacts.location(r["extra_info"], 20, 0) for r in rows]
    assert len(set(names)) == len(rows) and {step for step, _ in names} == {"eval_020"}
    assert names[0] == ("eval_020", "s100_g3") and names[-1] == ("eval_020", "s103_g211")  # g = 100 x stream + rung
    cal = probe_rows(parse_streams(STREAMS), [0], 500, 1, calibration=True)[0]["extra_info"]
    assert EpisodeArtifacts.location(cal, 0, 0) == ("calibration", "s500_g0")
    assert EpisodeArtifacts.location(cal, 0, 1) == ("calibration", "s500_g0_r1")
    train = make_row("raiden", "pick_cube", 21003, 5, split="train", index=0, plan_id=2)["extra_info"]
    assert [EpisodeArtifacts.location(train, 3, g) for g in (0, 7)] == [
        ("step_003", "s21003_g0"),
        ("step_003", "s21003_g7"),
    ]


def test_write_is_atomic_and_a_rerun_replaces_the_episode(tmp_path):
    artifacts = EpisodeArtifacts(dir=str(tmp_path))
    first = artifacts.write("step_001", "s1_g0", {"n": 1}, [{"turn": 0}], "sys", {"turn00_head.jpg": b"\xff\xd8\xff1"})
    assert first == tmp_path / "step_001" / "s1_g0" and (first / "turn00_head.jpg").read_bytes() == b"\xff\xd8\xff1"
    artifacts.write("step_001", "s1_g0", {"n": 2}, [], "sys")
    assert json.loads((first / "episode.json").read_text()) == {"n": 2} and not (first / "turn00_head.jpg").exists()
    assert [p.name for p in (tmp_path / "step_001").iterdir()] == ["s1_g0"]  # nothing temporary is left
