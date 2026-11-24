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
"""Per-step videos from the saved frames: which episodes are picked, when a step is rendered, the mp4 itself."""

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from verl.experimental.yam_pear import videos


def jpeg(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_episode(root: Path, step: str, name: str, task: str, reward: float, turns: int = 2, frames: bool = True):
    directory = root / step / name
    directory.mkdir(parents=True)
    records = []
    for i in range(turns):
        images = [f"turn{i:02d}_{cam}.jpg" for cam in videos.CAMERA_ORDER] if frames else []
        for image in images:
            (directory / image).write_bytes(jpeg((40 * i, 80, 120)))
        records.append({"turn": i, "images": images, "max_reward": reward * (i + 1) / turns})
    if frames:
        for cam in videos.CAMERA_ORDER:
            (directory / f"final_{cam}.jpg").write_bytes(jpeg((200, 200, 0)))
    episode = {"task": task, "sim": "fake", "seed": 3, "rung": 4, "success": reward >= 1.0, "max_reward": reward}
    episode.update(end_reason="success" if reward >= 1.0 else "max_turns", num_turns=turns, frames=frames)
    (directory / "turns.json").write_text(json.dumps(records))
    (directory / "episode.json").write_text(json.dumps(episode))


def make_step(root: Path, step: str):
    for i, reward in enumerate([0.1, 0.2, 0.4, 1.0]):
        write_episode(root, step, f"s3_g{i}", "pick_cube", reward)
    write_episode(root, step, "s9_g0", "pick_banana", 0.3)
    write_episode(root, step, "s7_g0", "pick_cube", 0.9, frames=False)  # no frames: never picked


def test_selects_median_and_best_per_task(tmp_path):
    make_step(tmp_path, "step_003")
    picks = videos.select(videos.load_episodes(tmp_path / "step_003"))
    assert [(kind, e["task"], e["max_reward"]) for kind, e in picks] == [
        ("median", "pick_banana", 0.3),  # one episode: the median is also the best
        ("median", "pick_cube", 0.2),
        ("best", "pick_cube", 1.0),
    ]
    assert len(videos.select(videos.load_episodes(tmp_path / "step_003"), per_task=1)) == 2


def test_a_step_is_rendered_once_the_next_one_exists(tmp_path):
    out = tmp_path / "videos"
    for step in ("step_001", "eval_001", "step_002"):
        make_step(tmp_path, step)
    assert [p.name for p in videos.pending_steps(tmp_path, out, include_last=False)] == ["eval_001", "step_001"]
    assert len(videos.pending_steps(tmp_path, out, include_last=True)) == 3
    (out / "step_001").mkdir(parents=True)
    (out / "step_001" / "index.json").write_text("[]")
    assert [p.name for p in videos.pending_steps(tmp_path, out, include_last=False)] == ["eval_001"]


def test_frames_show_every_turn_and_the_end(tmp_path):
    write_episode(tmp_path, "step_001", "s3_g0", "pick_cube", 1.0, turns=3)
    (episode,) = videos.load_episodes(tmp_path / "step_001")
    frames = videos.episode_frames(episode, "step 001", height=96)
    assert len(frames) == 4  # three turns and the final view
    assert all(f.size == (3 * 128, 96 + videos.CAPTION_HEIGHT) for f in frames)  # three cameras side by side


def test_videos_are_written_and_indexed(tmp_path):
    try:
        videos.find_ffmpeg()
    except RuntimeError:
        pytest.skip("no ffmpeg")
    make_step(tmp_path, "step_001")
    make_step(tmp_path, "step_002")
    videos.main([str(tmp_path), "--height", "96"])  # a one-off run renders the newest step too
    for step in ("step_001", "step_002"):
        out = tmp_path / "videos" / step
        index = json.loads((out / "index.json").read_text())
        assert [e["file"] for e in index] == [
            "pick_banana_r4_median_fail.mp4",
            "pick_cube_r4_median_fail.mp4",
            "pick_cube_r4_best_success.mp4",
        ]
        for entry in index:
            data = (out / entry["file"]).read_bytes()
            assert data[4:8] == b"ftyp" and len(data) > 1000
        assert not list(out.glob(".tmp-*"))
    assert videos.run_once(tmp_path, tmp_path / "videos", 2, 96, True) == 0  # nothing is rendered twice
