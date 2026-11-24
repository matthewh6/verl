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
"""A couple of videos per training step, from the frames :mod:`artifacts` saved.

For every finished ``step_<NNN>`` (and ``eval_<NNN>``) directory and every task in it, the episodes that
have frames are ranked by reward and two are turned into a video: the median one (what the policy typically
does) and the best one. A video shows what the model saw: one picture per turn, the three cameras side by
side, with the turn, the reward so far and the outcome written underneath. It is a slideshow of the
model's observations, not a continuous recording; the motion between turns needs a re-simulation.

    <dir>/videos/step_<NNN>/<task>_r<rung>_<median|best>_<success|fail>.mp4
    <dir>/videos/step_<NNN>/index.json        what each video is; its presence marks the step as done

    python -m verl.experimental.yam_pear.videos <artifacts dir>                    # everything finished so far
    python -m verl.experimental.yam_pear.videos <artifacts dir> --watch 120       # keep going during training
    python -m verl.experimental.yam_pear.videos <artifacts dir> --wandb-project yam_pear --wandb-run-id <id>

A step counts as finished when a later step exists (or with ``--include-last``, e.g. after training).
Encoding needs an ``ffmpeg`` binary: one on PATH, or the ``imageio-ffmpeg`` package.
"""

import argparse
import io
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

CAMERA_ORDER = ("head", "left_wrist", "right_wrist")
STEP_DIR = re.compile(r"^(step|eval)_(\d+)$")
OUTPUT_FPS = 10
CAPTION_HEIGHT = 34


def find_ffmpeg() -> str:
    """Path of an ffmpeg binary (PATH first, then the imageio-ffmpeg package)."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError("no ffmpeg found: install ffmpeg or `pip install imageio-ffmpeg`") from exc


def load_episodes(step_dir: Path) -> list[dict[str, Any]]:
    """The episodes of a step directory that have frames, as ``episode.json`` plus their ``dir``."""
    episodes = []
    for path in sorted(step_dir.glob("*/episode.json")):
        episode = json.loads(path.read_text())
        if episode.get("frames") and any(path.parent.glob("turn*_*.jpg")):
            episodes.append({**episode, "dir": path.parent})
    return episodes


def select(episodes: list[dict[str, Any]], per_task: int = 2) -> list[tuple[str, dict[str, Any]]]:
    """(kind, episode) to render: per task the median-reward episode, then the best one if it is another."""
    by_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for episode in episodes:
        by_task.setdefault((str(episode.get("sim", "")), str(episode["task"])), []).append(episode)
    chosen = []
    for key in sorted(by_task):
        ranked = sorted(by_task[key], key=lambda e: (float(e["max_reward"]), bool(e["success"]), e["dir"].name))
        picks = [("median", ranked[(len(ranked) - 1) // 2])]
        if ranked[-1] is not picks[0][1]:
            picks.append(("best", ranked[-1]))
        chosen.extend(picks[: max(per_task, 0)])
    return chosen


def _cameras(names: list[str]) -> list[str]:
    known = [c for c in CAMERA_ORDER if c in names]
    return known + sorted(n for n in names if n not in known)


def _font(size: int = 18):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose(images: list[Image.Image], caption: str, height: int = 360) -> Image.Image:
    """The camera pictures side by side at one height, with a caption strip underneath (even-sized)."""
    scaled = [im.convert("RGB").resize((max(2, round(im.width * height / im.height)), height)) for im in images]
    width = sum(im.width for im in scaled)
    width += width % 2
    canvas = Image.new("RGB", (width, height + CAPTION_HEIGHT), (18, 18, 18))
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width
    ImageDraw.Draw(canvas).text((10, height + 7), caption, fill=(235, 235, 235), font=_font())
    return canvas


def episode_frames(episode: dict[str, Any], label: str = "", height: int = 360) -> list[Image.Image]:
    """One composed picture per turn (what the model saw before acting), then the final view."""
    directory: Path = episode["dir"]
    turns = json.loads((directory / "turns.json").read_text())
    head = " · ".join(p for p in (label, str(episode["task"]), f"rung {episode.get('rung', '?')}") if p)
    frames = []
    reward = 0.0
    for turn in turns:
        files = {name.split("_", 1)[1][: -len(".jpg")]: name for name in turn.get("images", [])}
        if not files:
            continue
        images = [Image.open(io.BytesIO((directory / files[c]).read_bytes())) for c in _cameras(list(files))]
        caption = f"{head} · turn {turn['turn'] + 1}/{len(turns)} · reward so far {reward:.2f}"
        frames.append(compose(images, caption, height))
        reward = float(turn.get("max_reward", reward))
    final = {p.name[len("final_") : -len(".jpg")]: p for p in directory.glob("final_*.jpg")}
    if final:
        images = [Image.open(io.BytesIO(final[c].read_bytes())) for c in _cameras(list(final))]
        outcome = "SUCCESS" if episode.get("success") else f"no success ({episode.get('end_reason')})"
        frames.append(compose(images, f"{head} · end · reward {float(episode['max_reward']):.2f} · {outcome}", height))
    return frames


def encode(frames: list[Image.Image], out: Path, seconds_per_turn: float = 0.6, final_hold: float = 1.5) -> None:
    """Write the pictures as an H.264 mp4 that browsers play (each held ``seconds_per_turn``)."""
    if not frames:
        raise ValueError("no frames to encode")
    size = frames[0].size
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".tmp-{out.name}")
    command = [
        find_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size[0]}x{size[1]}", "-r", str(OUTPUT_FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", "-movflags", "+faststart", "-f", "mp4", str(tmp),
    ]  # fmt: skip
    data = bytearray()
    for i, frame in enumerate(frames):
        hold = final_hold if i == len(frames) - 1 else seconds_per_turn
        raw = (frame if frame.size == size else frame.resize(size)).tobytes()
        data += raw * max(1, round(hold * OUTPUT_FPS))
    result = subprocess.run(command, input=bytes(data), capture_output=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='replace')[-500:]}")
    tmp.replace(out)


def render_step(step_dir: Path, out_root: Path, per_task: int = 2, height: int = 360) -> list[dict[str, Any]]:
    """Render one step directory's videos and write its ``index.json``; returns the index entries."""
    out_dir = out_root / step_dir.name
    index = []
    for kind, episode in select(load_episodes(step_dir), per_task):
        outcome = "success" if episode.get("success") else "fail"
        name = f"{episode['task']}_r{episode.get('rung', 'x')}_{kind}_{outcome}.mp4"
        encode(episode_frames(episode, step_dir.name.replace("_", " "), height), out_dir / name)
        index.append(
            {
                "file": name,
                "kind": kind,
                "sim": episode.get("sim"),
                "task": episode["task"],
                "rung": episode.get("rung"),
                "seed": episode["seed"],
                "success": bool(episode.get("success")),
                "max_reward": float(episode["max_reward"]),
                "end_reason": episode.get("end_reason"),
                "num_turns": episode.get("num_turns"),
                "episode": episode["dir"].name,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


def pending_steps(root: Path, out_root: Path, include_last: bool) -> list[Path]:
    """Step directories that are finished and have no videos yet, oldest first."""
    steps = sorted(
        (p for p in root.iterdir() if p.is_dir() and STEP_DIR.match(p.name)),
        key=lambda p: (int(STEP_DIR.match(p.name).group(2)), p.name),
    )
    last_training = max((int(p.name[5:]) for p in steps if p.name.startswith("step_")), default=None)
    todo = []
    for path in steps:
        number = int(STEP_DIR.match(path.name).group(2))
        # training step N is over once step N+1 has started; eval_N runs before training step N+1 as well
        finished = include_last or (last_training is not None and number < last_training)
        if finished and not (out_root / path.name / "index.json").exists():
            todo.append(path)
    return todo


class WandbUploader:
    """Logs each step's videos to a wandb run, against their own step axis (``videos/global_step``)."""

    def __init__(self, project: str, run_id: Optional[str] = None, name: Optional[str] = None):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, id=run_id, name=name, resume="allow" if run_id else None)
        # an own axis: the trainer may already have logged later steps into the same run
        self.run.define_metric("videos/global_step")
        self.run.define_metric("videos/*", step_metric="videos/global_step")

    def upload(self, step_name: str, out_dir: Path, index: list[dict[str, Any]]) -> None:
        split, number = STEP_DIR.match(step_name).groups()
        payload: dict[str, Any] = {"videos/global_step": int(number)}
        for entry in index:
            key = f"videos/{'train' if split == 'step' else 'eval'}/{entry['task']}/{entry['kind']}"
            caption = f"{step_name} rung {entry['rung']} reward {entry['max_reward']:.2f} {entry['end_reason']}"
            payload[key] = self.wandb.Video(str(out_dir / entry["file"]), caption=caption, format="mp4")
        if len(payload) > 1:
            self.run.log(payload)


def run_once(root: Path, out_root: Path, per_task: int, height: int, include_last: bool, uploader=None) -> int:
    """Render (and upload) every pending step; returns how many were rendered."""
    done = 0
    for step_dir in pending_steps(root, out_root, include_last):
        index = render_step(step_dir, out_root, per_task, height)
        if uploader is not None:
            uploader.upload(step_dir.name, out_root / step_dir.name, index)
        print(f"{step_dir.name}: {len(index)} videos -> {out_root / step_dir.name}", flush=True)
        done += 1
    return done


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dir", help="the artifacts directory (YAM_PEAR_ARTIFACTS_DIR)")
    parser.add_argument("--out", help="where the videos go (default: <dir>/videos)")
    parser.add_argument("--per-task", type=int, default=2, help="videos per task per step: 1 = median, 2 = + best")
    parser.add_argument("--height", type=int, default=360, help="height of the camera pictures in the video")
    parser.add_argument("--watch", type=float, default=0, help="keep running, looking for new steps every N seconds")
    parser.add_argument("--include-last", action="store_true", help="also render the newest step (training is over)")
    parser.add_argument("--wandb-project", help="upload the videos to this wandb project")
    parser.add_argument("--wandb-run-id", help="attach to this run (the trainer's) instead of a new one")
    parser.add_argument("--wandb-name", help="name of the new wandb run")
    args = parser.parse_args(argv)

    root = Path(args.dir).expanduser()
    out_root = Path(args.out).expanduser() if args.out else root / "videos"
    uploader = WandbUploader(args.wandb_project, args.wandb_run_id, args.wandb_name) if args.wandb_project else None
    include_last = args.include_last or not args.watch  # a one-off run renders everything that is there
    while True:
        if root.is_dir():
            run_once(root, out_root, args.per_task, args.height, include_last, uploader)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
