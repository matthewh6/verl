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
"""Episode artifacts in the layout of the station drivers: ``episode.json`` + ``turns.json`` per episode.

Every finished episode gets a small record, enough to re-simulate it (sim, task, seed, start state, the
actions) and to classify its behaviour (replies, feedback, results, rewards, end reason). The field names
are those ``raiden_pear.py`` / ``abc_pear_cur.py`` ``main()`` write, without token ids and log-probs, so the
existing report tools read them. Camera frames (three JPEGs per turn, the bytes the simulator sent) are
only kept for a sample of episodes.

    <dir>/step_<NNN>/s<seed>_g<i>/          training: NNN = the trainer's global step, i = rollout in the group
    <dir>/eval_<NNN>/s<seed>_g<group>/      validation (group = 100 x stream + rung, as the orchestrator's probe)
    <dir>/calibration/s<seed>_g<group>/     calibration episodes

An episode directory is written under a temporary name and renamed into place, so a reader never sees a
partial episode and several agent loop workers can write into one (shared) directory.
"""

import json
import os
import shutil
import uuid
import zlib
from pathlib import Path
from typing import Any, Optional

FRAME_MODES = ("none", "sample", "all")


class EpisodeArtifacts:
    """Writes episode records, and frames for the configured sample.

    Args:
        dir: Root directory; ``None`` (the default) disables artifacts altogether.
        frames: Frames of TRAINING episodes: ``"none"``, ``"all"``, or ``"sample"`` = one group (all of its
            same-seed rollouts) per stream per step, chosen by a hash of (step, stream) so that every worker
            picks the same one. ``"sample"`` needs rows planned by the curriculum dataloader, which carry
            the group's position in its stream; other rows get no frames.
        val_frames: Whether to keep the frames of validation and calibration episodes (all of them).
    """

    def __init__(self, dir: Optional[str] = None, frames: str = "none", val_frames: bool = False):
        if frames not in FRAME_MODES:
            raise ValueError(f"artifacts.frames must be one of {FRAME_MODES}, got {frames!r}")
        self.dir = Path(os.path.expanduser(str(dir))) if dir and str(dir).lower() not in ("null", "none") else None
        self.frames = frames
        # the flag may arrive as text (an environment variable through the config): "false" must not be truthy
        if isinstance(val_frames, str):
            if val_frames.strip().lower() not in ("true", "false", "1", "0", "yes", "no", ""):
                raise ValueError(f"artifacts.val_frames must be a boolean, got {val_frames!r}")
            val_frames = val_frames.strip().lower() in ("true", "1", "yes")
        self.val_frames = bool(val_frames)

    @property
    def enabled(self) -> bool:
        """Whether anything is written."""
        return self.dir is not None

    @staticmethod
    def sampled_group(plan_id: int, sim: str, task: str, groups_in_stream: int) -> int:
        """The position (within its stream) of the one group whose frames are kept this step."""
        return zlib.crc32(f"{plan_id}:{sim}:{task}".encode()) % groups_in_stream

    def wants_frames(self, info: dict[str, Any]) -> bool:
        """Whether the frames of the episode described by a row's ``extra_info`` are kept."""
        if not self.enabled:
            return False
        if info.get("split") == "val":
            return self.val_frames
        if self.frames != "sample":
            return self.frames == "all"
        groups, position = int(info.get("groups_in_stream", 0)), int(info.get("group_index", -1))
        if groups <= 0 or position < 0:
            return False
        return position == self.sampled_group(int(info["plan_id"]), info["sim"], info["task"], groups)

    @staticmethod
    def location(info: dict[str, Any], global_steps: int, session_id: int) -> tuple[str, str]:
        """(step directory name, episode directory name) of an episode."""
        if info.get("split") == "val":
            step = "calibration" if info.get("calibration") else f"eval_{int(global_steps):03d}"
            name = f"s{info['seed']}_g{int(info.get('group', info.get('index', 0)))}"
            return step, name + (f"_r{session_id}" if session_id else "")
        return f"step_{int(global_steps):03d}", f"s{info['seed']}_g{session_id}"

    def write(
        self,
        step: str,
        name: str,
        episode: dict[str, Any],
        turns: list[dict[str, Any]],
        system_prompt: str,
        frames: Optional[dict[str, bytes]] = None,
    ) -> Optional[Path]:
        """Write one episode directory atomically; returns its path (None when disabled).

        Args:
            step: Step directory name from :meth:`location`.
            name: Episode directory name from :meth:`location`.
            episode: The content of ``episode.json``.
            turns: The content of ``turns.json``.
            system_prompt: The episode's system prompt (``system_prompt.md``).
            frames: File name -> JPEG bytes, or None when the episode's frames are not kept.
        """
        if self.dir is None:
            return None
        parent = self.dir / step
        parent.mkdir(parents=True, exist_ok=True)
        tmp = parent / f".tmp-{name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        tmp.mkdir()
        try:
            (tmp / "system_prompt.md").write_text(system_prompt)
            for file_name, data in (frames or {}).items():
                (tmp / file_name).write_bytes(data)
            (tmp / "turns.json").write_text(json.dumps(turns, indent=1))
            (tmp / "episode.json").write_text(json.dumps(episode))
            final = parent / name
            try:
                os.rename(tmp, final)
            except OSError:  # the episode was run again (a reissued prompt): the newer one replaces it
                shutil.rmtree(final, ignore_errors=True)
                os.rename(tmp, final)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return final
