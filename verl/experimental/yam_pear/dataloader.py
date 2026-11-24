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
"""Episode rows, the episode outcome log, and the curriculum dataloader (``data.custom_dataloader``).

verl has no curriculum interface; its hook for a custom prompt source is ``data.custom_dataloader``
(``verl/utils/dataset/dataloader.py``), which the trainer only pulls batches from. The feedback path the
curriculum needs is added here, outside verl core: every finished episode is appended by the agent loop to
an outcome log (JSON lines in a directory both sides can see), and the training loader reads the new lines
each time it plans a step.

One row is one (sim, task, seed, rung) start; ``rollout.n`` same-seed rollouts of a row form a GRPO group.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

from verl.experimental.yam_pear.curriculum import RUNGS, LadderCurriculum, stream_key, success_table

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

AGENT_NAME = "yam_pear_agent"
CALIBRATION_DONE = "calibration.done"
PLAN_LOG = "curriculum.jsonl"


def parse_streams(streams: str | Iterable[Any]) -> list[tuple[str, str, int]]:
    """Parse ``"sim:task:groups,..."`` (or a list of such strings / triples) into (sim, task, groups)."""
    items = streams.split(",") if isinstance(streams, str) else list(streams)
    out = []
    for item in items:
        sim, task, n = item.split(":") if isinstance(item, str) else item
        out.append((str(sim), str(task), int(n)))
    if not out:
        raise ValueError("no streams given")
    return out


def parse_rungs(rungs: str | int | Iterable[int] | None) -> list[int]:
    """Parse ``"3,7,11"`` (or a list) into rung indices."""
    if rungs is None or rungs == "":
        return []
    if isinstance(rungs, int):
        return [rungs]
    return [int(r) for r in (rungs.split(",") if isinstance(rungs, str) else rungs) if str(r) != ""]


def make_row(
    sim: str, task: str, seed: int, rung: int, *, split: str, index: int, plan_id: int = -1, calibration: bool = False
) -> dict[str, Any]:
    """One dataset row: an episode start.

    Args:
        sim: Simulator name, a key of the agent loop's ``sims`` config.
        task: Task name inside the simulator.
        seed: Scene seed; rollouts of the same row share it.
        rung: Index into ``RUNGS`` (start state and turn cap).
        split: ``"train"`` or ``"val"``.
        index: The row's ``extra_info.index``.
        plan_id: The curriculum step that planned the row (-1: not planned by the curriculum).
        calibration: Whether the row is a calibration episode (seeds the estimates, not an evaluation).

    Returns:
        The row, in the layout of verl's RL parquet files (``prompt``, ``data_source``, ``extra_info``).
    """
    prepose, max_turns = RUNGS[rung]
    prefix = "calibration/" if calibration else ""
    return {
        "data_source": f"{prefix}{stream_key(sim, task)}/rung{rung:02d}",
        "agent_name": AGENT_NAME,
        # The model never sees this message: the agent loop builds the station prompt from the simulator.
        "prompt": [{"role": "user", "content": f"{stream_key(sim, task)} seed {seed} rung {rung}"}],
        "extra_info": {
            "index": int(index),
            "sim": sim,
            "task": task,
            "seed": int(seed),
            "rung": int(rung),
            "prepose": prepose,
            "max_turns": int(max_turns),
            "split": split,
            "plan_id": int(plan_id),
            "calibration": bool(calibration),
        },
    }


def probe_rows(
    streams: list[tuple[str, str, int]], rungs: list[int], seed0: int, n_seeds: int, *, calibration: bool = False
) -> list[dict[str, Any]]:
    """Held-out episodes: ``n_seeds`` seeds x one row per stream and rung (the orchestrator's ``probe``)."""
    rows = []
    for sim, task, _ in streams:
        for rung in rungs:
            for i in range(n_seeds):
                rows.append(make_row(sim, task, seed0 + i, rung, split="val", index=len(rows), calibration=calibration))
    return rows


class OutcomeLog:
    """Append-only episode outcomes shared between the agent loop workers and the dataloader.

    Every process appends to its own file ``outcomes-<host>-<pid>.jsonl`` in ``directory`` (one JSON object
    per line, flushed per record), so writers never interleave. Readers remember a byte offset per file and
    only take complete lines. The directory must be visible to every agent loop worker and to the trainer.
    """

    def __init__(self, directory: str | os.PathLike):
        self.directory = Path(directory)
        self.offsets: dict[str, int] = {}

    def append(self, record: dict[str, Any]) -> None:
        """Write one outcome."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"outcomes-{os.uname().nodename}-{os.getpid()}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_new(self) -> list[dict[str, Any]]:
        """Return the outcomes appended since the previous call."""
        records = []
        for path in sorted(self.directory.glob("outcomes-*.jsonl")):
            with open(path, "rb") as f:
                f.seek(self.offsets.get(path.name, 0))
                data = f.read()
            complete = data.rfind(b"\n") + 1
            for line in data[:complete].splitlines():
                if line.strip():
                    records.append(json.loads(line))
            self.offsets[path.name] = self.offsets.get(path.name, 0) + complete
        return records


class CurriculumDataLoader:
    """``data.custom_dataloader`` target: adaptive skill-ladder training batches and held-out validation.

    Train split: an endless stream. Whenever its queue is empty it plans one step: it folds the outcomes of
    the steps planned before into the controller (:meth:`LadderCurriculum.observe`, once per planned step,
    in order), asks the controller for every stream's rungs, and queues one row per group with a fresh
    seed. This assumes the synchronous trainer (``trainer.v1.trainer_mode=sync``), where a step's episodes
    have all finished before the next batch is fetched; ``data.train_batch_size`` must be a multiple of the
    fetch size and equal the number of groups per step (the sum over ``streams``).

    Val split: the rows of ``data.val_files`` (see ``examples/yam_pear/create_dataset.py``). With
    ``calibrate=True`` the first validation pass of a run also runs calibration episodes (``cal_rungs`` x
    ``cal_seeds`` held-out seeds per stream); the training loader seeds the controller's counts from their
    outcomes, as the orchestrator's calibration probe did. Use it with ``trainer.val_before_train=True``.

    Args:
        streams: ``"sim:task:groups,..."``: the tasks and how many groups each gets per step.
        outcome_dir: Directory of the outcome log; the agent loop must write to the same one.
        lo: Lower edge of the success band.
        hi: Upper edge of the success band.
        holdout_rungs: Rungs never trained on, e.g. ``"7"``.
        seed0: First training seed; step k, stream s, group i gets ``seed0 + 10000 k + 100 s + i``.
        calibrate: Whether to seed the estimates from calibration episodes in the first validation pass.
        cal_rungs: Rungs of the calibration episodes.
        cal_seeds: Seeds per stream and rung in the calibration.
        cal_seed0: First calibration seed.
        init_counts: Optional path to a JSON ``{stream: {rung: [successes, episodes]}}`` (the orchestrator's
            ``controller.json``) to start from; disables calibration.
        _data_config: Supplied by verl: the read-only ``data`` config.
        _split: Supplied by verl: ``"train"`` or ``"val"``.
        _batch_size: Supplied by verl: prompts per yielded batch.
        _tokenizer: Supplied by verl, unused.
        _processor: Supplied by verl, unused.
        _collate_fn: Supplied by verl: collates a list of row dicts into a batch.
    """

    def __init__(
        self,
        streams: str,
        outcome_dir: str,
        lo: float = 0.2,
        hi: float = 0.8,
        holdout_rungs: str | None = "",
        seed0: int = 1000,
        calibrate: bool = True,
        cal_rungs: str = "0,1,2,3,5,7,11",
        cal_seeds: int = 8,
        cal_seed0: int = 500,
        init_counts: Optional[str] = None,
        *,
        _data_config: Any,
        _split: str,
        _batch_size: int,
        _tokenizer: Any = None,
        _processor: Any = None,
        _collate_fn: Callable[[list[dict]], dict],
    ):
        self.streams = parse_streams(streams)
        self.split = _split
        self.batch_size = int(_batch_size)
        self.collate_fn = _collate_fn
        self.data_config = _data_config
        self.directory = Path(os.path.expanduser(str(outcome_dir)))
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = OutcomeLog(self.directory)
        self.seed0 = int(seed0)
        self.cal_rungs = parse_rungs(cal_rungs)
        self.cal_seeds = int(cal_seeds)
        self.cal_seed0 = int(cal_seed0)

        counts = json.loads(Path(os.path.expanduser(init_counts)).read_text()) if init_counts else None
        self.calibrate = bool(calibrate) and counts is None
        self.curriculum = LadderCurriculum(lo=lo, hi=hi, holdout_rungs=parse_rungs(holdout_rungs), counts=counts)

        self.groups_per_step = sum(n for _, _, n in self.streams)
        if _split == "train" and self.groups_per_step % self.batch_size:
            raise ValueError(
                f"the streams give {self.groups_per_step} groups per step, which the fetch size {self.batch_size} "
                "does not divide; set data.train_batch_size to the number of groups per step."
            )
        # training state (see state_dict)
        self.plan_id = 0  # the next step to plan
        self.observed = 0  # steps below this one have been folded into the controller
        self.queue: list[dict[str, Any]] = []
        self.tables: dict[int, list[dict[str, Any]]] = {}  # planned step -> its outcomes so far
        self.calibrated = not self.calibrate
        self.cal_outcomes: list[dict[str, Any]] = []
        self.val_rows: Optional[list[dict[str, Any]]] = None

    # ------------------------------------------------------------------ shared

    def _batch(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        items = []
        for row in rows:
            item = {k: v for k, v in row.items() if k != "prompt"}
            item["raw_prompt"] = row["prompt"]
            item["index"] = row["extra_info"]["index"]
            # as RLHFDataset: a tensor column so that the batch is never tensor-free
            item["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
            items.append(item)
        return self.collate_fn(items)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self._iter_train() if self.split == "train" else self._iter_val()

    # ------------------------------------------------------------------ train

    def _collect(self) -> None:
        """Sort the outcomes logged since the last call into planned steps and calibration episodes."""
        for record in self.log.read_new():
            if record.get("calibration"):
                self.cal_outcomes.append(record)
            elif record.get("split") == "train" and record.get("plan_id", -1) >= self.observed:
                self.tables.setdefault(int(record["plan_id"]), []).append(record)

    def plan_step(self) -> dict[str, Any]:
        """Observe the finished steps, then plan and queue the next one. Returns the plan record."""
        self._collect()
        if not self.calibrated:
            table = success_table(self.cal_outcomes)
            if table:
                for stream, rungs in table.items():
                    self.curriculum.counts[stream] = {r: [float(c[0]), float(c[1])] for r, c in rungs.items()}
            else:
                logger.warning(
                    "calibrate=True but the outcome log holds no calibration episodes (is "
                    "trainer.val_before_train on?); starting from uninformed estimates."
                )
            self.calibrated = True
            self.cal_outcomes = []
        observed_tables = {}
        while self.observed < self.plan_id:
            table = success_table(self.tables.pop(self.observed, []))
            for sim, task, _ in self.streams:
                self.curriculum.observe(stream_key(sim, task), table.get(stream_key(sim, task), {}))
            observed_tables[self.observed] = table
            self.observed += 1

        k, plans = self.plan_id, {}
        for si, (sim, task, n) in enumerate(self.streams):
            rungs, p = self.curriculum.plan(stream_key(sim, task), n)
            plans[stream_key(sim, task)] = {"rungs": rungs, "estimate": p}
            for i, rung in enumerate(rungs):
                seed = self.seed0 + k * 10000 + si * 100 + i
                index = k * 10000 + si * 100 + i
                self.queue.append(make_row(sim, task, seed, rung, split="train", index=index, plan_id=k))
        self.plan_id += 1
        record = {
            "plan_id": k,
            "plan": plans,
            "observed": observed_tables,
            "counts": self.curriculum.state_dict()["counts"],
        }
        with open(self.directory / PLAN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        logger.info("curriculum step %s: rungs %s", k, json.dumps({s: v["rungs"] for s, v in plans.items()}))
        return record

    def _iter_train(self) -> Iterator[dict[str, Any]]:
        while True:
            if not self.queue:
                self.plan_step()
            rows, self.queue = self.queue[: self.batch_size], self.queue[self.batch_size :]
            yield self._batch(rows)

    def state_dict(self) -> dict[str, Any]:
        """Everything a resumed run needs to continue the curriculum where it stopped."""
        return {
            "curriculum": self.curriculum.state_dict(),
            "plan_id": self.plan_id,
            "observed": self.observed,
            "queue": list(self.queue),
            "tables": {int(k): list(v) for k, v in self.tables.items()},
            "calibrated": self.calibrated,
            "offsets": dict(self.log.offsets),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict`."""
        self.curriculum.load_state_dict(state["curriculum"])
        self.plan_id = int(state["plan_id"])
        self.observed = int(state["observed"])
        self.queue = list(state["queue"])
        self.tables = {int(k): list(v) for k, v in state["tables"].items()}
        self.calibrated = bool(state["calibrated"])
        self.log.offsets = dict(state["offsets"])

    # ------------------------------------------------------------------ val

    def _load_val_rows(self) -> list[dict[str, Any]]:
        import pandas as pd

        files = self.data_config.get("val_files")
        files = [files] if isinstance(files, str) else list(files or [])
        if not files:
            raise ValueError("data.val_files is required for validation")
        rows = []
        for path in files:
            for row in pd.read_parquet(os.path.expanduser(path)).to_dict("records"):
                row["prompt"] = [dict(m) for m in row["prompt"]]
                row["extra_info"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in row["extra_info"].items()}
                rows.append(row)
        return rows

    def _iter_val(self) -> Iterator[dict[str, Any]]:
        if self.val_rows is None:
            self.val_rows = self._load_val_rows()
        rows = list(self.val_rows)
        marker = self.directory / CALIBRATION_DONE
        calibrating = self.calibrate and not marker.exists()
        if calibrating:
            cal = probe_rows(self.streams, self.cal_rungs, self.cal_seed0, self.cal_seeds, calibration=True)
            for i, row in enumerate(cal):
                row["extra_info"]["index"] = len(rows) + i
            rows += cal
        for start in range(0, len(rows), self.batch_size):
            yield self._batch(rows[start : start + self.batch_size])
        if calibrating:
            marker.write_text("calibration episodes were submitted\n")
