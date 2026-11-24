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
"""The curriculum dataloader through verl's ``data.custom_dataloader`` hook, with a simulated outcome log."""

import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

import verl.utils.tensordict_utils as tu
from verl.experimental.yam_pear.curriculum import RUNGS, LadderCurriculum, stream_key
from verl.experimental.yam_pear.dataloader import (
    AGENT_NAME,
    CALIBRATION_DONE,
    PLAN_LOG,
    OutcomeLog,
    make_row,
    parse_streams,
    probe_rows,
)
from verl.utils.dataset.dataloader import create_custom_dataloader, validate_custom_batch
from verl.utils.dataset.rl_dataset import collate_fn

STREAMS = "raiden:pick_cube:5,raiden:pick_banana:5,abc:put_relative:6"
TARGET = "verl.experimental.yam_pear.dataloader.CurriculumDataLoader"


def make_loader(tmp_path, split, batch_size=16, **overrides):
    val_file = tmp_path / "val.parquet"
    if not val_file.exists():
        pd.DataFrame(probe_rows(parse_streams(STREAMS), [3, 7, 11], 100, 2)).to_parquet(val_file)
    loader_cfg = {"_target_": TARGET, "streams": STREAMS, "outcome_dir": str(tmp_path / "outcomes"), **overrides}
    data = OmegaConf.create({"custom_dataloader": loader_cfg, "custom_cls": None, "val_files": str(val_file)})
    return create_custom_dataloader(
        data, split=split, batch_size=batch_size, tokenizer=None, processor=None, collate_fn=collate_fn
    )


def play(tmp_path, batch, success_of):
    """What the agent loop does for a training batch: log 8 same-seed episodes per row."""
    log = OutcomeLog(tmp_path / "outcomes")
    for info in batch["extra_info"]:
        for session in range(8):
            ok = success_of(info, session)
            log.append({**{k: info[k] for k in ("split", "plan_id", "calibration", "sim", "task", "seed", "rung")},
                        "success": ok, "reward": float(ok), "session_id": session})  # fmt: skip


def test_rows_carry_the_rung_start_state_and_turn_cap():
    row = make_row("raiden", "pick_cube", 7, 2, split="train", index=5, plan_id=3)
    info = row["extra_info"]
    assert (info["prepose"], info["max_turns"]) == RUNGS[2] == ("0.5,6,aligned", 8)
    assert row["agent_name"] == AGENT_NAME and row["data_source"] == "raiden:pick_cube/rung02"
    assert info["index"] == 5 and info["plan_id"] == 3 and info["seed"] == 7 and info["split"] == "train"
    assert row["prompt"][0]["role"] == "user"
    probe = probe_rows(parse_streams(STREAMS), [3, 11], 100, 4)
    assert len(probe) == 3 * 2 * 4 and {r["extra_info"]["seed"] for r in probe} == {100, 101, 102, 103}
    assert RUNGS[11] == ("x", 25) and probe[-1]["extra_info"]["prepose"] == "x"


def test_train_batches_satisfy_verls_contract_and_follow_the_reference_plan(tmp_path):
    loader = make_loader(tmp_path, "train", calibrate=False)
    it = iter(loader)
    reference = LadderCurriculum()
    for step in range(4):
        batch = next(it)
        validate_custom_batch(batch, split="train", batch_size=16)
        assert len(tu.get_tensordict(batch)) == 16  # what the trainer does with a fetched batch
        infos = list(batch["extra_info"])
        assert all(name == AGENT_NAME for name in batch["agent_name"])
        expected_rungs, expected_seeds = [], []
        for si, (sim, task, n) in enumerate(parse_streams(STREAMS)):
            rungs, _ = reference.plan(stream_key(sim, task), n)
            expected_rungs += rungs
            expected_seeds += [1000 + step * 10000 + si * 100 + i for i in range(n)]  # the orchestrator's seeds
        assert [i["rung"] for i in infos] == expected_rungs
        assert [i["seed"] for i in infos] == expected_seeds
        assert all(i["plan_id"] == step and i["split"] == "train" for i in infos)
        assert [i["prepose"] for i in infos] == [RUNGS[r][0] for r in expected_rungs]
        assert len(set(batch["index"])) == 16

        def success_of(info, session, step=step):
            return (info["seed"] + session + step) % 3 != 0 if info["rung"] < 2 else session == 0

        play(tmp_path, batch, success_of)
        # the reference loop: observe every stream with this step's table, then plan the next step
        for sim, task, _ in parse_streams(STREAMS):
            table = {}
            for info in infos:
                if (info["sim"], info["task"]) == (sim, task):
                    c = table.setdefault(str(info["rung"]), [0, 0])
                    c[0] += sum(success_of(info, s) for s in range(8))
                    c[1] += 8
            reference.observe(stream_key(sim, task), table)
    assert loader.curriculum.counts != reference.counts  # step 3 is observed when step 4 is planned
    next(it)
    assert loader.curriculum.counts == reference.counts
    plans = [json.loads(line) for line in (tmp_path / "outcomes" / PLAN_LOG).read_text().splitlines()]
    assert [p["plan_id"] for p in plans] == [0, 1, 2, 3, 4] and "raiden:pick_cube" in plans[0]["plan"]


def test_fetch_size_smaller_than_a_step_and_lost_steps(tmp_path):
    loader = make_loader(tmp_path, "train", batch_size=1, calibrate=False)
    it = iter(loader)
    step0 = [next(it) for _ in range(16)]
    assert {b["extra_info"][0]["plan_id"] for b in step0} == {0} and loader.plan_id == 1
    before = {s: dict(c) for s, c in loader.curriculum.counts.items()}
    assert before == {}
    assert next(it)["extra_info"][0]["plan_id"] == 1  # nothing was logged for step 0: counts only fade
    assert all(c == [0.0, 0.0] for t in loader.curriculum.counts.values() for c in t.values())
    with pytest.raises(ValueError, match="groups per step"):
        make_loader(tmp_path, "train", batch_size=5, calibrate=False)


def test_holdout_rungs_are_never_planned(tmp_path):
    loader = make_loader(tmp_path, "train", calibrate=False, holdout_rungs="0,1,7")
    it = iter(loader)
    seen = set()
    for step in range(25):
        batch = next(it)
        seen |= {info["rung"] for info in batch["extra_info"]}
        play(tmp_path, batch, lambda info, session: session < 5)  # everything in the band keeps moving up
    assert seen and not seen & {0, 1, 7}
    assert max(seen) > 7  # the ladder is climbed past the held-out rung


def test_validation_and_calibration_seed_the_estimates(tmp_path):
    val = make_loader(tmp_path, "val", batch_size=16, cal_rungs="0,3", cal_seeds=4)
    batches = list(val)
    for batch in batches:
        validate_custom_batch(batch, split="val", batch_size=16)
    infos = [info for batch in batches for info in batch["extra_info"]]
    sources = [s for batch in batches for s in batch["data_source"]]
    evals = [i for i in infos if not i["calibration"]]
    cals = [i for i in infos if i["calibration"]]
    assert len(evals) == 3 * 3 * 2 and len(cals) == 3 * 2 * 4
    assert all(i["split"] == "val" for i in infos) and len({i["index"] for i in infos}) == len(infos)
    assert {s.split("/")[0] for s in sources} == {
        "calibration",
        "raiden:pick_cube",
        "raiden:pick_banana",
        "abc:put_relative",
    }
    assert {i["seed"] for i in cals} == {500, 501, 502, 503} and {i["seed"] for i in evals} == {100, 101}
    assert (tmp_path / "outcomes" / CALIBRATION_DONE).exists()

    log = OutcomeLog(tmp_path / "outcomes")
    for info in infos:  # rung 0 always works, rung 3 never
        log.append({**info, "success": info["rung"] == 0, "reward": float(info["rung"] == 0)})
    # later validation passes are the evaluation only
    assert sum(len(b["extra_info"]) for b in val) == len(evals)

    train = make_loader(tmp_path, "train", cal_rungs="0,3", cal_seeds=4)
    next(iter(train))
    for sim, task, _ in parse_streams(STREAMS):
        # only calibration episodes seed the counts, not the evaluation episodes on rungs 3, 7, 11
        assert train.curriculum.counts[stream_key(sim, task)] == {"0": [4.0, 4.0], "3": [0.0, 4.0]}


def test_init_counts_replace_calibration(tmp_path):
    counts = {"raiden:pick_cube": {"0": [8, 8], "1": [5, 8]}}
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(counts))
    val = make_loader(tmp_path, "val", init_counts=str(path))
    assert all(not info["calibration"] for batch in val for info in batch["extra_info"])
    train = make_loader(tmp_path, "train", init_counts=str(path))
    assert (
        train.curriculum.estimate("raiden:pick_cube")[:2]
        == LadderCurriculum(counts=counts).estimate("raiden:pick_cube")[:2]
    )


def test_resume_continues_the_same_curriculum(tmp_path):
    def success_of(info, session):
        return (info["seed"] * 7 + session) % 4 < 2

    loader = make_loader(tmp_path, "train", calibrate=False)
    it = iter(loader)
    for _ in range(3):
        batch = next(it)
        play(tmp_path, batch, success_of)
    state = json.loads(json.dumps(loader.state_dict()))  # survives serialisation
    following = next(it)

    resumed = make_loader(tmp_path, "train", calibrate=False)
    resumed.load_state_dict({**state, "tables": {int(k): v for k, v in state["tables"].items()}})
    again = next(iter(resumed))
    assert list(again["extra_info"]) == list(following["extra_info"])
    assert resumed.curriculum.counts == loader.curriculum.counts


def test_outcome_log_reads_only_complete_new_lines(tmp_path):
    log = OutcomeLog(tmp_path)
    log.append({"a": 1})
    assert log.read_new() == [{"a": 1}] and log.read_new() == []
    path = next(tmp_path.glob("outcomes-*.jsonl"))
    with open(path, "a") as f:
        f.write('{"a": 2}\n{"a": 3')  # a writer caught mid-line
    assert log.read_new() == [{"a": 2}]
    with open(path, "a") as f:
        f.write("}\n")
    assert log.read_new() == [{"a": 3}]
    assert OutcomeLog(tmp_path).read_new() == [{"a": 1}, {"a": 2}, {"a": 3}]
