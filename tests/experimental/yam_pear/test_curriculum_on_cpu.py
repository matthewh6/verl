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
"""The curriculum controller against the reference behaviour of ``grpo_loop_adapt.py``.

``_Reference`` holds ``estimate`` / ``plan`` / ``observe`` copied from the orchestrator (only its module
globals ``a.lo``, ``a.hi``, ``HOLD`` and ``RUNGS`` are turned into attributes), so the port is checked
against the code it replaces rather than against a description of it.
"""

import copy
import random

import pytest

from verl.experimental.yam_pear.curriculum import RUNGS, LadderCurriculum, stream_key, success_table

REFERENCE_RUNGS = [("0,6,grasped", 4), ("0,6,around", 6), ("0.5,6,aligned", 8), ("1,6", 12), ("2.5,7", 12), ("4,8", 12),
                   ("6,9", 12), ("8,10", 14), ("11,11", 14), ("15,12", 16), ("-1,0", 20), ("x", 25)]  # fmt: skip


class _Reference:
    def __init__(self, lo, hi, hold):
        self.lo, self.hi, self.HOLD = lo, hi, set(hold)

    def estimate(self, est):
        a, HOLD, RUNGS = self, self.HOLD, REFERENCE_RUNGS  # noqa: F841, N806
        p, guessed = [], False
        for r in range(len(RUNGS)):
            s, n = est.get(str(r), [0, 0])
            if n >= 4:
                p.append((s + 0.5) / (n + 1))
                guessed = False
            elif r in HOLD:
                p.append(p[-1] if p else 0.5)
            else:
                p.append((0.4 if guessed else 0.7) * p[-1] if p else 0.5)
                guessed = True
        return p

    def plan(self, est, n_groups):
        a, HOLD, RUNGS, estimate = self, self.HOLD, REFERENCE_RUNGS, self.estimate  # noqa: N806
        p = estimate(est)
        ok = [r for r in range(len(RUNGS)) if r not in HOLD]
        cand = [r for r in ok if a.lo + 0.1 <= p[r] <= a.hi]
        if not cand:
            r = min(ok, key=lambda r: abs(p[r] - 0.5))
            cand = [r] + ([min(x for x in ok if x > r)] if p[r] > a.hi and r < max(ok) else [])
        mid, hard, easy = min(cand, key=lambda r: abs(p[r] - 0.6)), max(cand), min(cand)
        n_probe = 2 if n_groups >= 5 and hard != mid else 1
        rungs = [hard] * n_probe + [easy if i % 3 == 2 else mid for i in range(n_probe, n_groups)]
        return rungs, [round(x, 2) for x in p]

    def observe(self, est, tab):
        RUNGS = REFERENCE_RUNGS  # noqa: N806
        for r in range(len(RUNGS)):
            s, n = est.get(str(r), [0, 0])
            c = tab.get(str(r))
            est[str(r)] = (
                [round(0.75 * s + c[0], 3), round(0.75 * n + c[1], 3)]
                if c
                else [round(0.95 * s, 3), round(0.95 * n, 3)]
            )


def test_ladder_matches_the_orchestrator():
    assert RUNGS == REFERENCE_RUNGS


@pytest.mark.parametrize("hold", [(), (7,), (3, 4), (0,), (11,), (5, 11)])
@pytest.mark.parametrize("band", [(0.2, 0.8), (0.3, 0.7)])
def test_matches_reference_over_simulated_training(hold, band):
    """Drive both controllers with the same outcomes for many steps; every plan and count must agree."""
    lo, hi = band
    rng = random.Random(hash((hold, band)) & 0xFFFF)
    for trial in range(20):
        # optional calibration counts, as the orchestrator's calibration probe produces
        calibration = {str(r): [rng.randint(0, 8), 8] for r in (0, 1, 2, 3, 5, 7, 11)} if trial % 2 else {}
        reference, ref_est = _Reference(lo, hi, hold), copy.deepcopy(calibration)
        ported = LadderCurriculum(lo=lo, hi=hi, holdout_rungs=hold, counts={"s": copy.deepcopy(calibration)})
        skill = rng.uniform(0.0, 12.0)  # rungs below the skill mostly succeed
        for step in range(60):
            n_groups = rng.choice([1, 2, 4, 5, 6, 8])
            assert ported.estimate("s") == reference.estimate(ref_est)
            rungs, p = ported.plan("s", n_groups)
            assert (rungs, p) == reference.plan(ref_est, n_groups)
            assert len(rungs) == n_groups and not set(rungs) & set(hold)
            table = {}
            for rung in rungs:
                chance = min(0.97, max(0.03, 0.5 + 0.25 * (skill - rung)))
                wins = sum(rng.random() < chance for _ in range(8))
                c = table.setdefault(str(rung), [0, 0])
                c[0] += wins
                c[1] += 8
            if step % 11 == 10:
                table = {}  # a step whose episodes were all lost: everything only fades
            ported.observe("s", table)
            reference.observe(ref_est, table)
            assert ported.counts["s"] == ref_est
            skill += rng.uniform(-0.1, 0.4)


def test_estimate_rules():
    c = LadderCurriculum(counts={"s": {"0": [7, 8], "1": [3.5, 3.9]}})
    p = c.estimate("s")
    assert p[0] == pytest.approx(7.5 / 9)  # measured: (s + 0.5) / (n + 1)
    assert p[1] == pytest.approx(0.7 * p[0])  # fewer than 4 episodes: untried, a near neighbour
    assert p[2] == pytest.approx(0.4 * p[1])  # a second untried rung in a row
    held = LadderCurriculum(holdout_rungs=[1], counts={"s": {"0": [7, 8]}})
    q = held.estimate("s")
    assert q[1] == q[0]  # an unmeasured held-out rung carries the rung below forward
    assert q[2] == pytest.approx(0.7 * q[0])  # ... and does not count as an untried rung
    measured = LadderCurriculum(holdout_rungs=[1], counts={"s": {"0": [7, 8], "1": [0, 8]}})
    assert measured.estimate("s")[1] == pytest.approx(0.5 / 9)  # calibration may measure a held-out rung
    assert LadderCurriculum().estimate("unknown")[0] == 0.5


def test_plan_allocation_and_holdout():
    counts = {"s": {"0": [8, 8], "1": [6, 8], "2": [4, 8], "3": [3, 8], "4": [0, 8]}}
    c = LadderCurriculum(counts=counts)
    # estimates 0.94, 0.72, 0.5, 0.39, 0.06: candidates in [0.3, 0.8] are rungs 1, 2, 3; nearest 0.6 is rung 2
    assert c.plan("s", 6)[0] == [3, 3, 1, 2, 2, 1]  # two probes on the hardest, every third on the easiest
    assert c.plan("s", 4)[0] == [3, 2, 1, 2]  # fewer than five groups: one probe
    held = LadderCurriculum(holdout_rungs=[3], counts=counts)
    assert held.plan("s", 6)[0] == [2, 2, 1, 2, 2, 1]  # hardest == nearest 0.6: one probe
    with pytest.raises(ValueError):
        LadderCurriculum(holdout_rungs=range(len(RUNGS)))


def test_plan_outside_the_band():
    counts = {str(r): [8, 8] for r in range(10)}
    # nothing in the band; rung 10 (0.83) is the rung nearest one half and is above the band: it and the next
    solved = LadderCurriculum(counts={"s": {**counts, "10": [7, 8], "11": [0, 8]}})
    assert solved.plan("s", 5)[0] == [11, 11, 10, 10, 10]
    # the rung nearest one half is below the band: all groups go to it
    stuck = LadderCurriculum(counts={"s": {**counts, "10": [8, 8], "11": [1, 8]}})
    assert stuck.plan("s", 5)[0] == [11] * 5
    # Reference quirk, kept: an exact tie in distance from one half goes to the lowest rung.
    tie = LadderCurriculum(counts={"s": {**counts, "10": [8, 8], "11": [0, 8]}})
    assert tie.plan("s", 5)[0] == [1, 1, 0, 0, 0]


def test_state_roundtrip_and_success_table():
    c = LadderCurriculum(holdout_rungs=[7])
    c.observe("a:b", {"0": [5, 8]})
    restored = LadderCurriculum(holdout_rungs=[7])
    restored.load_state_dict(copy.deepcopy(c.state_dict()))
    assert restored.counts == c.counts and restored.plan("a:b", 5) == c.plan("a:b", 5)
    outcomes = [
        {"sim": "a", "task": "b", "rung": 2, "success": True, "reward": 1.0},
        {"sim": "a", "task": "b", "rung": 2, "success": False, "reward": 0.5},
        {"sim": "a", "task": "c", "rung": 0, "success": False, "reward": 0.0},
    ]
    assert success_table(outcomes) == {
        stream_key("a", "b"): {"2": [1, 2, 0.75]},
        stream_key("a", "c"): {"0": [0, 1, 0.0]},
    }
