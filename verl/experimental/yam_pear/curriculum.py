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
"""Adaptive skill-ladder curriculum: the controller of ``grpo_loop_adapt.py`` (``estimate`` / ``plan`` / ``observe``).

Pure Python. Every task has the same ladder of start states (``RUNGS``): a scripted expert carries out the
first part of a top-down grasp and the model does the rest; rung 0 leaves only the lift, the last rung is
the whole task from Home. For every task the controller keeps a decayed success count per rung, and each
step gives the task's groups to rungs whose estimated success is inside a band, because group-relative
advantages only carry a gradient when a group of same-seed rollouts has mixed outcomes; a probe group or
two goes to the hardest such rung, or one rung past it once it works half the time. Held-out rungs are
never planned (they are still evaluated, which measures transfer between start states).
"""

from typing import Any, Iterable, Optional

# (start state for the scripted expert, "R,H[,stage]" or "x" = Home; turn cap)
RUNGS: list[tuple[str, int]] = [
    ("0,6,grasped", 4),
    ("0,6,around", 6),
    ("0.5,6,aligned", 8),
    ("1,6", 12),
    ("2.5,7", 12),
    ("4,8", 12),
    ("6,9", 12),
    ("8,10", 14),
    ("11,11", 14),
    ("15,12", 16),
    ("-1,0", 20),
    ("x", 25),
]

Counts = dict[str, list[float]]  # rung (as str) -> [decayed successes, decayed episodes]


def stream_key(sim: str, task: str) -> str:
    """The controller's name for a (simulator, task) stream."""
    return f"{sim}:{task}"


class LadderCurriculum:
    """Per-task success estimates per rung, and the per-step allocation of a task's groups to rungs.

    Args:
        lo: Lower edge of the success band (raised by 0.1 when choosing candidates).
        hi: Upper edge of the success band.
        holdout_rungs: Rungs that are never trained on.
        num_rungs: Length of the ladder, default ``len(RUNGS)``.
        counts: Initial per-stream counts, ``{stream: {rung: [successes, episodes]}}`` (e.g. a calibration).
    """

    def __init__(
        self,
        lo: float = 0.2,
        hi: float = 0.8,
        holdout_rungs: Iterable[int] = (),
        num_rungs: Optional[int] = None,
        counts: Optional[dict[str, Counts]] = None,
    ):
        self.lo = float(lo)
        self.hi = float(hi)
        self.holdout = {int(r) for r in holdout_rungs}
        self.num_rungs = len(RUNGS) if num_rungs is None else int(num_rungs)
        if not [r for r in range(self.num_rungs) if r not in self.holdout]:
            raise ValueError("every rung is held out; nothing can be trained on")
        self.counts: dict[str, Counts] = {}
        for stream, table in (counts or {}).items():
            self.counts[stream] = {str(r): [float(c[0]), float(c[1])] for r, c in table.items()}

    def estimate(self, stream: str) -> list[float]:
        """Success estimate per rung.

        Measured where there are at least 4 (decayed) episodes; a held-out rung carries the rung below it
        forward; any other untried rung is 0.7 x the rung below (0.4 x for a second untried rung in a row).
        """
        est = self.counts.get(stream, {})
        p: list[float] = []
        guessed = False
        for r in range(self.num_rungs):
            s, n = est.get(str(r), [0, 0])
            if n >= 4:
                p.append((s + 0.5) / (n + 1))
                guessed = False
            elif r in self.holdout:
                p.append(p[-1] if p else 0.5)
            else:
                p.append((0.4 if guessed else 0.7) * p[-1] if p else 0.5)
                guessed = True
        return p

    def plan(self, stream: str, n_groups: int) -> tuple[list[int], list[float]]:
        """The rungs for a task's groups this step, and the (rounded) estimates they were chosen from.

        Candidates are the trainable rungs estimated inside [lo + 0.1, hi]. The probe is the hardest
        candidate, or the next trainable rung above it once the hardest candidate's estimate is at least
        0.5, whatever the (possibly stale) estimate of that next rung says. Two groups go to the probe if
        there are at least five groups, the probe is not the rung nearest 0.6 and its estimate is at least
        0.15; otherwise one. Of the rest, every third group goes to the easiest candidate and the others to
        the candidate nearest 0.6.
        """
        p = self.estimate(stream)
        ok = [r for r in range(self.num_rungs) if r not in self.holdout]
        cand = [r for r in ok if self.lo + 0.1 <= p[r] <= self.hi]
        if not cand:  # nothing in the band: the rung nearest one half, and the next one up if it is solved
            r = min(ok, key=lambda r: abs(p[r] - 0.5))
            cand = [r] + ([min(x for x in ok if x > r)] if p[r] > self.hi and r < max(ok) else [])
        mid, hard, easy = min(cand, key=lambda r: abs(p[r] - 0.6)), max(cand), min(cand)
        # One rung past the hardest candidate once that one works half the time: a low estimate measured
        # many steps ago would otherwise keep the next rung from ever being tried again.
        above = [r for r in ok if r > hard]
        probe = above[0] if above and p[hard] >= 0.5 else hard
        n_probe = 2 if n_groups >= 5 and probe != mid and p[probe] >= 0.15 else 1
        rungs = [probe] * n_probe + [easy if i % 3 == 2 else mid for i in range(n_probe, n_groups)]
        return rungs, [round(x, 2) for x in p]

    def observe(self, stream: str, table: dict[str, list[float]]) -> None:
        """Fold one step's outcomes into the decayed counts.

        Args:
            stream: The stream the outcomes belong to.
            table: ``{rung: [successes, episodes, ...]}`` of this step; rungs not run are absent.

        A rung that was run keeps 3/4 of its old evidence; the others fade by 0.85 a step, so an estimate
        nobody refreshed counts as unmeasured (fewer than 4 episodes) after about ten steps.
        """
        est = self.counts.setdefault(stream, {})
        for r in range(self.num_rungs):
            s, n = est.get(str(r), [0, 0])
            c = table.get(str(r))
            if c:
                est[str(r)] = [round(0.75 * s + c[0], 3), round(0.75 * n + c[1], 3)]
            else:
                est[str(r)] = [round(0.85 * s, 3), round(0.85 * n, 3)]

    def state_dict(self) -> dict[str, Any]:
        """The controller state that a resumed run needs."""
        return {"counts": {s: {r: list(c) for r, c in t.items()} for s, t in self.counts.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict`."""
        self.counts = {
            s: {str(r): [float(c[0]), float(c[1])] for r, c in t.items()} for s, t in state["counts"].items()
        }


def success_table(outcomes: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    """``{stream: {rung: [successes, episodes, mean reward]}}`` of a set of episode outcomes."""
    out: dict[str, dict[str, list[float]]] = {}
    for o in outcomes:
        c = out.setdefault(stream_key(o["sim"], o["task"]), {}).setdefault(str(o["rung"]), [0, 0, 0.0])
        c[0] += bool(o["success"])
        c[1] += 1
        c[2] += float(o["reward"])
    return {s: {r: [c[0], c[1], round(c[2] / c[1], 3)] for r, c in sorted(t.items())} for s, t in out.items()}
