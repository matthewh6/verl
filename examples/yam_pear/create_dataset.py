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
"""Write the train/val parquet files of the YAM/PEAR recipe.

One row is one episode start (sim, task, seed, rung); ``actor_rollout_ref.rollout.n=8`` rollouts of a row
are the group of same-seed rollouts GRPO normalises within.

* ``val.parquet``: the held-out evaluation, ``--eval-seeds`` seeds from ``--eval-seed0`` on ``--eval-rungs``
  for every stream. This is what the curriculum dataloader validates on.
* ``train.parquet``: a STATIC ladder, ``--train-seeds`` seeds on every trainable rung of every stream. It is
  only used when training without the curriculum dataloader (verl's built-in dataset, uniform over rungs);
  with ``data.custom_dataloader`` the training rows are planned on the fly and this file is not read.

    python examples/yam_pear/create_dataset.py --local_save_dir ~/data/yam_pear
"""

import argparse
import os

import pandas as pd

from verl.experimental.yam_pear.curriculum import RUNGS
from verl.experimental.yam_pear.dataloader import make_row, parse_rungs, parse_streams, probe_rows


def build_train_rows(streams, holdout, seed0, n_seeds):
    """Static training rows: every trainable rung of every stream, ``n_seeds`` seeds each."""
    rows = []
    for si, (sim, task, _) in enumerate(streams):
        for rung in range(len(RUNGS)):
            if rung in holdout:
                continue
            for i in range(n_seeds):
                seed = seed0 + si * 100000 + rung * 1000 + i
                rows.append(make_row(sim, task, seed, rung, split="train", index=len(rows)))
    return rows


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local_save_dir", default="~/data/yam_pear")
    parser.add_argument("--streams", default="raiden:pick_cube:5,raiden:pick_banana:5,abc:put_relative:6")
    parser.add_argument("--eval-rungs", default="3,7,11")
    parser.add_argument("--eval-seeds", type=int, default=16)
    parser.add_argument("--eval-seed0", type=int, default=100)
    parser.add_argument("--holdout-rungs", default="", help="rungs left out of train.parquet")
    parser.add_argument("--train-seeds", type=int, default=64)
    parser.add_argument("--train-seed0", type=int, default=1_000_000)
    args = parser.parse_args()

    streams = parse_streams(args.streams)
    out = os.path.expanduser(args.local_save_dir)
    os.makedirs(out, exist_ok=True)
    val = probe_rows(streams, parse_rungs(args.eval_rungs), args.eval_seed0, args.eval_seeds)
    train = build_train_rows(streams, set(parse_rungs(args.holdout_rungs)), args.train_seed0, args.train_seeds)
    pd.DataFrame(val).to_parquet(os.path.join(out, "val.parquet"))
    pd.DataFrame(train).to_parquet(os.path.join(out, "train.parquet"))
    print(f"wrote {len(train)} train rows and {len(val)} val rows to {out}")


if __name__ == "__main__":
    main()
