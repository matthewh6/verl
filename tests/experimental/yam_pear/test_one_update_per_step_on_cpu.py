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
"""The shipped trainer config takes exactly one optimizer update per training step.

verl steps the optimizer once per mini-batch (``EngineWorker.train_mini_batch`` -> ``train_batch``:
zero_grad, gradient accumulated over the micro-batches, one ``optimizer_step``). A step's samples are
(groups x rollout.n x turns) rows, the trainer pads them to a multiple of the mini-batch
(``_get_required_batch_multiple`` / ``upsample_batch_to_divisible_size``) and asks for mini-batches of
``ppo_mini_batch_size x rollout.n`` rows. These tests pin the config to "one mini-batch whatever the
episode lengths" using verl's own arithmetic, and check that the padding rows do not change the loss.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict

import verl.utils.tensordict_utils as tu
from verl.experimental.yam_pear.curriculum import RUNGS
from verl.experimental.yam_pear.dataloader import parse_streams
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.config import omega_conf_to_dataclass

CONFIG_DIR = Path(__file__).resolve().parents[3] / "examples" / "yam_pear" / "config"


@pytest.fixture(scope="module")
def config():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="yam_pear_grpo")
    GlobalHydra.instance().clear()
    return cfg


def padded_rows(cfg, real_rows: int, dp_size: int) -> int:
    """Rows the actor update receives: the trainer's own batch multiple, applied as upsample_batch does."""
    trainer = SimpleNamespace(config=cfg, use_critic=False, global_steps=1)
    multiple = PPOTrainer._get_required_batch_multiple(trainer, dp_size)
    return real_rows + (-real_rows) % multiple


def test_the_mini_batch_covers_the_longest_possible_step(config):
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    groups = sum(n for _, _, n in parse_streams(config.data.custom_dataloader.streams))
    assert config.data.train_batch_size == groups
    most_rows = groups * rollout.n * max(turn_cap for _, turn_cap in RUNGS)  # every episode at the longest cap
    assert actor.ppo_mini_batch_size * rollout.n >= most_rows
    assert actor.ppo_epochs == 1
    assert config.trainer.v1.trainer_mode == "sync"  # parameter_sync_step == 1: one _update_actor per step


def test_the_config_is_accepted_with_a_mini_batch_above_train_batch_size(config):
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert actor.ppo_mini_batch_size > config.data.train_batch_size and actor.use_dynamic_bsz
    actor.validate(config.trainer.n_gpus_per_node, config.data.train_batch_size, config.actor_rollout_ref.model)
    rollout = config.actor_rollout_ref.rollout
    # engine_workers asserts these mirror the actor's; a micro-batch must hold the longest sample
    assert rollout.log_prob_use_dynamic_bsz is True
    assert rollout.log_prob_max_token_len_per_gpu == actor.ppo_max_token_len_per_gpu
    assert actor.ppo_max_token_len_per_gpu >= rollout.prompt_length + rollout.response_length


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("turns_per_episode", [1, 4, 7, 25])
def test_one_optimizer_update_for_any_episode_lengths(config, dp_size, turns_per_episode):
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    real_rows = config.data.train_batch_size * rollout.n * turns_per_episode - (3 if turns_per_episode > 1 else 0)
    rows = padded_rows(config, real_rows, dp_size)
    # EngineWorker.train_mini_batch, per data-parallel rank
    mini_batch_size = actor.ppo_mini_batch_size * rollout.n  # what PPOTrainer._update_actor sends
    assert mini_batch_size % dp_size == 0
    rows_per_rank, mini_batch_per_rank = rows // dp_size, mini_batch_size // dp_size
    assert rows_per_rank // mini_batch_per_rank * actor.ppo_epochs == 1  # total_num_iterations
    data = TensorDict({"x": torch.zeros(rows_per_rank)}, batch_size=rows_per_rank)
    mini_batches = list(tu.make_iterator(data, mini_batch_size=mini_batch_per_rank, epochs=actor.ppo_epochs, seed=1))
    assert len(mini_batches) == 1 and len(mini_batches[0]) == rows_per_rank  # one train_batch -> one optimizer step


def test_padding_rows_do_not_change_the_token_mean_loss_or_its_gradient(config):
    """Padding rows have an all-zero loss mask: with token-mean they are in neither the sum nor the count."""
    assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
    torch.manual_seed(0)
    real, padding, width = 5, 11, 7
    weights = torch.randn(real + padding, width, requires_grad=True)
    target = torch.randn(real + padding, width)
    mask = torch.zeros(real + padding, width)
    mask[:real] = (torch.rand(real, width) > 0.3).float()
    mask[0, 0] = 1.0

    def loss_of(rows):
        loss_mat = (weights[:rows] * target[:rows]) ** 2
        # the engine passes the mini-batch's global token count: loss_mask.sum()
        return agg_loss(loss_mat, mask[:rows], "token-mean", batch_num_tokens=mask[:rows].sum())

    with_padding, without = loss_of(real + padding), loss_of(real)
    assert torch.allclose(with_padding, without)
    (grad_padded,) = torch.autograd.grad(with_padding, weights)
    (grad_real,) = torch.autograd.grad(without, weights)
    assert torch.allclose(grad_padded, grad_real) and torch.all(grad_padded[real:] == 0)
