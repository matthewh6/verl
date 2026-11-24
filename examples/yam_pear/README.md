# YAM/PEAR: skill-ladder GRPO of a vision-language model on simulated YAM stations

Branch `tri-pear`. This recipe ports a custom single-GPU robot-RL loop (`grpo_loop_adapt.py` and its
simulator drivers) onto verl: a vision-language model (Qwen3.8-27B-PEAR, LoRA) controls a simulated bimanual
YAM station turn by turn, and is trained with GRPO on the episode's best shaped reward, with an adaptive
curriculum over start states.

**Status: the recipe has not been run end to end with the real model.** No GPU run of any kind was made.
What was and was not checked is listed under [What is verified](#what-is-verified).

## Layout

`recipe/` in this repository is a git submodule of `verl-project/verl-recipe`, so files cannot be committed
there on a branch of the verl fork. The recipe lives in the fork itself:

| Path | What |
| --- | --- |
| `verl/experimental/yam_pear/sim_bridge.py` | Standalone script (stdlib only) run inside a simulator's own venv: JSON lines over stdin/stdout around the driver's `Episode`. Also the `fake` simulator. |
| `verl/experimental/yam_pear/sim_client.py` | Async client that spawns the bridge with a per-simulator command. |
| `verl/experimental/yam_pear/protocol.py` | The station conversation: system prompt, short-context prompt, action parsing. |
| `verl/experimental/yam_pear/agent_loop.py` | `YamPearAgentLoop`, registered as `yam_pear_agent`. |
| `verl/experimental/yam_pear/curriculum.py` | `LadderCurriculum`: the `estimate` / `plan` / `observe` controller and `RUNGS`. |
| `verl/experimental/yam_pear/artifacts.py` | `EpisodeArtifacts`: per-episode `episode.json` / `turns.json` and sampled camera frames. |
| `verl/experimental/yam_pear/dataloader.py` | Episode rows, the outcome log, `CurriculumDataLoader` (`data.custom_dataloader`). |
| `verl/experimental/yam_pear/prompts/` | `station_system_prompt.md`, `raiden_environment.md`, `abc_environment.md` (copied unchanged), `fake_environment.md`. |
| `verl/experimental/yam_pear/config/` | Agent loop configs: `agent_loop.yaml` (raiden + ABC), `agent_loop_fake.yaml`. |
| `examples/yam_pear/config/yam_pear_grpo.yaml` | The trainer config. |
| `examples/yam_pear/run_qwen3_8_27b_pear_fsdp.sh` | Launch script. |
| `examples/yam_pear/create_dataset.py` | Writes `train.parquet` / `val.parquet`. |
| `tests/experimental/yam_pear/` | CPU tests. |

The only change to existing verl files is one line in `setup.py` (`experimental/*/prompts/*.md` added to
the package data, so the prompt files are installed with the package).

## How it works

**Simulators.** The two simulators (TRI raiden-sim, python 3.11; ABC sim, python 3.12) each have their own
virtual environment, which is not verl's. The agent loop starts `sim_bridge.py` with that environment's
interpreter (command, working directory and environment variables per simulator in `agent_loop.yaml`) and
talks to it over pipes: `reset` (sim, task, seed, prepose) returns the instruction and an observation;
`step` (action) returns feedback, observation, reward, success, done; camera frames are base64 JPEGs. The
bridge imports the existing driver module (`raiden_pear.py` / `abc_pear_cur.py`) from its directory and wraps
its `Episode`; the drivers are not copied. One bridge process serves one episode.

**Episode.** `YamPearAgentLoop.run` follows the drivers' `main()`: system prompt from
`station_system_prompt.md` + the simulator's environment text + the instruction; each turn the measured
state, the reminder, the previous execution feedback and three labelled camera frames; the answer must end
with one fenced json action; end reasons `success`, `max_turns` (the rung's turn cap), `parse`, `budget`
(completion cut at the token limit, or a prompt that does not fit). The episode's reward is the simulator's
best shaped reward (1 = success).

**Short context, and how verl represents it.** Every turn's prompt is rebuilt: system + the last K = 4 turns
as text (state and answer, no images, no reasoning) + this turn's frames. Consecutive prompts therefore do
not share a growing prefix, and an episode cannot be one `prompt | response` token sequence, which is what a
single `AgentLoopOutput` (and verl's context merging, `ct_merge_context_msg`) describes. The supported
form in the V1 trainer is for `run` to return **a list of `AgentLoopOutput`, one per turn**:
`AgentLoopWorkerTQ._agent_loop_postprocess` stores each output as its own sample under
`{uid}_{session_id}_{index}` and gives all of them the final output's reward, and
`compute_advantage_for_multi_trajectories` computes the GRPO advantage once per session, from the final
outputs of the prompt's `rollout.n` sessions, and broadcasts it to every turn of the session. That is the
same credit assignment as the loop being ported (one group-normalised advantage per episode, applied to all
of its turns). Each output is a single generation, so the sampler's top-k head per output is accepted by
score centering (`rollout.topk_log_probs`), which is otherwise limited to single-turn loops.

**Data.** One row is (sim, task, seed, rung). `rollout.n = 8` rollouts of a row are the group of same-seed
rollouts. `create_dataset.py` writes `val.parquet` (held-out seeds on the evaluation rungs; validation
metrics are reported per `sim:task/rungNN`) and a static `train.parquet` that is only used without the
curriculum.

**Curriculum.** verl has no curriculum interface. Its hook for a custom prompt source is
`data.custom_dataloader`, which the trainer only pulls batches from; the feedback path is added in the
recipe: the agent loop appends every finished episode to an outcome log (`outcome_dir`, JSON lines), and
`CurriculumDataLoader` reads the new lines each time it plans a step. Planning is `LadderCurriculum`, a
port of `estimate` / `plan` / `observe`, including the held-out rungs (`holdout_rungs`) and the per-task
group allocation (`streams: "sim:task:groups,..."`), with the orchestrator's seed scheme. With
`calibrate: true` the first validation pass also runs calibration episodes, whose outcomes seed the
estimates. Plans and counts are written to `<outcome_dir>/curriculum.jsonl`; the loader implements
`state_dict` / `load_state_dict` for resume.

Constraints of this hook: `outcome_dir` must be on a filesystem that the trainer and all agent loop workers
see, and the loader needs the synchronous trainer (`trainer.v1.trainer_mode=sync`), where all episodes of a
step have finished before the next batch is fetched. `data.train_batch_size` must equal the number of groups
per step (the sum over `streams`).

## Hardware

The loop being ported runs on ONE 80 GB H100 and alternates: vLLM up, rollouts, vLLM down, one trainer
step. A bf16 27B model is about 54 GB, so sampler and trainer cannot be on the card together.

What verl offers, from its code and docs (none of it run here):

* `trainer.v1.trainer_mode=sync` is verl's version of that alternation: the vLLM engine and the FSDP trainer
  are colocated on the same GPUs ("hybrid"), vLLM is put to sleep after sampling and woken (with a weight
  update) after the step. With unmerged LoRA the sleeping engine keeps its base weights in host memory
  (level-1 sleep) and only the adapter is synced; `actor_rollout_ref.model.lora.resync_base=true` discards
  the base on sleep and re-syncs it on every update instead. `actor.fsdp_config.param_offload=true` moves
  the trainer's weights to host memory while rollouts run. This is the configuration in
  `yam_pear_grpo.yaml` (`trainer.n_gpus_per_node=1`), and it is the only way verl has to fit one GPU.
* Whether it actually fits a 27B model in 80 GB is **not verified**: it needs each phase alone to fit (the
  trainer with gradient checkpointing on prompts of a few thousand tokens with three images), no transient
  overlap of the two copies during wake-up and weight sync, and roughly 110 GB of host memory for the two
  parked copies. `rollout.gpu_memory_utilization` (0.45 in the config) is a guess that will need tuning.
* The safe placement is 2 GPUs with the same synchronous mode (`NGPUS=2`): the trainer is then sharded over
  both. `separate_async` (a rollout GPU and a trainer GPU that never switch) is the other 2-GPU layout verl
  has, but rollouts then run ahead of training, which the curriculum dataloader does not support.

## One update per step

A step's training samples are (groups x 8 x turns) rows, between 128 and 3200 depending on how long the
episodes ran. verl takes one optimizer step per *mini-batch*: `train_batch` zeroes the gradient, runs
forward/backward over the mini-batch's micro-batches, accumulating the gradient, and steps the optimizer
once. So "one update per step" means "one mini-batch per step". The trainer only offers a fixed mini-batch
size, `ppo_mini_batch_size x rollout.n` rows, and pads every batch to a multiple of it; there is no
"whole batch" setting in the config (the worker's `num_mini_batch` argument is not reachable from it).

The config therefore sets `ppo_mini_batch_size: 400` = 16 groups x 25, the longest turn cap in the ladder,
so that 3200 rows is the most a step can have and every step is padded to exactly one mini-batch.
`use_dynamic_bsz: true` is needed for verl to accept a mini-batch larger than `train_batch_size`;
micro-batches are then packed up to `ppo_max_token_len_per_gpu` tokens (it must cover the longest sample).
If you change `streams` or the ladder, keep `ppo_mini_batch_size = data.train_batch_size x longest turn
cap`; `test_one_update_per_step_on_cpu.py` fails otherwise.

What the padding does. Padding rows are synthetic 128-token sequences with an all-zero loss mask and zero
reward, under a uid of their own (so they are in no GRPO group). With `loss_agg_mode: token-mean` the loss is
`sum(masked token losses) / (loss-mask tokens in the mini-batch)`: padding rows add nothing to the sum and
nothing to the count, so the loss and its gradient are the same as for a true single batch of the real rows
(checked on CPU for verl's `agg_loss`). This would not hold for the `seq-mean-*` modes, which divide by the
number of rows. The cost is compute: with short episodes most of the 3200 rows are padding, each a
128-token forward/backward (about 10-15% extra tokens at 6 turns per episode). Not verified on a GPU: that
padding rows (which carry no images) batch correctly with multimodal rows in one micro-batch, and the
memory of token-packed micro-batches for this model (`ppo_max_token_len_per_gpu` may need lowering, but not
below `prompt_length + response_length`).

## Run

```bash
# once: the environment (GPU node)
uv sync --frozen --extra vllm --extra fsdp

# smoke run with the fake simulator (still needs the model and a GPU)
SIM=fake MODEL_PATH=/path/to/Qwen3.8-27B-PEAR bash examples/yam_pear/run_qwen3_8_27b_pear_fsdp.sh \
    trainer.total_training_steps=2 data.custom_dataloader.cal_seeds=1

# the real simulators
PEAR_ROOT=/data/users/<you>/pear ABC_SIM_DIR=/path/to/abc-sim/abc \
MODEL_PATH=$PEAR_ROOT/work/models/Qwen3.8-27B-PEAR \
bash examples/yam_pear/run_qwen3_8_27b_pear_fsdp.sh data.custom_dataloader.holdout_rungs=7
```

`PEAR_ROOT` must hold `raiden/venv`, `raiden/src`, `raiden/assets_local` and `tools/cur3` (the drivers and
`prepose.py`); `ABC_SIM_DIR` is the ABC checkout with its `.venv`. Edit
`verl/experimental/yam_pear/config/agent_loop.yaml` if the layout differs. `EGL_GPU` selects the GPU MuJoCo
renders on (default 0; the old scripts used 3).

No API key is involved: verl's vLLM replicas are in-process engines, not an HTTP server, so nothing in the
recipe reads or stores the node's key file.

To check a simulator environment without verl:

```bash
echo '{"cmd": "reset", "sim": "raiden", "task": "pick_cube", "seed": 3, "prepose": "0,6,grasped"}' | \
  MUJOCO_GL=egl RAIDEN_SIM_ASSETS=$PEAR_ROOT/raiden/assets_local PYTHONPATH=$PEAR_ROOT/raiden/src \
  $PEAR_ROOT/raiden/venv/bin/python verl/experimental/yam_pear/sim_bridge.py \
    --backend driver --driver-dir $PEAR_ROOT/tools/cur3/raiden --driver-module raiden_pear | head -c 600
```

## Episode artifacts

Off by default. Set `artifacts.dir` in the agent loop config (`agent_loop.yaml` reads it from the
environment) to get the old pipeline's per-episode files back:

```bash
YAM_PEAR_ARTIFACTS_DIR=/path/to/run/episodes \      # unset = disabled
YAM_PEAR_ARTIFACT_FRAMES=sample \                   # training frames: none | sample | all (default sample)
YAM_PEAR_ARTIFACT_VAL_FRAMES=false \                # frames of all validation/calibration episodes
bash examples/yam_pear/run_qwen3_8_27b_pear_fsdp.sh
```

```text
<dir>/step_<NNN>/s<seed>_g<i>/        training; NNN = the trainer's global step (1-based), i = rollout in the group
<dir>/eval_<NNN>/s<seed>_g<group>/    validation; group = 100 x stream + rung, as the old probe named them
<dir>/calibration/s<seed>_g<group>/   calibration episodes (first validation pass)
    episode.json  turns.json  system_prompt.md  [turnNN_<camera>.jpg ...  final_<camera>.jpg]
```

* **Records, for every episode.** `episode.json` and `turns.json` use the field names of `raiden_pear.py` /
  `abc_pear_cur.py` `main()`, without token ids and log-probs. `episode.json`: `task`, `seed`, `start`,
  `randomization_request`, `instruction`, `prepose` (the simulator's report, with `radius` / `height` /
  `stage`), `lift` / `lift_pick` for ABC, `end_reason`, `success`, `max_reward`, `num_turns`, `effort`,
  `history_k`, `max_tokens`, `wall_s`, `completion_tokens`; and, new, `sim`, `rung`, `split`,
  `global_steps`, `plan_id` (the curriculum's 0-based step), `uid` (the group), `session_id`, `frames`.
  `turns.json`, per turn: `turn`, `state`, `user_text`, `reasoning`, `answer`, `finish_reason`, `usage`,
  `action`, `feedback`, `result`, `success`, `max_reward` (the reward so far), `model_seconds`,
  `sim_seconds`, `images`, and `parse_error` on the turn that ended a `parse` / `budget` episode. Not
  written: `randomization`, `sim_steps*`, `lift_body`, the dense action recordings. A rollout that fails
  (server or simulator error) writes nothing.
* **Frames, for a sample.** With `frames: sample`, the three JPEGs of every turn (the bytes the simulator
  sent, before verl's resize) are kept for one group per stream per training step: all 8 rollouts of the
  group whose position in its stream equals a hash of (curriculum step, stream), so every worker picks the
  same group without coordination and the choice moves around between steps. This needs rows planned by the
  curriculum dataloader; with the static `train.parquet`, `sample` keeps no frames. `val_frames: true` keeps
  the frames of all validation and calibration episodes.
* **Writing.** Each episode directory is written under a temporary name and renamed into place, so readers
  never see half an episode and any number of agent loop workers can write to one directory, which must be
  on a filesystem they share. A prompt that is run again (after a resume) replaces its directory.

**Report tools and videos.** The behaviour-report tools read these files as they are:
`behavior_report/extract_inc.py` globs `<run>/episodes/step_*/*/{episode.json,turns.json}` and keeps each
turn's `action` and `result`, so point `artifacts.dir` at `<run>/episodes`; `pipeline.py` uses
`task`, `prepose`, `success`, `max_reward` and `num_turns`. Videos need nothing saved by verl beyond the
records: `replay_video.py` rebuilds the scene from `episode.json` (`task`, `seed`, `start`, `prepose`,
`lift`, `lift_pick`), replays the recorded `action` of every turn in the simulator and renders it, then
prints the replayed success and reward next to the recorded ones. Two things to adapt in those tools:
step directories are numbered by verl's global step, which starts at 1 (the old loop's `step_000` is
`step_001` here; `plan_id` in `episode.json` is the old number), and only episodes of the sampled groups
have saved frames to show next to a replay. The tools themselves were not run against these records (their
paths are hard-wired to the node); the tests check the field names, and that replaying the recorded actions
from `episode.json` reproduces the recorded outcome in the fake simulator.

## Tests

```bash
uv sync --frozen --extra cpu
.venv/bin/python -m pytest tests/experimental/yam_pear -q
```

Covered on CPU: the bridge protocol against the fake simulator in a subprocess (and the driver backend
against a stub driver); action parsing and end reasons; short-context prompt building; the curriculum
controller against a verbatim copy of the reference `estimate` / `plan` / `observe` over simulated training;
the dataloader through verl's `create_custom_dataloader`; the agent loop end to end with a mocked LLM server
and the fake simulator, including verl's own per-output storage (`_agent_loop_postprocess`) and
`compute_advantage_for_multi_trajectories` on what the loop returns. Also: one optimizer update per step from
the shipped config; episode records complete and re-simulable, frame sampling of exactly one group per stream
per step, artifacts off by default.

`test_recorded_episode_on_cpu.py` is skipped unless `YAM_PEAR_MODEL_DIR` (the PEAR tokenizer/processor
files; no weights are loaded) and `YAM_PEAR_EPISODE_DIR` (an episode recorded by `raiden_pear.py --token-ids`)
are set. It checks that this recipe's prompts, tokenized by verl, equal the prompt token ids vLLM reported.

## What is verified

Checked, on CPU only:

* The CPU tests above.
* The trainer config composes and passes `verl.utils.config.validate_config` (with the vLLM-only LoRA rank
  check stubbed, since vLLM is not installed in the CPU environment), and verl's dataloader factory builds
  the curriculum loader from it.
* With the real PEAR tokenizer, processor and chat template (no weights): verl's Continuous Token builder
  handles the station messages, and for one recorded 6-turn raiden episode of the old pipeline every turn's
  prompt token ids equal vLLM's, the system prompt and user texts are identical, and the answer extraction
  agrees with vLLM's reasoning parser.
* The bridge and the agent loop against the real raiden-sim driver (local checkout, software rendering, a
  scripted policy instead of the model): reset with a prepose start state, steps, reward, turn cap.

Not verified (needs the GPU node):

* Any run with the model: rollouts through verl's vLLM replicas, the LoRA trainer, weight sync, memory fit.
* The ABC simulator through the bridge (only a stub driver was used for the driver backend's wiring).
* Score centering on this data: that the sampler's top-32 head is returned for multimodal requests, and that
  trainer and sampler log-probs agree on these prompts (watch `training/rollout_probs_diff_*`).
* `target_modules` as a PEFT regex through verl's FSDP LoRA path and its sync into vLLM.
* That vLLM receives pixel-identical frames to the old pipeline. verl extracts images with
  `qwen_vl_utils` (smart resize) before the processor; the 640x360 raiden frames become 640x352 there. The
  token counts match the old pipeline's; sampler and trainer inside verl see the same resized frames.
* Checkpoint resume of the curriculum loader inside the trainer (only `state_dict` round trips are tested).

## Differences from the pipeline being ported

* **Loss.** verl's own GRPO, REINFORCE with truncated token importance weights and score centering
  (`rollout_correction`: `bypass_mode`, `loss_type=reinforce`, `rollout_is=token`, threshold 2,
  `score_centering`); `grpo_trainer_once.py` is not ported. verl normalises a group's advantage as
  `(r - mean) / (std + 1e-6)` with the sample standard deviation (n - 1); the old loop used the population
  standard deviation, `1e-4`, and zeroed groups whose std was below `1e-3`.
* **Turns trained on.** Every turn of every episode is a sample. The old trainer had a
  `--turns-per-episode 4` option.
* **Updates per step.** One optimizer update per training step, as before; see
  [One update per step](#one-update-per-step) for how.
* **Failed rollouts.** A server or simulator failure raises; verl marks the rollout as failed. The old loop
  redid such episodes up to four times (`trainer.v1.sampler.sync_refill_failed_groups` is verl's refill).
* **Calibration and evaluation** run as verl validation (`val_before_train`, `test_freq`), on the seeds and
  rungs of the old `probe`; calibration episodes appear under `calibration/...` in the validation metrics
  of the first pass.
* **Episode artifacts** are opt-in ([Episode artifacts](#episode-artifacts)): records for every episode,
  frames for a sample. The old loop saved every frame of every episode.
* **Not ported:** the vLLM/tmux process management and GPU sharing with another tenant, `--object-state`,
  `--record-dense`, full-history mode (`--history-k -1`), `ABC_EXTRA`, and `expert_check.py` /
  `stage_check.py`.
