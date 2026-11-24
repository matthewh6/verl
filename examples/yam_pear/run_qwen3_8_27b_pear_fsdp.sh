#!/usr/bin/env bash
# Skill-ladder GRPO of Qwen3.8-27B-PEAR on simulated YAM stations (LoRA, score centering, vLLM rollouts).
# Run from the verl repo root. See examples/yam_pear/README.md; NOT yet run end to end with the real model.
#
#   PEAR_ROOT=/data/users/<you>/pear ABC_SIM_DIR=/path/to/abc-sim/abc \
#   MODEL_PATH=$PEAR_ROOT/work/models/Qwen3.8-27B-PEAR bash examples/yam_pear/run_qwen3_8_27b_pear_fsdp.sh
#
# Episode records and sampled frames (off unless set): YAM_PEAR_ARTIFACTS_DIR=<run>/episodes,
# YAM_PEAR_ARTIFACT_FRAMES=none|sample|all, YAM_PEAR_ARTIFACT_VAL_FRAMES=true|false. See the README.
# YAM_PEAR_VIDEOS=1 also writes a couple of videos per step to <run>/episodes/videos while training runs
# (YAM_PEAR_VIDEO_ARGS="--wandb-project <p> --wandb-run-id <id>" uploads them).
#
# Extra arguments are passed to hydra, e.g. data.custom_dataloader.holdout_rungs=7 trainer.total_training_steps=50
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-"${HOME}/models/Qwen3.8-27B-PEAR"}
DATA_DIR=${DATA_DIR:-"${HOME}/data/yam_pear"}
NGPUS=${NGPUS:-1}                       # 1: trainer and vLLM take turns on one GPU; see the README for 2
STREAMS=${STREAMS:-"raiden:pick_cube:5,raiden:pick_banana:5,abc:put_relative:6"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8_27b_pear_ladder_grpo}
SIM=${SIM:-real}                        # real | fake (the recipe's pure-Python simulator, for smoke runs)

# The simulators' own environments (agent_loop.yaml reads these in every agent loop worker).
if [ "${SIM}" = real ]; then
    : "${PEAR_ROOT:?set PEAR_ROOT to the PEAR work tree (raiden/venv, raiden/src, raiden/assets_local, tools/cur3)}"
    : "${ABC_SIM_DIR:?set ABC_SIM_DIR to the ABC simulator checkout (the directory holding .venv)}"
    export PEAR_ROOT ABC_SIM_DIR EGL_GPU=${EGL_GPU:-0}
    AGENT_CONFIG=verl/experimental/yam_pear/config/agent_loop.yaml
else
    STREAMS=${FAKE_STREAMS:-"fake:pick_cube:5,fake:pick_banana:5,fake:put_relative:6"}
    AGENT_CONFIG=verl/experimental/yam_pear/config/agent_loop_fake.yaml
fi

PYTHON=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    PYTHON=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi

[ -f "${DATA_DIR}/val.parquet" ] || "${PYTHON[@]}" examples/yam_pear/create_dataset.py --local_save_dir "${DATA_DIR}" --streams "${STREAMS}"

if [ "${YAM_PEAR_VIDEOS:-0}" != 0 ] && [ -n "${YAM_PEAR_ARTIFACTS_DIR:-}" ]; then
    # shellcheck disable=SC2086
    "${PYTHON[@]}" -m verl.experimental.yam_pear.videos "${YAM_PEAR_ARTIFACTS_DIR}" --watch 120 ${YAM_PEAR_VIDEO_ARGS:-} &
    VIDEO_PID=$!
    trap 'kill "${VIDEO_PID}" 2>/dev/null || true' EXIT
fi

"${PYTHON[@]}" -m verl.trainer.main_ppo \
    --config-path="${PWD}/examples/yam_pear/config" --config-name=yam_pear_grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.custom_dataloader.streams="'${STREAMS}'" \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_CONFIG}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "${RAY[@]}" \
    "$@"
