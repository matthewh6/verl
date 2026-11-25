#!/usr/bin/env bash
# Online GRPO | Qwen3.5-0.8B | single GPU | Reasoning Gym suite, with every training batch archived
# (token ids, rewards, log-probs, advantages) and an HF checkpoint every SAVE_FREQ steps.
# usage: run_online.sh <name> [hydra overrides...]
set -euo pipefail
NAME=$1; shift
P=${TC_P:-$HOME/projects/tutor-curriculum}; V=${TC_V:-$P/ext/verl-replay}; PY=${TC_PY:-$P/ext/verl/.venv/bin/python}
D=${TC_D:-$HOME/datasets/tutor_rg}; RB=${TC_R:-$P/runs/replay}; BASE=${TC_VMODEL:-$HOME/datasets/models/Qwen3.5-0.8B}; VMEM=${TC_VLLM_MEM:-0.35}; EMEM=${TC_EVAL_MEM:-0.4}; OUT=$RB/$NAME
RT=${TC_RAYTMP:-/tmp/ray_tc}; mkdir -p $OUT $RT
export TC_OUTCOMES=$OUT/outcomes.jsonl PYTHONPATH=$V:$P:${PYTHONPATH:-} RAY_TMPDIR=$RT WANDB_MODE=disabled PYTHONUNBUFFERED=1 TRITON_DEFAULT_FP_FUSION=1
cd $V
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files=$D/train.parquet data.val_files=$D/val.parquet \
  data.train_batch_size=32 data.max_prompt_length=1024 data.max_response_length=400 \
  data.filter_overlong_prompts=True data.truncation=error data.shuffle=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  +data.custom_dataloader._target_=verl_rg.loader.CurriculumLoader +data.custom_dataloader.sampler=uniform \
  reward.custom_reward_function.path=$P/verl_rg/reward.py reward.custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path=$BASE \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=2e-6 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
  "actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]" \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.prompt_length=1024 actor_rollout_ref.rollout.response_length=400 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=$VMEM \
  actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.val_kwargs.n=4 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  trainer.critic_warmup=0 "trainer.logger=[console,tensorboard]" trainer.project_name=replay trainer.experiment_name=$NAME \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.balance_batch=False trainer.default_local_dir=$OUT/ckpt \
  trainer.val_before_train=True trainer.save_freq=${SAVE_FREQ:-20} trainer.test_freq=20 trainer.total_training_steps=120 \
  +trainer.rollout_archive_dir=$OUT/archive \
  ray_kwargs.ray_init.runtime_env.py_executable=null \
  ${TC_KV_BYTES:+"+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$TC_KV_BYTES"} \
  "$@" 2>&1 | tee $OUT/train.log
