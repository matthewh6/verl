#!/usr/bin/env bash
# Full replication chain for one online run: online GRPO with archive -> evals -> students (replay) -> evals.
# usage: chain.sh <run name> [online-run hydra overrides...]     env: N_EVAL (samples per question, default 16)
set -uo pipefail
RUN=$1; shift
H=$(dirname "$(readlink -f "$0")")
P=${TC_P:-$HOME/projects/tutor-curriculum}; V=${TC_V:-$P/ext/verl-replay}; PY=${TC_PY:-$P/ext/verl/.venv/bin/python}
D=${TC_D:-$HOME/datasets/tutor_rg}; RB=${TC_R:-$P/runs/replay}; BASE=${TC_VMODEL:-$HOME/datasets/models/Qwen3.5-0.8B}; VMEM=${TC_VLLM_MEM:-0.35}; EMEM=${TC_EVAL_MEM:-0.4}
R=$RB/$RUN; N=${N_EVAL:-16}; T=${T_STEPS:-120}; RAYBIN=$(dirname $PY)/ray
export PYTHONPATH=$P; mkdir -p $R/evals
say() { echo "$(date +%F_%T) $*" | tee -a $R/chain.log; }
ev() {  # ev <group> <step> <ckpt dir>
  [ -f $R/evals/$1/step_$(printf %04d $2).json ] && return 0
  bash $H/wait_gpu.sh; mkdir -p $R/evals/$1
  $PY $H/eval_ckpt.py --ckpt $3 --out $R/evals/$1/step_$(printf %04d $2).json --n $N --mem $EMEM --val $D/val.parquet --project $P > $R/evals/$1/log_$2.txt 2>&1 || say "EVAL FAILED $1 $2"
}
student() {  # student <name> <save-at list> <replay_train args...>
  local name=$1 saves=$2; shift 2
  local last=${saves##*,}
  [ -f $R/evals/$name/step_$(printf %04d $last).json ] && return 0
  bash $H/wait_gpu.sh; say "student $name start"
  rm -rf $R/stu/$name
  $PY $H/replay_train.py --archive $R/archive --out $R/stu/$name --model $BASE --save-at $saves "$@" > $R/stu_$name.log 2>&1 || { say "STUDENT FAILED $name"; return 1; }
  for s in ${saves//,/ }; do ev $name $s $R/stu/$name/ckpt_$(printf %04d $s); [ $s != $last ] && rm -rf $R/stu/$name/ckpt_$(printf %04d $s); done
  say "student $name done"
}
if ! grep -q "step:$T " $R/train.log 2>/dev/null; then
  bash $H/wait_gpu.sh; say "online start"; rm -rf $R/archive $R/ckpt $R/outcomes*
  bash $H/run_online.sh $RUN trainer.total_training_steps=$T "$@" > /dev/null 2>&1
  $RAYBIN stop --force >/dev/null 2>&1; sleep 5
  grep -q "step:$T " $R/train.log || { say "ONLINE FAILED"; exit 1; }
fi
say "online done"
ALL=$(seq -s, 20 20 $T); HALF=$((T/2)),$T
ev online 0 $BASE
for s in ${ALL//,/ }; do ev online $s $R/ckpt/global_step_$s/actor/huggingface; done
student sft_chrono $ALL --loss sft --order chrono
student pg_chrono $ALL --loss pg --order chrono
student sft_last1 $HALF --loss sft --order last:1 --updates $T
student pgis_chrono $ALL --loss pg_is --order chrono
student sft_shuffle $HALF --loss sft --order shuffle
student sft_reverse $HALF --loss sft --order reverse
student sft_last5 $HALF --loss sft --order last:5 --updates $T
student sft_last20 $HALF --loss sft --order last:20 --updates $T
student sftpos_chrono $HALF --loss sft_pos --order chrono
say CHAIN_DONE
