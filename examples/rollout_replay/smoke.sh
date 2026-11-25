#!/usr/bin/env bash
# 2-step end-to-end check: online run with archive + checkpoint, 2 student updates, small eval
set -uo pipefail
H=$(dirname "$(readlink -f "$0")")
P=${TC_P:-$HOME/projects/tutor-curriculum}; V=${TC_V:-$P/ext/verl-replay}; PY=${TC_PY:-$P/ext/verl/.venv/bin/python}
D=${TC_D:-$HOME/datasets/tutor_rg}; RB=${TC_R:-$P/runs/replay}; BASE=${TC_VMODEL:-$HOME/datasets/models/Qwen3.5-0.8B}; VMEM=${TC_VLLM_MEM:-0.35}; EMEM=${TC_EVAL_MEM:-0.4}
R=$RB
bash $H/wait_gpu.sh
rm -rf $R/smoke; SAVE_FREQ=2 bash $H/run_online.sh smoke trainer.total_training_steps=2 trainer.test_freq=-1 trainer.val_before_train=False > /dev/null 2>&1
$(dirname $PY)/ray stop --force >/dev/null 2>&1; sleep 5
ls -la $R/smoke/archive $R/smoke/ckpt/*/ $R/smoke/ckpt/*/* 2>&1 | head -40
export PYTHONPATH=$P
for loss in sft pg_is; do $PY $H/replay_train.py --archive $R/smoke/archive --out $R/smoke/stu_$loss --model $BASE --loss $loss 2>&1 | grep -v Warning | tail -8; done
$PY $H/eval_ckpt.py --ckpt $R/smoke/stu_sft/ckpt_0002 --out $R/smoke/eval_stu.json --n 2 --mem $EMEM --val $D/val.parquet --project $P 2>&1 | tail -30
C=$(ls -d $R/smoke/ckpt/global_step_2/actor/huggingface 2>/dev/null || ls -d $R/smoke/ckpt/global_step_2/actor 2>/dev/null)
$PY $H/eval_ckpt.py --ckpt $C --out $R/smoke/eval_on.json --n 2 --mem $EMEM --val $D/val.parquet --project $P 2>&1 | tail -30
echo SMOKE_DONE
