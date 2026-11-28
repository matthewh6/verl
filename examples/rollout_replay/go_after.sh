#!/usr/bin/env bash
# Run replication chains after the shared verl queue's current last run: append a blank line (the queue runner idles
# on it, later appended runs wait), wait for that last run to end, run the chains, then remove the blank line.
# usage: go_after.sh <last queue run name> <chain run>:<seed> [<chain run>:<seed> ...]
H=$(dirname "$(readlink -f "$0")"); P=$HOME/projects/tutor-curriculum; R=$P/runs/replay; Q=$P/verl_rg/queue.txt; F=$P/runs/verl/failed.log; A=$1; shift
f0=$(grep -c "FAILED $A\$" $F 2>/dev/null); printf '\n' >> $Q; n=$(wc -l < $Q)
echo "$(date +%F_%T) queue held at line $n (after $A)" >> $R/go.log
release() { [ -z "$(sed -n "${n}p" $Q)" ] && sed -i "${n}d" $Q; echo "$(date +%F_%T) queue released" >> $R/go.log; }
trap release EXIT
until [ -f $P/runs/verl/$A/done ] || [ "$(grep -c "FAILED $A\$" $F 2>/dev/null)" -gt "${f0:-0}" ]; do sleep 60; done
for x in "$@"; do bash $H/chain.sh ${x%%:*} +data.custom_dataloader.seed=${x##*:} data.seed=${x##*:} >> $R/go.log 2>&1; done
