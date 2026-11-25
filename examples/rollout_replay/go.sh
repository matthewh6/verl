#!/usr/bin/env bash
# Hold the shared verl queue (a blank line after the named run makes verl_rg/chain.sh idle once that run ends),
# run the replication chain on the freed GPU, then release the queue by removing the blank line.
# usage: go.sh <queue run to hold after> <chain.sh args...>
H=$(dirname "$(readlink -f "$0")"); P=$HOME/projects/tutor-curriculum; R=$P/runs/replay; Q=$P/verl_rg/queue.txt; mkdir -p $R; A=$1; shift
n=$(grep -n -m1 "^$A " $Q | cut -d: -f1); [ -z "$n" ] && { echo "no queue line $A" >> $R/go.log; exit 1; }
[ -n "$(sed -n "$((n+1))p" $Q)" ] && sed -i "${n}G" $Q
echo "$(date +%F_%T) queue held after line $n ($A)" >> $R/go.log
release() { m=$(grep -n -m1 "^$A " $Q | cut -d: -f1); [ -n "$m" ] && [ -z "$(sed -n "$((m+1))p" $Q)" ] && sed -i "$((m+1))d" $Q; echo "$(date +%F_%T) queue released" >> $R/go.log; }
trap release EXIT
bash $H/wait_gpu.sh
bash $H/chain.sh "$@" >> $R/go.log 2>&1
