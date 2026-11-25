#!/usr/bin/env bash
# block until no verl trainer of this user is running (set TC_NOWAIT=1 on a machine where the GPU is not shared with a verl queue)
[ -n "${TC_NOWAIT:-}" ] && exit 0
while pgrep -f "verl.trainer.main_ppo" >/dev/null; do sleep 20; done; sleep 15
