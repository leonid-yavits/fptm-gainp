#!/bin/sh
# Usage: run_queue.sh <sweep file> <log dir>. Runs two jobs at a time, skips finished runs,
# and resumes unfinished runs from ckpt.pkl.
cd "$(dirname "$0")"
SWEEP="$1"; export LOGDIR="$2"; mkdir -p "$LOGDIR"
grep -v '^$' "$SWEEP" | while read -r line; do
  root=$(echo "$line" | sed 's/.*RESULTS_ROOT=\([^ ]*\).*/\1/')
  dt=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="run_fptm.py"){print $(i+1)"/"$(i+2); exit}}')
  [ -f "$root/${dt}_s0/summary.json" ] && continue
  echo "$line"
done | xargs -P 2 -I{} sh -c '{} >> "$LOGDIR/$(echo "{}" | awk "{for(i=1;i<=NF;i++) if(\$i==\"run_fptm.py\"){print \$(i+1)\"_\"\$(i+2); exit}}").log" 2>&1'
