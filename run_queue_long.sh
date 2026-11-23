#!/bin/sh
# Runs sweep_long.txt two jobs at a time; skips finished runs; unfinished runs resume from ckpt.pkl.
cd "$(dirname "$0")"
mkdir -p logs_long
grep -v '^$' sweep_long.txt | while read -r line; do
  root=$(echo "$line" | sed 's/.*RESULTS_ROOT=\([^ ]*\).*/\1/'); ds=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="run_fptm.py"){print $(i+1); print $(i+2); exit}}' | paste -sd' ')
  set -- $ds
  if [ -f "$root/$1/$2_s0/summary.json" ]; then continue; fi
  echo "$line"
done | xargs -P 2 -I{} sh -c '{} >> logs_long/$(echo "{}" | awk "{for(i=1;i<=NF;i++) if(\$i==\"run_fptm.py\"){print \$(i+1)\"_\"\$(i+2); exit}}").log 2>&1'
