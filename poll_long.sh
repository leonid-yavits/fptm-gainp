#!/bin/sh
# Relaunch the queue if no trainer is running; print a one-line status per run.
cd "$(dirname "$0")"
if ! pgrep -f run_queue_long.sh >/dev/null; then
  if [ "$(grep -c . sweep_long.txt)" -gt "$(ls results_*_long/*/*/summary.json 2>/dev/null | wc -l)" ]; then
    nohup ./run_queue_long.sh >> logs_queue_long.txt 2>&1 &
    echo "RELAUNCHED $(date +%H:%M)"
  fi
fi
for f in logs_long/*.log; do printf "%s %s\n" "$(basename $f .log)" "$(tail -n1 $f | cut -c1-24)"; done
echo "done: $(ls results_*_long/*/*/summary.json 2>/dev/null | wc -l)/18"
