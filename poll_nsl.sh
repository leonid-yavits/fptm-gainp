#!/bin/sh
# Relaunch the NSL-KDD queue if it is not running and runs remain; print one status line per run.
cd "$(dirname "$0")"
if ! pgrep -f run_queue.sh >/dev/null; then
  if [ "$(grep -c . sweep_nslkdd_ah.txt)" -gt "$(ls results_nslkdd_ah/*/*/summary.json 2>/dev/null | wc -l)" ]; then
    nohup ./run_queue.sh sweep_nslkdd_ah.txt logs_nsl_ah >> logs_queue_nsl.txt 2>&1 &
    echo "RELAUNCHED $(date +%H:%M)"
  fi
fi
for f in logs_nsl_ah/*.log; do [ -f "$f" ] && printf "%s %s | " "$(basename $f .log | sed 's/nslkdd_AH_//')" "$(tail -n1 $f | cut -c11-14)"; done
echo "done: $(ls results_nslkdd_ah/*/*/summary.json 2>/dev/null | wc -l)/10"
