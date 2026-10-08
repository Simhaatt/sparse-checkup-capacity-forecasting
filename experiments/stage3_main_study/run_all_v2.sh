#!/bin/bash
# R4A_compute_amendment: origins 10 then 15; pooled; origin 20 only as an addendum at the very end.
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/r4
set -e
while ps -ww -eo args | grep -q "[r]4_models.py search --origin 10"; do sleep 60; done
for o in 10 15; do
  grep -q "ORIGIN_DONE $o" logs/run_all.log || { nice -n 5 python3 r4_models.py search --origin $o --workers 2; nice -n 5 python3 r4_models.py final --origin $o --workers 2; echo "ORIGIN_DONE $o"; }
done
grep -q POOLED_DONE logs/run_all.log || { nice -n 5 python3 r4_pooled.py --workers 2; echo POOLED_DONE; }
echo R4_ALL_DONE
