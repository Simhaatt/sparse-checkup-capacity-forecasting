#!/bin/bash
# Idempotent: (re)start every round-4 chain that is not running. Safe after container restarts.
cd "$(dirname "$0")"  # repo edit: was cd /home/claude/r4
running() { ps -ww -eo args | awk -v p="$1" 'index($0,p)>0 && index($0,"awk -v p")==0 {f=1} END{exit !f}'; }
if ! running "/bin/bash ./run_all_v2.sh" && ! grep -q R4_ALL_DONE logs/run_all.log; then nohup ./run_all_v2.sh >> logs/run_all.log 2>&1 & echo "started run_all"; fi
if ! running "python3 r4_ablate.py" && ! grep -q ABL_DONE logs/ablate.log 2>/dev/null; then
  nohup sh -c "python3 r4_ablate.py arima_cache && python3 r4_ablate.py snl && python3 r4_ablate.py luh && echo ABL_DONE" >> logs/ablate.log 2>&1 & echo "started ablate"; fi
if ! running "r4_chronos_ft.py" && ! grep -q CHRONOS_DONE logs/chronos_ft.log 2>/dev/null; then
  nohup sh -c "python3 -W ignore r4_chronos_ft.py snl && python3 -W ignore r4_chronos_ft.py luh && echo CHRONOS_DONE" >> logs/chronos_ft.log 2>&1 & echo "started chronos"; fi
if grep -q ABL_DONE logs/ablate.log 2>/dev/null && ! running "r4_uncert.py" && ! grep -q UNCERT_DONE logs/uncert.log 2>/dev/null; then
  nohup sh -c "python3 -W ignore r4_uncert.py && echo UNCERT_DONE" >> logs/uncert.log 2>&1 & echo "started uncert"; fi
if ! running "r4_g.py" && ! grep -q G_DONE logs/g.log 2>/dev/null; then
  nohup sh -c "([ -f tables/r4_g3_noschedule_summary.csv ] || python3 -W ignore r4_g.py g3) && ([ -f tables/r4_g4_robust_pairs.csv ] || python3 -W ignore r4_g.py g4) && while ! grep -q UNCERT_DONE logs/uncert.log 2>/dev/null; do sleep 120; done; python3 -W ignore r4_g.py g5 && echo G_DONE" >> logs/g.log 2>&1 & echo "started g"; fi
if grep -q CHRONOS_DONE logs/chronos_ft.log 2>/dev/null && ! running "CFT_FULL" && ! grep -q G7_DONE logs/g7.log 2>/dev/null; then
  nohup sh -c "CFT_FULL=1 CFT_LR=1e-5 CFT_STEPS=300 CFT_CKPT=100,200,300 CFT_TAG=_full_lr1e-5 python3 -W ignore r4_chronos_ft.py snl && echo G7_DONE" >> logs/g7.log 2>&1 & echo "started g7"; fi
